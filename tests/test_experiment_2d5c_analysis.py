import math
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c_analysis as analysis  # noqa: E402


def condition(values, targets=None):
    values = [float(value) for value in values]
    targets = len(values) * 1024 if targets is None else int(targets)
    return {
        "per_sequence_ce": values,
        "per_sequence_nll": [value * 1024 for value in values],
        "validation_loss": sum(values) / len(values),
        "validation_targets": targets,
        "paired_sequences": len(values),
    }


def family_conditions(family, real, b3_off, b3_shuffle, b5_off, b5_shuffle, both_off, both_shuffle):
    prefix = family.upper()
    return {
        f"{prefix}_ALL_REAL": condition(real),
        f"{prefix}_B3_RECURRENCE_OFF": condition(b3_off),
        f"{prefix}_B3_SHUFFLED": condition(b3_shuffle),
        f"{prefix}_B5_RECURRENCE_OFF": condition(b5_off),
        f"{prefix}_B5_SHUFFLED": condition(b5_shuffle),
        f"{prefix}_B3_B5_BOTH_OFF": condition(both_off),
        f"{prefix}_B3_B5_BOTH_SHUFFLED": condition(both_shuffle),
    }


def ci(point=0.0, lower=-0.1, upper=0.1):
    return {"point_estimate": point, "lower_95": lower, "upper_95": upper}


def classification_rows():
    names = (
        "architecture_fixed_minus_c",
        "architecture_c_minus_fixed_penalty",
        "c_combined_off_gain",
        "c_combined_sequence_gap",
        "combined_off_gain_lift",
        "combined_sequence_gap_lift",
    )
    return {name: ci() for name in names}


def fake_cache_audit(b3_history, b5_history, *, other_history=5):
    cache_rows = []
    cache_logical = 0
    for block in range(1, 13):
        history = b3_history if block == 3 else b5_history if block == 5 else other_history
        tensor_bytes = history * 768 * 2
        tensor = {
            "shape": [1, history, 768],
            "dtype": "torch.bfloat16",
            "expected_bytes": tensor_bytes,
            "actual_bytes": tensor_bytes,
            "exact": True,
        }
        cache_rows.append({"block": block, "key": dict(tensor), "value": dict(tensor)})
        cache_logical += 2 * tensor_bytes
    ring_rows = {}
    ring_logical = 0
    for name in ("h7", "h8", "h10", "h12"):
        tensor_bytes = 1023 * 768 * 2
        ring_rows[name] = {
            "shape": [1, 1023, 768],
            "dtype": "torch.bfloat16",
            "expected_bytes": tensor_bytes,
            "actual_bytes": tensor_bytes,
            "exact": True,
        }
        ring_logical += tensor_bytes
    total = cache_logical + ring_logical
    return {
        "b3_historical_local_kv": b3_history,
        "b5_historical_local_kv": b5_history,
        "cache_storage": cache_rows,
        "ring_storage": ring_rows,
        "logical_payload_bytes": total,
        "actual_unique_storage_bytes": total,
        "storage_alias_free": True,
        "physical_storage_exact": True,
        "passed": True,
    }


class Experiment2D5CAnalysisTests(unittest.TestCase):
    def test_final_contrasts_and_pressure_lifts_are_directly_paired(self):
        c = family_conditions(
            "C",
            [3.00, 3.10, 3.20, 3.30],
            [3.04, 3.14, 3.24, 3.34],
            [3.03, 3.13, 3.23, 3.33],
            [3.02, 3.12, 3.22, 3.32],
            [3.01, 3.11, 3.21, 3.31],
            [3.07, 3.17, 3.27, 3.37],
            [3.05, 3.15, 3.25, 3.35],
        )
        fixed = family_conditions(
            "F",
            [2.99, 3.09, 3.19, 3.29],
            [3.00, 3.10, 3.20, 3.30],
            [2.995, 3.095, 3.195, 3.295],
            [3.00, 3.10, 3.20, 3.30],
            [2.995, 3.095, 3.195, 3.295],
            [3.01, 3.11, 3.21, 3.31],
            [3.00, 3.10, 3.20, 3.30],
        )
        validated = analysis.validate_final_condition_data(
            c, fixed, expected_sequences=4, expected_targets=4096
        )
        vectors = analysis.build_final_contrast_vectors(validated["arrays"])
        self.assertTrue(all(abs(value + 0.01) < 1e-12 for value in vectors["architecture_fixed_minus_c"]))
        self.assertTrue(all(abs(value - 0.03) < 1e-12 for value in vectors["b3_off_gain_lift"]))
        self.assertTrue(all(abs(value - 0.05) < 1e-12 for value in vectors["combined_off_gain_lift"]))
        self.assertTrue(validated["passed"])
        self.assertEqual(validated["condition_count"], 14)

    def test_shared_index_bootstrap_is_deterministic_and_reverse_contrast_mirrors_ci(self):
        vectors = {
            "forward": [0.1, -0.2, 0.3, 0.0, 0.4],
            "reverse": [-0.1, 0.2, -0.3, 0.0, -0.4],
            "derived": [0.2, 0.1, -0.1, 0.4, 0.0],
        }
        first = analysis.paired_bootstrap_contrasts(
            vectors, seed=2_026_083_003, resamples=2_000, chunk_size=17
        )
        second = analysis.paired_bootstrap_contrasts(
            vectors, seed=2_026_083_003, resamples=2_000, chunk_size=113
        )
        self.assertEqual(
            first["shared_resample_index_stream_sha256"],
            second["shared_resample_index_stream_sha256"],
        )
        self.assertTrue(first["same_resampled_sequence_indices_for_all_contrasts"])
        forward = first["contrasts"]["forward"]
        reverse = first["contrasts"]["reverse"]
        self.assertAlmostEqual(forward["point_estimate"], -reverse["point_estimate"], places=15)
        self.assertAlmostEqual(forward["lower_95"], -reverse["upper_95"], places=15)
        self.assertAlmostEqual(forward["upper_95"], -reverse["lower_95"], places=15)
        self.assertEqual(forward["positive_per_sequence_differences"], 3)
        self.assertEqual(forward["zero_per_sequence_differences"], 1)
        expected_se = statistics_stderr(vectors["forward"])
        self.assertAlmostEqual(forward["paired_standard_error"], expected_se, places=15)

    def test_adaptation_recovery_uses_ratio_of_paired_cluster_means(self):
        parent = [3.0, 3.1, 3.2, 3.3]
        c0 = [3.02, 3.12, 3.22, 3.32]
        fixed = [2.99, 3.09, 3.19, 3.29]
        c191 = [3.0, 3.1, 3.2, 3.3]
        result = analysis.adaptation_recovery_summary(
            parent, c0, c191, fixed, resamples=1_000, chunk_size=31
        )
        self.assertAlmostEqual(result["initial_shock"]["point_estimate"], 0.02)
        self.assertAlmostEqual(result["final_matched_core_penalty"]["point_estimate"], 0.01)
        self.assertAlmostEqual(result["recovery_fraction"]["point_estimate"], 0.5)
        self.assertTrue(result["recovery_fraction"]["defined"])
        self.assertTrue(result["same_resampled_sequence_indices"])

        undefined = analysis.adaptation_recovery_summary(
            parent, parent, c191, fixed, resamples=100, chunk_size=9
        )
        self.assertIsNone(undefined["recovery_fraction"]["point_estimate"])
        self.assertFalse(undefined["recovery_fraction"]["defined"])

    def test_classification_tree_all_named_branches_and_strict_boundaries(self):
        invalid = analysis.classification_decision({}, audit_passed=False)
        self.assertEqual(invalid["classification"], "INVALID — NO SCIENTIFIC CONCLUSION")

        strong = classification_rows()
        strong["architecture_fixed_minus_c"] = ci(0.0004, 0.0001, 0.0007)
        strong["architecture_c_minus_fixed_penalty"] = ci(-0.0004, -0.0007, -0.0001)
        for name in (
            "c_combined_off_gain",
            "c_combined_sequence_gap",
            "combined_off_gain_lift",
            "combined_sequence_gap_lift",
        ):
            strong[name] = ci(0.001, 0.0002, 0.002)
        decision = analysis.classification_decision(strong, audit_passed=True)
        self.assertEqual(
            decision["classification"],
            "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED WITH ABSOLUTE CE IMPROVEMENT",
        )

        partial = classification_rows()
        partial["architecture_fixed_minus_c"] = ci(-0.002, -0.003, -0.0012)
        partial["architecture_c_minus_fixed_penalty"] = ci(0.002, 0.0012, 0.003)
        for name in (
            "c_combined_off_gain",
            "c_combined_sequence_gap",
            "combined_off_gain_lift",
            "combined_sequence_gap_lift",
        ):
            partial[name] = ci(0.001, 0.0001, 0.002)
        self.assertEqual(
            analysis.classification_decision(
                partial, audit_passed=True, meaningful_recovery=True
            )["classification"],
            "W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL",
        )

        recovery = dict(strong)
        recovery["combined_sequence_gap_lift"] = ci(0.0, -0.0002, 0.0002)
        self.assertEqual(
            analysis.classification_decision(recovery, audit_passed=True)["decision_branch"],
            "C_RECOVERY_WITHOUT_ESTABLISHED_DEPENDENCE",
        )

        path_only = classification_rows()
        path_only["architecture_c_minus_fixed_penalty"] = ci(0.0005, 0.0001, 0.0012)
        path_only["architecture_fixed_minus_c"] = ci(-0.0005, -0.0012, -0.0001)
        path_only["c_combined_off_gain"] = ci(0.001, 0.0001, 0.002)
        path_only["c_combined_sequence_gap"] = ci(0.0, -0.0002, 0.0002)
        self.assertEqual(
            analysis.classification_decision(path_only, audit_passed=True)["decision_branch"],
            "D_PATH_UTILITY_WITHOUT_ALIGNED_MEMORY",
        )

        failure = classification_rows()
        failure["architecture_fixed_minus_c"] = ci(-0.002, -0.003, -0.0012)
        failure["architecture_c_minus_fixed_penalty"] = ci(0.002, 0.0012, 0.003)
        self.assertEqual(
            analysis.classification_decision(failure, audit_passed=True)["decision_branch"],
            "E_PERSISTENT_FAILURE",
        )

        boundary = classification_rows()
        boundary["architecture_fixed_minus_c"] = ci(-0.001, -0.002, 0.0)
        boundary["architecture_c_minus_fixed_penalty"] = ci(0.001, 0.0, 0.001)
        for name in (
            "c_combined_off_gain",
            "c_combined_sequence_gap",
            "combined_off_gain_lift",
            "combined_sequence_gap_lift",
        ):
            boundary[name] = ci(0.001, 0.0, 0.002)
        unresolved = analysis.classification_decision(boundary, audit_passed=True)
        self.assertEqual(unresolved["decision_branch"], "F_MIXED_OR_BOUNDARY_CROSSING")
        self.assertFalse(unresolved["flags"]["practical_noninferiority_established"])

    def test_bf16_logical_and_physical_state_comparison_is_component_exact(self):
        fixed = fake_cache_audit(31, 63)
        current = fake_cache_audit(1, 1)
        report = analysis.compare_bf16_persistent_state(fixed, current)
        self.assertEqual(report["logical"]["reduction_bytes"], 282_624)
        self.assertEqual(report["logical"]["reduction_kib"], 276.0)
        self.assertEqual(
            report["allocated_unique_storage"]["reduction_bytes"], 282_624
        )
        self.assertTrue(report["checks"]["only_b3_b5_local_kv_changed"])
        self.assertTrue(report["checks"]["recurrent_rings_allocated_unchanged"])
        self.assertTrue(report["passed"])

        preallocated = fake_cache_audit(1, 1)
        preallocated["actual_unique_storage_bytes"] = fixed["actual_unique_storage_bytes"]
        for block in (3, 5):
            preallocated["cache_storage"][block - 1]["key"]["actual_bytes"] = (
                fixed["cache_storage"][block - 1]["key"]["actual_bytes"]
            )
            preallocated["cache_storage"][block - 1]["value"]["actual_bytes"] = (
                fixed["cache_storage"][block - 1]["value"]["actual_bytes"]
            )
        preallocated["physical_storage_exact"] = False
        preallocated["passed"] = False
        measured = analysis.compare_bf16_persistent_state(fixed, preallocated)
        self.assertEqual(measured["logical"]["reduction_bytes"], 282_624)
        self.assertEqual(measured["allocated_unique_storage"]["reduction_bytes"], 0)
        self.assertFalse(measured["passed"])

    def test_longitudinal_mechanism_interactions(self):
        values = {
            "all_real": 3.0,
            "b3_off": 3.01,
            "b3_shuffled": 3.008,
            "b5_off": 3.02,
            "b5_shuffled": 3.012,
            "b3_b5_off": 3.04,
            "b3_b5_shuffled": 3.03,
        }
        row = analysis.mechanism_metrics(values)
        self.assertAlmostEqual(row["combined_recurrent_gain"], 0.04)
        self.assertAlmostEqual(
            row["off_interaction_combined_minus_individual_sum"], 0.01
        )
        milestones = {update: values for update in (0, 48, 96, 144, 191)}
        longitudinal = analysis.longitudinal_core_summary(milestones)
        self.assertTrue(longitudinal["passed"])
        self.assertEqual(list(longitudinal["c"]), ["0", "48", "96", "144", "191"])

    def test_final_audit_and_report_validation_fail_closed(self):
        checks = {name: True for name in analysis.REQUIRED_FINAL_AUDIT_CHECKS}
        valid = analysis.validate_final_audit({"checks": checks, "passed": True})
        self.assertTrue(valid["classification_allowed"])
        broken = dict(checks)
        broken["causality_tests_passed"] = False
        invalid = analysis.validate_final_audit({"checks": broken, "passed": False})
        self.assertEqual(invalid["status"], "INVALID — NO SCIENTIFIC CONCLUSION")

        bootstrap = {
            "contrasts": {name: ci() for name in analysis.FINAL_CONTRAST_TERMS}
        }
        summary = {
            "classification": "W2/W2 REPRESENTATION-PRESSURE RESULT UNRESOLVED",
            "fixed_all_real_ce": 3.0,
            "c_all_real_ce": 3.001,
            "bootstrap": bootstrap,
            "bf16_persistent_state": {
                "logical": {},
                "allocated_unique_storage": {},
            },
            "final_checkpoint": {"path": "/tmp/final.pt", "sha256": "a" * 64},
            "git_branch": "branch",
            "git_commit": "commit",
            "git_tag": "tag",
            "runpod_status": "stopped",
        }
        report = analysis.validate_final_report_data(
            summary,
            {"checks": checks, "passed": True},
            analysis.REQUIRED_ARTIFACTS,
            report_text="STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS\n",
        )
        self.assertTrue(report["passed"])


def statistics_stderr(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


if __name__ == "__main__":
    unittest.main()
