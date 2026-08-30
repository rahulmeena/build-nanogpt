#!/usr/bin/env python3
"""Deterministic post-evaluation analysis helpers for Experiment 2D5C.

This module is deliberately independent of Torch and of the training driver.
It implements only transformations of already-collected evaluation and cache
audit data:

* the preregistered shared-index paired sequence bootstrap;
* the architecture, mechanism, and pressure-lift contrasts;
* longitudinal mechanism and adaptation-recovery summaries;
* the frozen classification tree and next-experiment recommendation logic;
* BF16 persistent-state accounting from deployment cache audits; and
* fail-closed validation of final audit/report inputs.

The large-panel routines require NumPy because the accepted project bootstrap
uses ``numpy.random.default_rng``.  NumPy is imported lazily so simple audit and
classification tooling remains importable in lightweight environments.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


BOOTSTRAP_SEED = 2_026_083_003
BOOTSTRAP_RESAMPLES = 50_000
EXPECTED_LARGE_SEQUENCES = 2_048
EXPECTED_LARGE_TARGETS = 2_097_152
NONINFERIORITY_MARGIN_CE = 0.001
EXPECTED_BF16_LOCAL_KV_REDUCTION_BYTES = 282_624
EXPECTED_BF16_LOCAL_KV_REDUCTION_KIB = 276.0


CONTROL_SUFFIXES = (
    "ALL_REAL",
    "B3_RECURRENCE_OFF",
    "B3_SHUFFLED",
    "B5_RECURRENCE_OFF",
    "B5_SHUFFLED",
    "B3_B5_BOTH_OFF",
    "B3_B5_BOTH_SHUFFLED",
)
FINAL_CONDITIONS = tuple(
    f"{family}_{suffix}" for family in ("C", "F") for suffix in CONTROL_SUFFIXES
)

_CONTROL_ALIASES = {
    "all_real": "ALL_REAL",
    "b3_off": "B3_RECURRENCE_OFF",
    "b3_recurrence_off": "B3_RECURRENCE_OFF",
    "b3_shuffled": "B3_SHUFFLED",
    "b5_off": "B5_RECURRENCE_OFF",
    "b5_recurrence_off": "B5_RECURRENCE_OFF",
    "b5_shuffled": "B5_SHUFFLED",
    "b3_b5_off": "B3_B5_BOTH_OFF",
    "b3_b5_both_off": "B3_B5_BOTH_OFF",
    "both_off": "B3_B5_BOTH_OFF",
    "b3_b5_shuffled": "B3_B5_BOTH_SHUFFLED",
    "b3_b5_both_shuffled": "B3_B5_BOTH_SHUFFLED",
    "both_shuffled": "B3_B5_BOTH_SHUFFLED",
}


# Each contrast is a linear combination of paired per-sequence CE values.
# Positive A-B means B has lower CE and is better, exactly as preregistered.
FINAL_CONTRAST_TERMS: dict[str, tuple[tuple[str, float], ...]] = {
    "architecture_fixed_minus_c": (
        ("F_ALL_REAL", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "architecture_c_minus_fixed_penalty": (
        ("C_ALL_REAL", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "c_b3_off_gain": (
        ("C_B3_RECURRENCE_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "c_b3_sequence_gap": (
        ("C_B3_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "c_b5_off_gain": (
        ("C_B5_RECURRENCE_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "c_b5_sequence_gap": (
        ("C_B5_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "c_combined_off_gain": (
        ("C_B3_B5_BOTH_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "c_combined_sequence_gap": (
        ("C_B3_B5_BOTH_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
    ),
    "f_b3_off_gain": (
        ("F_B3_RECURRENCE_OFF", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "f_b3_sequence_gap": (
        ("F_B3_SHUFFLED", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "f_b5_off_gain": (
        ("F_B5_RECURRENCE_OFF", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "f_b5_sequence_gap": (
        ("F_B5_SHUFFLED", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "f_combined_off_gain": (
        ("F_B3_B5_BOTH_OFF", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    "f_combined_sequence_gap": (
        ("F_B3_B5_BOTH_SHUFFLED", 1.0),
        ("F_ALL_REAL", -1.0),
    ),
    # Difference-in-differences are constructed at the paired sequence level,
    # never by subtracting independently bootstrapped aggregates.
    "b3_off_gain_lift": (
        ("C_B3_RECURRENCE_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B3_RECURRENCE_OFF", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
    "b3_sequence_gap_lift": (
        ("C_B3_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B3_SHUFFLED", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
    "b5_off_gain_lift": (
        ("C_B5_RECURRENCE_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B5_RECURRENCE_OFF", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
    "b5_sequence_gap_lift": (
        ("C_B5_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B5_SHUFFLED", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
    "combined_off_gain_lift": (
        ("C_B3_B5_BOTH_OFF", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B3_B5_BOTH_OFF", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
    "combined_sequence_gap_lift": (
        ("C_B3_B5_BOTH_SHUFFLED", 1.0),
        ("C_ALL_REAL", -1.0),
        ("F_B3_B5_BOTH_SHUFFLED", -1.0),
        ("F_ALL_REAL", 1.0),
    ),
}


CLASSIFICATIONS = (
    "INVALID — NO SCIENTIFIC CONCLUSION",
    "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED",
    "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED WITH ABSOLUTE CE IMPROVEMENT",
    "W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL",
    "W2/W2 RECOVERS WITHOUT ESTABLISHED B3/B5 RECURRENT DEPENDENCE",
    "RECURRENT PATH UTILITY ESTABLISHED; ALIGNED SEQUENCE MEMORY NOT ESTABLISHED",
    "W2/W2 PERSISTENT DEGRADATION; DEEP-RECURRENT SUBSTITUTION NOT SUPPORTED",
    "W2/W2 REPRESENTATION-PRESSURE RESULT UNRESOLVED",
)


REQUIRED_FINAL_AUDIT_CHECKS = (
    "source_checkpoint_sha_exact",
    "c_started_from_2d3a_source",
    "fixed_control_checkpoint_sha_exact",
    "exactly_one_newly_trained_arm",
    "fixed_control_optimizer_steps_zero",
    "c_optimizer_steps_exact",
    "c_new_targets_exact",
    "final_global_update_exact",
    "final_cumulative_targets_exact",
    "replay_191_batches_exact",
    "replay_chain_hash_exact",
    "initial_terminal_loader_cursor_hashes_exact",
    "pass_cadence_exact",
    "optimizer_continuity",
    "scheduler_continuity",
    "rng_continuity",
    "midpoint_fresh_process_restart_success",
    "parameter_count_unchanged",
    "state_dict_keys_exact",
    "fixed_writers_preserved",
    "b3_b5_lag_coverage_exact",
    "local_recurrent_nonoverlap",
    "causality_tests_passed",
    "deployment_cache_tests_passed",
    "control_specificity_tests_passed",
    "ce_only_objective",
    "attached_writer_gradients",
    "analysis_input_identities_exact",
    "secondary_parallel_c0_c96_c191_completed",
    "all_required_core_conditions_completed",
    "all_14_large_conditions_completed",
    "per_sequence_pairing_intact",
    "large_targets_exact_every_condition",
    "historical_panel_disjointness_checked_where_possible",
    "fourteen_condition_evaluation_not_reduced",
    "memory_accounting_completed",
    "final_checkpoint_strict_reopen_passed",
    "remote_local_checkpoint_sha_match",
    "git_branch_commit_tag_pushed_verified",
    "worktree_clean",
    "no_a_b_or_250m_training",
)


CRITICAL_INVALID_AUDIT_CHECKS = frozenset(
    {
        "source_checkpoint_sha_exact",
        "c_started_from_2d3a_source",
        "fixed_control_checkpoint_sha_exact",
        "exactly_one_newly_trained_arm",
        "fixed_control_optimizer_steps_zero",
        "c_optimizer_steps_exact",
        "c_new_targets_exact",
        "final_global_update_exact",
        "final_cumulative_targets_exact",
        "replay_191_batches_exact",
        "replay_chain_hash_exact",
        "initial_terminal_loader_cursor_hashes_exact",
        "pass_cadence_exact",
        "causality_tests_passed",
        "control_specificity_tests_passed",
        "analysis_input_identities_exact",
        "secondary_parallel_c0_c96_c191_completed",
        "all_14_large_conditions_completed",
        "per_sequence_pairing_intact",
        "large_targets_exact_every_condition",
        "fourteen_condition_evaluation_not_reduced",
        "no_a_b_or_250m_training",
    }
)


REQUIRED_ARTIFACTS = (
    "SCOPE_LOCK.json",
    "SOURCE_PROVENANCE.json",
    "FIXED_CONTROL_PROVENANCE.json",
    "ENVIRONMENT_MANIFEST.json",
    "ARCHITECTURE_MANIFEST_C.json",
    "DATA_REPLAY_LEDGER.jsonl",
    "DATA_REPLAY_AUDIT.json",
    "PANEL_MANIFEST_CORE.json",
    "PANEL_MANIFEST_LARGE.json",
    "PREFLIGHT_TESTS.json",
    "DISPOSABLE_SMOKE_REPORT.json",
    "INITIAL_GEOMETRY_SHOCK.json",
    "TRAINING_LOG.jsonl",
    "MILESTONE_CHECKPOINTS.json",
    "MIDPOINT_RESTART_AUDIT.json",
    "TRUE_INCREMENTAL_LONGITUDINAL_CORE.json",
    "REPRESENTATION_PRESSURE_DIAGNOSTICS.json",
    "LARGE_FINAL_PER_SEQUENCE_LOSSES.json",
    "LARGE_FINAL_BOOTSTRAP.json",
    "BF16_PERSISTENT_STATE_AUDIT.json",
    "FINAL_CHECKPOINT_PROVENANCE.json",
    "FINAL_AUDIT.json",
    "EXPERIMENT_2D5C_FINAL_REPORT.md",
)


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - deployment has NumPy
        raise RuntimeError(
            "2D5C paired bootstrap requires NumPy and numpy.random.default_rng"
        ) from exc
    return np


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sequence_values(value: Any, label: str) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("per_sequence_ce", "per_sequence_losses", "losses"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(f"{label} has no per-sequence CE array")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a per-sequence array")
    result = [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _condition_target_count(row: Any) -> int | None:
    if not isinstance(row, Mapping):
        return None
    for key in ("validation_targets", "targets", "target_count"):
        if key in row:
            value = int(row[key])
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            return value
    return None


def _normalize_family_conditions(
    family: str, conditions: Mapping[str, Any]
) -> dict[str, Any]:
    family = str(family).upper()
    if family not in ("C", "F"):
        raise ValueError("family must be C or F")
    result: dict[str, Any] = {}
    unknown: list[str] = []
    for source_name, row in conditions.items():
        name = str(source_name)
        upper = name.upper()
        prefix = f"{family}_"
        if upper.startswith(prefix) and upper[len(prefix) :] in CONTROL_SUFFIXES:
            suffix = upper[len(prefix) :]
        elif name.lower() in _CONTROL_ALIASES:
            suffix = _CONTROL_ALIASES[name.lower()]
        else:
            unknown.append(name)
            continue
        canonical = f"{family}_{suffix}"
        if canonical in result:
            raise ValueError(f"duplicate condition alias for {canonical}")
        result[canonical] = row
    expected = {f"{family}_{suffix}" for suffix in CONTROL_SUFFIXES}
    missing = sorted(expected - set(result))
    if unknown or missing:
        raise ValueError(
            f"{family} conditions are not the exact seven controls; "
            f"missing={missing}, unknown={sorted(unknown)}"
        )
    return result


def validate_final_condition_data(
    c_conditions: Mapping[str, Any],
    fixed_conditions: Mapping[str, Any],
    *,
    expected_sequences: int = EXPECTED_LARGE_SEQUENCES,
    expected_targets: int = EXPECTED_LARGE_TARGETS,
    aggregate_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Validate and normalize all fourteen paired large-panel conditions.

    The returned ``arrays`` are plain Python float lists so the validation
    artifact itself does not depend on NumPy serialization.
    """

    expected_sequences = int(expected_sequences)
    expected_targets = int(expected_targets)
    if expected_sequences <= 0 or expected_targets <= 0:
        raise ValueError("expected sequence/target counts must be positive")
    if expected_targets % expected_sequences:
        raise ValueError("targets must divide evenly across paired sequences")
    canonical = {
        **_normalize_family_conditions("C", c_conditions),
        **_normalize_family_conditions("F", fixed_conditions),
    }
    arrays: dict[str, list[float]] = {}
    rows: dict[str, dict[str, Any]] = {}
    targets_per_sequence = expected_targets // expected_sequences
    for name in FINAL_CONDITIONS:
        source = canonical[name]
        values = _sequence_values(source, name)
        if len(values) != expected_sequences:
            raise ValueError(
                f"{name} has {len(values)} sequences, expected {expected_sequences}"
            )
        declared_sequences = None
        if isinstance(source, Mapping):
            for key in ("paired_sequences", "sequence_count", "sequences"):
                if key in source:
                    declared_sequences = int(source[key])
                    break
        if declared_sequences is not None and declared_sequences != expected_sequences:
            raise ValueError(f"{name} declared sequence count is wrong")
        targets = _condition_target_count(source)
        if targets is None or targets != expected_targets:
            raise ValueError(
                f"{name} target count is {targets!r}, expected {expected_targets}"
            )

        calculated_ce = math.fsum(values) / expected_sequences
        aggregate_ce = calculated_ce
        aggregate_exact = True
        if isinstance(source, Mapping):
            for key in ("validation_loss", "ce", "loss"):
                if key in source:
                    aggregate_ce = _finite_float(source[key], f"{name}.{key}")
                    aggregate_exact = math.isclose(
                        aggregate_ce,
                        calculated_ce,
                        rel_tol=aggregate_tolerance,
                        abs_tol=aggregate_tolerance,
                    )
                    if not aggregate_exact:
                        raise ValueError(
                            f"{name} aggregate CE disagrees with per-sequence CE"
                        )
                    break
            nll = source.get("per_sequence_nll")
            if nll is not None:
                nll_values = _sequence_values(nll, f"{name}.per_sequence_nll")
                if len(nll_values) != expected_sequences:
                    raise ValueError(f"{name} per-sequence NLL length mismatch")
                maximum = max(
                    abs(loss - total / targets_per_sequence)
                    for loss, total in zip(values, nll_values)
                )
                if maximum > aggregate_tolerance:
                    raise ValueError(f"{name} CE/NLL pairing is inconsistent")
        arrays[name] = values
        rows[name] = {
            "ce": aggregate_ce,
            "calculated_mean_per_sequence_ce": calculated_ce,
            "sequence_count": expected_sequences,
            "target_count": expected_targets,
            "targets_per_sequence": targets_per_sequence,
            "aggregate_matches_per_sequence": aggregate_exact,
        }

    return {
        "schema": "exp2d5c_final_condition_validation_v1",
        "condition_order": list(FINAL_CONDITIONS),
        "condition_count": len(canonical),
        "sequence_count": expected_sequences,
        "target_count_per_condition": expected_targets,
        "targets_per_sequence": targets_per_sequence,
        "same_sequence_order_required": True,
        "conditions": rows,
        "arrays": arrays,
        "checks": {
            "exact_fourteen_conditions": len(canonical) == 14,
            "exact_sequence_count_every_condition": True,
            "exact_target_count_every_condition": True,
            "finite_per_sequence_ce": True,
            "aggregate_matches_per_sequence": True,
        },
        "passed": True,
    }


def build_final_contrast_vectors(
    condition_arrays: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Build every preregistered contrast directly per paired sequence."""

    missing = sorted(set(FINAL_CONDITIONS) - set(condition_arrays))
    if missing:
        raise ValueError(f"missing final conditions: {missing}")
    arrays = {
        name: _sequence_values(condition_arrays[name], name) for name in FINAL_CONDITIONS
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all condition arrays must have identical sequence counts")
    result: dict[str, list[float]] = {}
    for contrast, terms in FINAL_CONTRAST_TERMS.items():
        result[contrast] = [
            math.fsum(coefficient * arrays[condition][index] for condition, coefficient in terms)
            for index in range(next(iter(lengths)))
        ]
    return result


def _shared_bootstrap_distributions(
    vectors: Mapping[str, Sequence[float]],
    *,
    seed: int,
    resamples: int,
    chunk_size: int,
):
    np = _numpy()
    if not vectors:
        raise ValueError("at least one contrast vector is required")
    names = sorted(str(name) for name in vectors)
    arrays = {
        name: np.asarray(_sequence_values(vectors[name], name), dtype=np.float64)
        for name in names
    }
    lengths = {int(value.size) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("contrast vectors must have one shared sequence count")
    sequence_count = next(iter(lengths))
    resamples = int(resamples)
    chunk_size = int(chunk_size)
    if resamples <= 0 or chunk_size <= 0:
        raise ValueError("resamples and chunk_size must be positive")
    rng = np.random.default_rng(int(seed))
    distributions = {
        name: np.empty(resamples, dtype=np.float64) for name in names
    }
    index_digest = hashlib.sha256()
    index_digest.update(
        _canonical_json_bytes(
            {
                "domain": "experiment-2d5c/shared-paired-sequence-bootstrap/v1",
                "seed": int(seed),
                "resamples": resamples,
                "sequence_count": sequence_count,
                "index_dtype": "int64",
            }
        )
    )
    for start in range(0, resamples, chunk_size):
        end = min(start + chunk_size, resamples)
        indices = rng.integers(
            0,
            sequence_count,
            size=(end - start, sequence_count),
            dtype=np.int64,
        )
        index_digest.update(indices.tobytes(order="C"))
        for name in names:
            distributions[name][start:end] = arrays[name][indices].mean(
                axis=1, dtype=np.float64
            )
    return arrays, distributions, index_digest.hexdigest()


def _percentile_bounds(values, confidence: float):
    np = _numpy()
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = (1.0 - confidence) / 2.0
    try:
        low, high = np.percentile(
            values, [100.0 * alpha, 100.0 * (1.0 - alpha)], method="linear"
        )
    except TypeError:  # NumPy < 1.22
        low, high = np.percentile(
            values,
            [100.0 * alpha, 100.0 * (1.0 - alpha)],
            interpolation="linear",
        )
    return float(low), float(high)


def _bootstrap_row(values, distribution, confidence: float, target_count: int | None):
    np = _numpy()
    low, high = _percentile_bounds(distribution, confidence)
    sequence_count = int(values.size)
    point = float(values.mean(dtype=np.float64))
    standard_error = (
        float(values.std(ddof=1, dtype=np.float64) / math.sqrt(sequence_count))
        if sequence_count > 1
        else 0.0
    )
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    return {
        "point_estimate": point,
        "mean": point,
        "lower_95": low,
        "upper_95": high,
        "lower_2_5": low,
        "upper_97_5": high,
        "confidence": float(confidence),
        "positive_per_sequence_differences": positive,
        "negative_per_sequence_differences": negative,
        "zero_per_sequence_differences": sequence_count - positive - negative,
        "paired_standard_error": standard_error,
        "sequence_count": sequence_count,
        "target_count": None if target_count is None else int(target_count),
        "bootstrap_distribution_sha256": hashlib.sha256(
            distribution.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def paired_bootstrap_contrasts(
    contrast_vectors: Mapping[str, Sequence[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    chunk_size: int = 128,
    expected_sequences: int | None = None,
    target_count: int | None = None,
) -> dict[str, Any]:
    """Bootstrap all contrasts with one shared stream of sequence indices."""

    arrays, distributions, index_sha = _shared_bootstrap_distributions(
        contrast_vectors,
        seed=seed,
        resamples=resamples,
        chunk_size=chunk_size,
    )
    sequence_count = int(next(iter(arrays.values())).size)
    if expected_sequences is not None and sequence_count != int(expected_sequences):
        raise ValueError(
            f"bootstrap has {sequence_count} sequences, expected {expected_sequences}"
        )
    rows = {
        name: _bootstrap_row(arrays[name], distributions[name], confidence, target_count)
        for name in sorted(arrays)
    }
    return {
        "schema": "exp2d5c_shared_paired_sequence_bootstrap_v1",
        "seed": int(seed),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "method": "paired per-sequence cluster percentile bootstrap",
        "rng": "numpy.random.default_rng",
        "index_dtype": "int64",
        "sequence_count": sequence_count,
        "target_count_per_condition": None if target_count is None else int(target_count),
        "same_resampled_sequence_indices_for_all_contrasts": True,
        "no_independent_bootstrap_subtraction": True,
        "shared_resample_index_stream_sha256": index_sha,
        "contrasts": rows,
    }


def analyze_final_large_panel(
    c_conditions: Mapping[str, Any],
    fixed_conditions: Mapping[str, Any],
    *,
    expected_sequences: int = EXPECTED_LARGE_SEQUENCES,
    expected_targets: int = EXPECTED_LARGE_TARGETS,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Validate the fourteen-condition panel and produce its full analysis."""

    validation = validate_final_condition_data(
        c_conditions,
        fixed_conditions,
        expected_sequences=expected_sequences,
        expected_targets=expected_targets,
    )
    arrays = validation.pop("arrays")
    contrast_vectors = build_final_contrast_vectors(arrays)
    bootstrap = paired_bootstrap_contrasts(
        contrast_vectors,
        seed=seed,
        resamples=resamples,
        chunk_size=chunk_size,
        expected_sequences=expected_sequences,
        target_count=expected_targets,
    )
    definitions = {
        name: [
            {"condition": condition, "coefficient": coefficient}
            for condition, coefficient in terms
        ]
        for name, terms in FINAL_CONTRAST_TERMS.items()
    }
    return {
        "schema": "exp2d5c_final_large_panel_analysis_v1",
        "condition_validation": validation,
        "contrast_definitions": definitions,
        "positive_difference_convention": "positive A-B means B has lower CE and is better",
        "bootstrap": bootstrap,
        "passed": validation["passed"]
        and bootstrap["same_resampled_sequence_indices_for_all_contrasts"],
    }


def _aggregate_ce(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        for key in ("validation_loss", "ce", "loss"):
            if key in value:
                return _finite_float(value[key], f"{label}.{key}")
        return math.fsum(_sequence_values(value, label)) / len(_sequence_values(value, label))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = _sequence_values(value, label)
        return math.fsum(values) / len(values)
    return _finite_float(value, label)


def mechanism_metrics(conditions: Mapping[str, Any]) -> dict[str, float]:
    """Compute the six longitudinal effects plus interaction diagnostics."""

    normalized: dict[str, Any] = {}
    for name, value in conditions.items():
        lower = str(name).lower()
        if lower in _CONTROL_ALIASES:
            normalized[_CONTROL_ALIASES[lower]] = value
        else:
            upper = str(name).upper()
            suffix = upper.split("_", 1)[-1] if upper[:2] in ("C_", "F_") else upper
            if suffix in CONTROL_SUFFIXES:
                normalized[suffix] = value
    missing = sorted(set(CONTROL_SUFFIXES) - set(normalized))
    if missing:
        raise ValueError(f"mechanism conditions missing: {missing}")
    ce = {name: _aggregate_ce(normalized[name], name) for name in CONTROL_SUFFIXES}
    real = ce["ALL_REAL"]
    b3_gain = ce["B3_RECURRENCE_OFF"] - real
    b3_gap = ce["B3_SHUFFLED"] - real
    b5_gain = ce["B5_RECURRENCE_OFF"] - real
    b5_gap = ce["B5_SHUFFLED"] - real
    combined_gain = ce["B3_B5_BOTH_OFF"] - real
    combined_gap = ce["B3_B5_BOTH_SHUFFLED"] - real
    return {
        "all_real_ce": real,
        "b3_recurrent_gain": b3_gain,
        "b3_sequence_gap": b3_gap,
        "b5_recurrent_gain": b5_gain,
        "b5_sequence_gap": b5_gap,
        "combined_recurrent_gain": combined_gain,
        "combined_sequence_gap": combined_gap,
        "off_interaction_combined_minus_individual_sum": combined_gain - b3_gain - b5_gain,
        "shuffled_interaction_combined_minus_individual_sum": combined_gap - b3_gap - b5_gap,
    }


def longitudinal_core_summary(
    c_milestones: Mapping[int | str, Mapping[str, Any]],
    *,
    required_updates: Sequence[int] = (0, 48, 96, 144, 191),
    fixed_milestones: Mapping[int | str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize the preregistered true-incremental milestone controls."""

    c_rows: dict[str, Any] = {}
    for update in required_updates:
        source = c_milestones.get(update, c_milestones.get(str(update)))
        if source is None:
            raise ValueError(f"missing C milestone {update}")
        conditions = source.get("conditions", source)
        c_rows[str(int(update))] = mechanism_metrics(conditions)
    fixed_rows: dict[str, Any] = {}
    if fixed_milestones is not None:
        for update, source in fixed_milestones.items():
            conditions = source.get("conditions", source)
            fixed_rows[str(int(update))] = mechanism_metrics(conditions)
    return {
        "schema": "exp2d5c_longitudinal_core_summary_v1",
        "primary_mode": "deployment-equivalent true incremental",
        "c_required_updates": [int(value) for value in required_updates],
        "c": c_rows,
        "fixed_available": dict(sorted(fixed_rows.items(), key=lambda item: int(item[0]))),
        "interaction_interpretation": "redundancy/synergy diagnostic, not independent architectural proof",
        "passed": list(c_rows) == [str(int(value)) for value in required_updates],
    }


def adaptation_recovery_summary(
    parent0_real: Any,
    c0_real: Any,
    c191_real: Any,
    fixed191_real: Any,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    chunk_size: int = 128,
    minimum_positive_shock: float = 1e-12,
) -> dict[str, Any]:
    """Compute and paired-bootstrap initial shock, penalty, and recovery.

    Recovery is the ratio of paired cluster means in each resample:
    ``1 - mean(P_core) / mean(S)``.  It is never an average of per-sequence
    ratios.  A recovery fraction is withheld when the initial shock is not
    strictly greater than the explicit numerical denominator guard.
    """

    raw = {
        "parent0": _sequence_values(parent0_real, "parent0_real"),
        "c0": _sequence_values(c0_real, "c0_real"),
        "c191": _sequence_values(c191_real, "c191_real"),
        "fixed191": _sequence_values(fixed191_real, "fixed191_real"),
    }
    lengths = {len(row) for row in raw.values()}
    if len(lengths) != 1:
        raise ValueError("recovery inputs must use exactly paired sequences")
    shock = [c0 - parent for c0, parent in zip(raw["c0"], raw["parent0"])]
    penalty = [c - fixed for c, fixed in zip(raw["c191"], raw["fixed191"])]
    arrays, distributions, index_sha = _shared_bootstrap_distributions(
        {"initial_shock": shock, "final_matched_core_penalty": penalty},
        seed=seed,
        resamples=resamples,
        chunk_size=chunk_size,
    )
    shock_row = _bootstrap_row(
        arrays["initial_shock"], distributions["initial_shock"], confidence, None
    )
    penalty_row = _bootstrap_row(
        arrays["final_matched_core_penalty"],
        distributions["final_matched_core_penalty"],
        confidence,
        None,
    )
    shock_point = shock_row["point_estimate"]
    penalty_point = penalty_row["point_estimate"]
    threshold = float(minimum_positive_shock)
    fraction = None
    fraction_row: dict[str, Any] = {
        "point_estimate": None,
        "lower_95": None,
        "upper_95": None,
        "defined": False,
        "reason": "initial shock is nonpositive or too close to zero",
        "minimum_positive_shock": threshold,
    }
    if shock_point > threshold:
        np = _numpy()
        shock_boot = distributions["initial_shock"]
        penalty_boot = distributions["final_matched_core_penalty"]
        valid = shock_boot > threshold
        valid_count = int(np.count_nonzero(valid))
        fraction = 1.0 - penalty_point / shock_point
        fraction_row.update(
            {
                "point_estimate": float(fraction),
                "valid_bootstrap_resamples": valid_count,
                "invalid_bootstrap_resamples": int(resamples) - valid_count,
                "descriptive_positive_recovery": bool(fraction > 0.0),
                "meaningful_recovery_preregistered_threshold": None,
            }
        )
        if valid_count == int(resamples):
            recovery_boot = 1.0 - penalty_boot / shock_boot
            low, high = _percentile_bounds(recovery_boot, confidence)
            fraction_row.update(
                {
                    "lower_95": low,
                    "upper_95": high,
                    "defined": True,
                    "reason": None,
                    "bootstrap_distribution_sha256": hashlib.sha256(
                        recovery_boot.astype("<f8", copy=False).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        else:
            fraction_row["reason"] = (
                "one or more paired bootstrap resamples had a nonpositive or "
                "numerically tiny initial shock; recovery CI withheld"
            )
    return {
        "schema": "exp2d5c_adaptation_recovery_v1",
        "formulae": {
            "initial_shock": "C0_REAL - PARENT0_REAL",
            "final_matched_core_penalty": "C191_REAL - F191_REAL",
            "recovery_fraction": "1 - final_matched_core_penalty / initial_shock",
        },
        "initial_shock": shock_row,
        "final_matched_core_penalty": penalty_row,
        "recovery_fraction": fraction_row,
        "same_resampled_sequence_indices": True,
        "shared_resample_index_stream_sha256": index_sha,
        "seed": int(seed),
        "resamples": int(resamples),
    }


def _ci_values(row: Mapping[str, Any]) -> tuple[float, float, float]:
    point = row.get("point_estimate", row.get("mean"))
    lower = row.get("lower_95", row.get("lower_2_5"))
    upper = row.get("upper_95", row.get("upper_97_5"))
    if point is None or lower is None or upper is None:
        raise ValueError("contrast row lacks point estimate or 95% CI")
    return (
        _finite_float(point, "point estimate"),
        _finite_float(lower, "lower CI"),
        _finite_float(upper, "upper CI"),
    )


def _contrast_rows(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if "bootstrap" in value and isinstance(value["bootstrap"], Mapping):
        value = value["bootstrap"]
    if "contrasts" in value and isinstance(value["contrasts"], Mapping):
        value = value["contrasts"]
    return value


def classification_decision(
    bootstrap: Mapping[str, Any],
    *,
    audit_passed: bool,
    meaningful_recovery: bool | None = None,
    margin_ce: float = NONINFERIORITY_MARGIN_CE,
) -> dict[str, Any]:
    """Apply the protocol's A-through-F classification tree in order.

    The protocol does not preregister a numeric threshold for "meaningful"
    longitudinal recovery.  Therefore the partial-replacement branch requires
    an explicit audited boolean; a positive descriptive fraction is not
    silently promoted to that scientific judgment.
    """

    if not audit_passed:
        return {
            "classification": "INVALID — NO SCIENTIFIC CONCLUSION",
            "audit_passed": False,
            "classification_allowed": False,
            "decision_branch": "FINAL_AUDIT_FAILURE",
            "flags": {},
        }
    rows = _contrast_rows(bootstrap)
    needed = (
        "architecture_fixed_minus_c",
        "architecture_c_minus_fixed_penalty",
        "c_combined_off_gain",
        "c_combined_sequence_gap",
        "combined_off_gain_lift",
        "combined_sequence_gap_lift",
    )
    missing = [name for name in needed if name not in rows]
    if missing:
        raise ValueError(f"classification contrasts missing: {missing}")
    estimates = {name: _ci_values(rows[name]) for name in needed}
    superiority = estimates["architecture_fixed_minus_c"][1] > 0.0
    penalty = estimates["architecture_c_minus_fixed_penalty"]
    noninferior = penalty[2] < float(margin_ce)
    statistically_worse = penalty[1] > 0.0
    materially_worse = penalty[1] > float(margin_ce)
    c_off = estimates["c_combined_off_gain"][1] > 0.0
    c_aligned = estimates["c_combined_sequence_gap"][1] > 0.0
    gain_lift = estimates["combined_off_gain_lift"][1] > 0.0
    aligned_lift = estimates["combined_sequence_gap_lift"][1] > 0.0
    stronger = gain_lift and aligned_lift
    flags = {
        "practical_noninferiority_established": noninferior,
        "absolute_ce_superiority_established": superiority,
        "statistical_worsening_established": statistically_worse,
        "material_worsening_beyond_margin_established": materially_worse,
        "c_combined_off_gain_established": c_off,
        "c_combined_sequence_gap_established": c_aligned,
        "combined_off_gain_lift_established": gain_lift,
        "combined_sequence_gap_lift_established": aligned_lift,
        "recurrent_dependence_increased_established": stronger,
        "meaningful_longitudinal_recovery": meaningful_recovery,
        "noninferiority_margin_ce": float(margin_ce),
    }

    # Apply the labeled A-F branches in their protocol order.  This matters
    # for the intentional overlap between recovery-without-dependence (C) and
    # path-utility-without-alignment (D).
    if noninferior and c_off and c_aligned and gain_lift and aligned_lift:
        classification = "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED"
        if superiority:
            classification += " WITH ABSOLUTE CE IMPROVEMENT"
        branch = "A_STRONG_REPLACEMENT"
    elif (
        (statistically_worse or materially_worse)
        and c_off
        and c_aligned
        and stronger
        and meaningful_recovery is True
    ):
        classification = "W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL"
        branch = "B_PARTIAL_REPLACEMENT"
    elif noninferior and not (c_off and c_aligned and gain_lift and aligned_lift):
        classification = "W2/W2 RECOVERS WITHOUT ESTABLISHED B3/B5 RECURRENT DEPENDENCE"
        branch = "C_RECOVERY_WITHOUT_ESTABLISHED_DEPENDENCE"
    elif c_off and not c_aligned:
        classification = "RECURRENT PATH UTILITY ESTABLISHED; ALIGNED SEQUENCE MEMORY NOT ESTABLISHED"
        branch = "D_PATH_UTILITY_WITHOUT_ALIGNED_MEMORY"
    elif materially_worse and not (c_off and c_aligned and stronger):
        classification = "W2/W2 PERSISTENT DEGRADATION; DEEP-RECURRENT SUBSTITUTION NOT SUPPORTED"
        branch = "E_PERSISTENT_FAILURE"
    else:
        classification = "W2/W2 REPRESENTATION-PRESSURE RESULT UNRESOLVED"
        branch = "F_MIXED_OR_BOUNDARY_CROSSING"
    return {
        "classification": classification,
        "audit_passed": True,
        "classification_allowed": True,
        "decision_branch": branch,
        "flags": flags,
        "estimates_used": {
            name: {"point_estimate": row[0], "lower_95": row[1], "upper_95": row[2]}
            for name, row in estimates.items()
        },
    }


# Friendly alias for call sites that prefer a verb matching prior drivers.
classify_2d5c = classification_decision


def recommendation_after_c(
    classification: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, str]:
    """Recommend, but never execute, the next decomposition/diagnostic."""

    if classification not in CLASSIFICATIONS:
        raise ValueError(f"unknown 2D5C classification: {classification}")
    if classification == "INVALID — NO SCIENTIFIC CONCLUSION":
        return {"recommendation": "NEITHER", "reason": "repair the failed audit first"}
    if classification.startswith("W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED"):
        return {"recommendation": "NEITHER", "reason": "C succeeds cleanly"}
    if classification == "W2/W2 RECOVERS WITHOUT ESTABLISHED B3/B5 RECURRENT DEPENDENCE":
        return {
            "recommendation": "DIFFERENT_NEXT_DIAGNOSTIC",
            "reason": "prefer a residual-stream/full-integration control over A or B",
        }

    rows = _contrast_rows(bootstrap)

    def established(name: str) -> bool:
        return name in rows and _ci_values(rows[name])[1] > 0.0

    b3 = all(
        established(name)
        for name in ("c_b3_off_gain", "c_b3_sequence_gap", "b3_off_gain_lift", "b3_sequence_gap_lift")
    )
    b5 = all(
        established(name)
        for name in ("c_b5_off_gain", "c_b5_sequence_gap", "b5_off_gain_lift", "b5_sequence_gap_lift")
    )
    if b3 and not b5:
        return {
            "recommendation": "B",
            "reason": "B isolates the concerning B5 W2 change while retaining B3 W32",
        }
    if b5 and not b3:
        return {
            "recommendation": "A",
            "reason": "A isolates the concerning B3 W2 change while retaining B5 W64",
        }
    if classification == "W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL":
        return {
            "recommendation": "DIFFERENT_NEXT_DIAGNOSTIC",
            "reason": "compare A/B decomposition with modest intermediate W4/W8 windows; execute nothing",
        }
    return {
        "recommendation": "BOTH",
        "reason": "both destinations are mixed or implicated; A and B are needed for causal decomposition",
    }


def expected_bf16_local_kv_reduction(
    *,
    width: int = 768,
    element_size: int = 2,
    batch_size: int = 1,
    fixed_history: Mapping[str, int] | None = None,
    c_history: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return the protocol's expected raw local K+V payload reduction."""

    if int(width) <= 0 or int(element_size) <= 0 or int(batch_size) <= 0:
        raise ValueError("width, element_size, and batch_size must be positive")
    fixed = dict(fixed_history or {"B3": 31, "B5": 63})
    current = dict(c_history or {"B3": 1, "B5": 1})
    saved_positions = {
        block: int(fixed[block]) - int(current[block]) for block in ("B3", "B5")
    }
    if any(value < 0 for value in saved_positions.values()):
        raise ValueError("C history cannot exceed Fixed for the expected reduction")
    bytes_saved = (
        sum(saved_positions.values())
        * 2  # K plus V
        * int(width)
        * int(element_size)
        * int(batch_size)
    )
    return {
        "fixed_historical_positions": fixed,
        "c_historical_positions": current,
        "saved_historical_positions": saved_positions,
        "saved_positions_total": sum(saved_positions.values()),
        "width": int(width),
        "element_size_bytes": int(element_size),
        "batch_size": int(batch_size),
        "logical_reduction_bytes": bytes_saved,
        "logical_reduction_kib": bytes_saved / 1024.0,
        "formula": "saved positions * (K+V) * width * BF16 bytes * batch",
    }


def _cache_component_rows(audit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for row in audit.get("cache_storage", ()):
        block = f"B{int(row['block'])}_local_kv"
        tensors = [row.get("key"), row.get("value")]
        present = [tensor for tensor in tensors if tensor is not None]
        components[block] = {
            "category": "same_layer_local_kv",
            "logical_bytes": sum(int(tensor["expected_bytes"]) for tensor in present),
            "allocated_bytes": sum(int(tensor["actual_bytes"]) for tensor in present),
            "dtypes": sorted({str(tensor.get("dtype")) for tensor in present}),
            "shapes": [tensor.get("shape") for tensor in present],
            "storage_exact": all(bool(tensor.get("exact")) for tensor in present),
        }
    for name, tensor in audit.get("ring_storage", {}).items():
        components[f"{name}_recurrent_source_ring"] = {
            "category": "recurrent_source_ring",
            "logical_bytes": int(tensor["expected_bytes"]),
            "allocated_bytes": int(tensor["actual_bytes"]),
            "dtypes": [str(tensor.get("dtype"))],
            "shapes": [tensor.get("shape")],
            "storage_exact": bool(tensor.get("exact")),
        }
    other = audit.get("other_persistent_components", {})
    for name, row in other.items():
        components[f"other:{name}"] = {
            "category": "other_persistent_decoding_state",
            "logical_bytes": int(row["logical_bytes"]),
            "allocated_bytes": int(row["allocated_bytes"]),
            "dtypes": [str(row.get("dtype"))],
            "shapes": [row.get("shape")],
            "storage_exact": bool(row.get("storage_exact", True)),
        }
    return components


def compare_bf16_persistent_state(
    fixed_audit: Mapping[str, Any],
    c_audit: Mapping[str, Any],
    *,
    batch_size: int = 1,
    width: int = 768,
) -> dict[str, Any]:
    """Compare steady-state Fixed and C cache audits without double counting."""

    fixed_components = _cache_component_rows(fixed_audit)
    c_components = _cache_component_rows(c_audit)
    if set(fixed_components) != set(c_components):
        raise ValueError("Fixed and C persistent component sets differ")
    component_table = []
    for name in sorted(fixed_components):
        fixed = fixed_components[name]
        current = c_components[name]
        component_table.append(
            {
                "component": name,
                "category": fixed["category"],
                "fixed_logical_bytes": fixed["logical_bytes"],
                "c_logical_bytes": current["logical_bytes"],
                "logical_reduction_bytes": fixed["logical_bytes"] - current["logical_bytes"],
                "fixed_allocated_bytes": fixed["allocated_bytes"],
                "c_allocated_bytes": current["allocated_bytes"],
                "allocated_reduction_bytes": fixed["allocated_bytes"] - current["allocated_bytes"],
                "fixed_dtypes": fixed["dtypes"],
                "c_dtypes": current["dtypes"],
                "fixed_shapes": fixed["shapes"],
                "c_shapes": current["shapes"],
                "storage_exact": fixed["storage_exact"] and current["storage_exact"],
            }
        )
    fixed_logical = int(fixed_audit["logical_payload_bytes"])
    c_logical = int(c_audit["logical_payload_bytes"])
    fixed_physical = int(fixed_audit["actual_unique_storage_bytes"])
    c_physical = int(c_audit["actual_unique_storage_bytes"])
    fixed_component_logical = sum(
        row["fixed_logical_bytes"] for row in component_table
    )
    c_component_logical = sum(row["c_logical_bytes"] for row in component_table)
    fixed_component_allocated = sum(
        row["fixed_allocated_bytes"] for row in component_table
    )
    c_component_allocated = sum(
        row["c_allocated_bytes"] for row in component_table
    )
    expected = expected_bf16_local_kv_reduction(
        width=width, batch_size=batch_size
    )
    local_changed = {
        row["component"]
        for row in component_table
        if row["category"] == "same_layer_local_kv"
        and (
            row["logical_reduction_bytes"] != 0
            or row["allocated_reduction_bytes"] != 0
        )
    }
    recurrent = [
        row for row in component_table if row["category"] == "recurrent_source_ring"
    ]
    bf16_dtypes = {"torch.bfloat16", "bfloat16", "bf16"}
    all_bf16 = all(
        set(row["fixed_dtypes"] + row["c_dtypes"]).issubset(bf16_dtypes)
        for row in component_table
        if row["fixed_dtypes"] or row["c_dtypes"]
    )
    checks = {
        "fixed_cache_audit_passed": bool(fixed_audit.get("passed")),
        "c_cache_audit_passed": bool(c_audit.get("passed")),
        "batch_size_one": int(batch_size) == 1,
        "bf16_every_persistent_tensor": all_bf16,
        "fixed_b3_b5_history_exact": fixed_audit.get("b3_historical_local_kv") == 31
        and fixed_audit.get("b5_historical_local_kv") == 63,
        "c_b3_b5_history_exact": c_audit.get("b3_historical_local_kv") == 1
        and c_audit.get("b5_historical_local_kv") == 1,
        "only_b3_b5_local_kv_changed": local_changed
        == {"B3_local_kv", "B5_local_kv"},
        "recurrent_rings_logical_unchanged": all(
            row["logical_reduction_bytes"] == 0 for row in recurrent
        ),
        "recurrent_rings_allocated_unchanged": all(
            row["allocated_reduction_bytes"] == 0 for row in recurrent
        ),
        "recurrent_ring_shapes_unchanged": all(
            row["fixed_shapes"] == row["c_shapes"] for row in recurrent
        ),
        "logical_reduction_matches_282624_bytes": fixed_logical - c_logical
        == expected["logical_reduction_bytes"],
        "component_logical_totals_match_audit": fixed_component_logical
        == fixed_logical
        and c_component_logical == c_logical,
        "component_allocated_totals_match_unique_storage_audit": fixed_component_allocated
        == fixed_physical
        and c_component_allocated == c_physical,
        "unique_storage_accounting": bool(fixed_audit.get("storage_alias_free"))
        and bool(c_audit.get("storage_alias_free")),
        "physical_storage_audits_exact": bool(fixed_audit.get("physical_storage_exact"))
        and bool(c_audit.get("physical_storage_exact")),
    }
    return {
        "schema": "exp2d5c_bf16_persistent_state_audit_v1",
        "canonical_batch_size": int(batch_size),
        "batch_scaling": "all reported per-sequence tensor payloads scale linearly with active batch size",
        "expected_local_kv_reduction": expected,
        "logical": {
            "fixed_bytes": fixed_logical,
            "c_bytes": c_logical,
            "reduction_bytes": fixed_logical - c_logical,
            "reduction_kib": (fixed_logical - c_logical) / 1024.0,
        },
        "allocated_unique_storage": {
            "fixed_bytes": fixed_physical,
            "c_bytes": c_physical,
            "reduction_bytes": fixed_physical - c_physical,
            "reduction_kib": (fixed_physical - c_physical) / 1024.0,
            "physical_savings_claimed_only_from_measured_storage": True,
        },
        "component_table": component_table,
        "allocator_reserved_unused_excluded": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def account_unique_tensor_storages(named_tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Audit tensor-like persistent storage while deduplicating aliases.

    This duck-typed helper is convenient when a model exposes its live state
    tensors directly.  Full-storage contiguous tensors are required so the
    logical unique payload is unambiguous.
    """

    storages: dict[tuple[str, int, int], dict[str, Any]] = {}
    rows = []
    for name, tensor in sorted(named_tensors.items()):
        numel = int(tensor.numel())
        element_size = int(tensor.element_size())
        logical = numel * element_size
        raw = tensor.untyped_storage()
        actual = int(raw.nbytes())
        identity = (str(tensor.device), int(raw.data_ptr()), actual)
        exact = (
            int(tensor.storage_offset()) == 0
            and bool(tensor.is_contiguous())
            and logical == actual
        )
        row = {
            "name": str(name),
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
            "logical_bytes": logical,
            "allocated_storage_bytes": actual,
            "storage_identity_sha256": hashlib.sha256(
                _canonical_json_bytes(identity)
            ).hexdigest(),
            "full_storage_contiguous": exact,
            "alias": identity in storages,
        }
        rows.append(row)
        storages.setdefault(identity, row)
    return {
        "tensors": rows,
        "tensor_count": len(rows),
        "unique_storage_count": len(storages),
        "logical_unique_payload_bytes": sum(
            row["logical_bytes"] for row in storages.values()
        ),
        "actual_unique_storage_bytes": sum(identity[2] for identity in storages),
        "aliases_deduplicated": len(rows) - len(storages),
        "all_unique_tensors_full_storage_contiguous": all(
            row["full_storage_contiguous"] for row in storages.values()
        ),
    }


def validate_final_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete machine-readable final-audit check set."""

    checks = audit.get("checks", audit)
    if not isinstance(checks, Mapping):
        raise ValueError("final audit checks must be a mapping")
    missing = [name for name in REQUIRED_FINAL_AUDIT_CHECKS if name not in checks]
    failing = [name for name in REQUIRED_FINAL_AUDIT_CHECKS if checks.get(name) is not True]
    critical_failures = sorted(CRITICAL_INVALID_AUDIT_CHECKS.intersection(failing))
    declared = audit.get("passed")
    passed = not missing and not failing and declared is not False
    if critical_failures:
        status = "INVALID — NO SCIENTIFIC CONCLUSION"
    elif not passed:
        status = "AUDIT INCOMPLETE — SCIENTIFIC CLASSIFICATION WITHHELD"
    else:
        status = "PASS"
    return {
        "required_check_count": len(REQUIRED_FINAL_AUDIT_CHECKS),
        "missing_checks": missing,
        "failing_checks": failing,
        "critical_invalid_failures": critical_failures,
        "passed": passed,
        "classification_allowed": passed,
        "status": status,
    }


def validate_required_artifacts(artifacts: Mapping[str, Any] | Iterable[str]) -> dict[str, Any]:
    names = set(artifacts if not isinstance(artifacts, Mapping) else artifacts.keys())
    missing = [name for name in REQUIRED_ARTIFACTS if name not in names]
    return {
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "present_required_artifact_count": len(REQUIRED_ARTIFACTS) - len(missing),
        "missing_required_artifacts": missing,
        "passed": not missing,
    }


def validate_final_report_data(
    summary: Mapping[str, Any],
    final_audit: Mapping[str, Any],
    artifacts: Mapping[str, Any] | Iterable[str],
    *,
    report_text: str | None = None,
) -> dict[str, Any]:
    """Fail closed on the final report's separable required data fields."""

    audit_validation = validate_final_audit(final_audit)
    artifact_validation = validate_required_artifacts(artifacts)
    classification = summary.get("classification")
    contrasts = _contrast_rows(summary.get("bootstrap", summary.get("contrasts", {})))
    required_contrasts = set(FINAL_CONTRAST_TERMS)
    final_checkpoint = summary.get("final_checkpoint", {})
    memory = summary.get("bf16_persistent_state", summary.get("memory", {}))
    checks = {
        "classification_known": classification in CLASSIFICATIONS,
        "audit_allows_classification": audit_validation["classification_allowed"],
        "fixed_all_real_ce_finite": math.isfinite(float(summary.get("fixed_all_real_ce", math.nan))),
        "c_all_real_ce_finite": math.isfinite(float(summary.get("c_all_real_ce", math.nan))),
        "all_required_contrasts": required_contrasts.issubset(contrasts),
        "memory_accounting_present": isinstance(memory, Mapping)
        and "logical" in memory
        and "allocated_unique_storage" in memory,
        "checkpoint_path_present": bool(final_checkpoint.get("path", final_checkpoint.get("checkpoint"))),
        "checkpoint_sha256_valid": isinstance(final_checkpoint.get("sha256"), str)
        and len(final_checkpoint["sha256"]) == 64,
        "git_identity_present": all(summary.get(key) for key in ("git_branch", "git_commit", "git_tag")),
        "runpod_status_present": summary.get("runpod_status") in {"stopped", "not stopped", "STOPPED", "RUNNING"},
        "required_artifacts_present": artifact_validation["passed"],
    }
    if report_text is not None:
        checks["exact_terminal_phrase"] = report_text.rstrip().endswith(
            "STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS"
        )
    return {
        "audit_validation": audit_validation,
        "artifact_validation": artifact_validation,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_RESAMPLES",
    "EXPECTED_LARGE_SEQUENCES",
    "EXPECTED_LARGE_TARGETS",
    "NONINFERIORITY_MARGIN_CE",
    "FINAL_CONDITIONS",
    "FINAL_CONTRAST_TERMS",
    "CLASSIFICATIONS",
    "REQUIRED_FINAL_AUDIT_CHECKS",
    "CRITICAL_INVALID_AUDIT_CHECKS",
    "REQUIRED_ARTIFACTS",
    "validate_final_condition_data",
    "build_final_contrast_vectors",
    "paired_bootstrap_contrasts",
    "analyze_final_large_panel",
    "mechanism_metrics",
    "longitudinal_core_summary",
    "adaptation_recovery_summary",
    "classification_decision",
    "classify_2d5c",
    "recommendation_after_c",
    "expected_bf16_local_kv_reduction",
    "compare_bf16_persistent_state",
    "account_unique_tensor_storages",
    "validate_final_audit",
    "validate_required_artifacts",
    "validate_final_report_data",
]
