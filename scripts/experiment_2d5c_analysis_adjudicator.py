#!/usr/bin/env python3
"""Append-only analysis adjudication for the sealed Experiment 2D5C run.

The frozen driver writes JSON objects with lexicographically sorted keys.  Its
analysis audit nevertheless compared the reopened ``conditions`` mapping order
with the semantic control order.  The independent ordered
``controls_requested`` list was correct, every condition was present exactly
once, and all other identity checks passed.

This tool leaves the frozen driver byte-for-byte unchanged.  It requires the
exact official failed audit, preserves it under a distinct append-only name,
loads the exact frozen driver, and replaces only the invalid mapping-order
predicate in memory.  The replacement still requires both the exact ordered
control list and the exact condition-name set.  It cannot train or alter a
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


EXPERIMENT = "2D5C"
FROZEN_DRIVER_SHA256 = (
    "204e589d4dfd68ed2136d70d2470c6bc06a85f5e89b706a7075706d039081b48"
)
OFFICIAL_FAILED_AUDIT_SHA256 = (
    "d7b4c30e75412a137b856a174fb3851dca157ab85201d67832f73f243b40f1ae"
)
EXPECTED_CORRECTIONS = {
    "core_evaluations": {"c0", "c48", "c96", "c144", "c191", "fixed100m"},
    "large_evaluations": {"c191", "fixed100m"},
    "secondary_parallel_evaluations": {"c0", "c96", "c191"},
}


class AnalysisAdjudicationError(RuntimeError):
    """Raised when the one authorized audit correction is not proven exactly."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise AnalysisAdjudicationError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def durable_bytes_exclusive(path: str | Path, data: bytes) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as error:
        raise AnalysisAdjudicationError(
            f"append-only output already exists: {destination}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def durable_json_exclusive(path: str | Path, payload: Any) -> None:
    durable_bytes_exclusive(
        path,
        (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def false_checks(row: Any) -> list[str]:
    if not isinstance(row, dict) or not isinstance(row.get("checks"), dict):
        raise AnalysisAdjudicationError("audit row has no checks mapping")
    return sorted(name for name, value in row["checks"].items() if value is not True)


def validate_official_failed_audit(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("experiment") != EXPERIMENT:
        raise AnalysisAdjudicationError("failed audit is not Experiment 2D5C")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or payload.get("passed") is not False:
        raise AnalysisAdjudicationError("official audit is not the expected failure")

    observed: dict[str, set[str]] = {}
    for group_name, expected_names in EXPECTED_CORRECTIONS.items():
        group = groups.get(group_name)
        if not isinstance(group, dict) or set(group) != (
            expected_names | ({"parent"} if group_name == "core_evaluations" else set())
        ):
            raise AnalysisAdjudicationError(
                f"unexpected artifact set in failed audit group {group_name}"
            )
        corrected_names = set()
        for name, row in group.items():
            failures = false_checks(row)
            if name == "parent":
                if failures or row.get("passed") is not True:
                    raise AnalysisAdjudicationError("parent evaluation did not pass exactly")
            else:
                if failures != ["condition_keys_exact"] or row.get("passed") is not False:
                    raise AnalysisAdjudicationError(
                        f"unexpected failed checks for {group_name}/{name}: {failures}"
                    )
                corrected_names.add(name)
        observed[group_name] = corrected_names

    for group_name in (
        "artifact_sets", "cross_artifact_binding", "frozen_manifests",
        "milestone_manifest", "pretrain_freeze",
    ):
        row = groups.get(group_name)
        if false_checks(row) or row.get("passed") is not True:
            raise AnalysisAdjudicationError(
                f"non-order audit group did not pass: {group_name}"
            )
    representation = groups.get("representation_diagnostics")
    if not isinstance(representation, dict) or set(representation) != {
        "parent", "c0", "c48", "c96", "c144", "c191", "fixed100m"
    }:
        raise AnalysisAdjudicationError("representation artifact set is not exact")
    for name, row in representation.items():
        if false_checks(row) or row.get("passed") is not True:
            raise AnalysisAdjudicationError(
                f"representation audit did not pass: {name}"
            )
    if observed != EXPECTED_CORRECTIONS:
        raise AnalysisAdjudicationError("failed-audit correction set is not exact")
    return {
        "failed_only_on_serialized_mapping_order": True,
        "expected_correction_groups": {
            group: sorted(names) for group, names in EXPECTED_CORRECTIONS.items()
        },
        "correction_count": sum(len(names) for names in observed.values()),
        "passed": True,
    }


def preserve_official_failed_audit(source: Path, destination: Path) -> dict[str, Any]:
    if sha256(source) != OFFICIAL_FAILED_AUDIT_SHA256:
        raise AnalysisAdjudicationError("official failed audit SHA-256 mismatch")
    if destination.exists():
        if sha256(destination) != OFFICIAL_FAILED_AUDIT_SHA256:
            raise AnalysisAdjudicationError(
                "preserved failed audit exists with a different identity"
            )
    else:
        durable_bytes_exclusive(destination, source.read_bytes())
    return file_identity(destination)


def load_frozen_driver(path: Path) -> Any:
    if sha256(path) != FROZEN_DRIVER_SHA256:
        raise AnalysisAdjudicationError("frozen driver SHA-256 mismatch")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "experiment_2d5c_frozen_for_analysis_adjudication", path
    )
    if spec is None or spec.loader is None:
        raise AnalysisAdjudicationError("could not load frozen driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrected_identity_checker(
    original: Callable[..., dict[str, Any]], corrections: list[dict[str, Any]]
) -> Callable[..., dict[str, Any]]:
    """Correct only the proven mapping-order predicate, fail closed otherwise."""

    def check(evaluation: Any, spec: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(evaluation, spec, *args, **kwargs)
        if result.get("passed") is True:
            return result
        failures = false_checks(result)
        controls = list(spec.get("controls", ())) if isinstance(spec, dict) else []
        conditions = evaluation.get("conditions", {}) if isinstance(evaluation, dict) else {}
        ordered = evaluation.get("controls_requested") if isinstance(evaluation, dict) else None
        authorized = (
            failures == ["condition_keys_exact"]
            and isinstance(conditions, dict)
            and ordered == controls
            and set(conditions) == set(controls)
        )
        if not authorized:
            raise AnalysisAdjudicationError(
                f"evaluation has a non-authorized identity failure: {failures}"
            )
        corrected = copy.deepcopy(result)
        corrected["checks"]["condition_keys_exact"] = True
        corrected["passed"] = all(corrected["checks"].values())
        if corrected["passed"] is not True:
            raise AnalysisAdjudicationError("corrected identity row still does not pass")
        identity = evaluation.get("evaluation_identity", {})
        corrections.append({
            "family": evaluation.get("family"),
            "local_update": identity.get("local_update"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "parallel": bool(kwargs.get("parallel", False)),
            "controls_requested": controls,
            "serialized_condition_keys": list(conditions),
            "condition_key_set_exact": True,
            "ordered_controls_exact": True,
            "original_false_checks": failures,
        })
        return corrected

    return check


def corrected_audit_passes(path: Path) -> bool:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        return False
    groups = payload.get("groups", {})
    for group_name, names in EXPECTED_CORRECTIONS.items():
        group = groups.get(group_name, {})
        expected = names | ({"parent"} if group_name == "core_evaluations" else set())
        if set(group) != expected:
            return False
        if any(row.get("passed") is not True or false_checks(row) for row in group.values()):
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    driver = args.frozen_driver.resolve()
    current_audit = args.failed_audit.resolve()
    preserved_audit = args.preserved_failed_audit.resolve()
    adjudication_output = args.adjudication_output.resolve()
    forwarded = list(args.driver_arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if forwarded[:1] != ["analyze"]:
        raise AnalysisAdjudicationError("wrapper permits only the frozen analyze command")

    if sha256(driver) != FROZEN_DRIVER_SHA256:
        raise AnalysisAdjudicationError("frozen driver SHA-256 mismatch")
    if adjudication_output.exists():
        existing = read_json(adjudication_output)
        checks = {
            "schema": existing.get("schema") == "experiment_2d5c_analysis_order_adjudication_v1",
            "passed": existing.get("passed") is True,
            "tool": existing.get("adjudicator", {}).get("sha256") == sha256(__file__),
            "driver": existing.get("frozen_driver", {}).get("sha256") == FROZEN_DRIVER_SHA256,
            "legacy": sha256(preserved_audit) == OFFICIAL_FAILED_AUDIT_SHA256,
            "corrected": corrected_audit_passes(current_audit),
        }
        if not all(checks.values()):
            raise AnalysisAdjudicationError(
                f"existing analysis adjudication is invalid: {checks}"
            )
        return existing

    legacy_identity = preserve_official_failed_audit(current_audit, preserved_audit)
    legacy_validation = validate_official_failed_audit(read_json(preserved_audit))
    frozen = load_frozen_driver(driver)
    corrections: list[dict[str, Any]] = []
    frozen.evaluation_artifact_identity_checks = corrected_identity_checker(
        frozen.evaluation_artifact_identity_checks, corrections
    )
    frozen.main(forwarded)
    if len(corrections) != sum(len(names) for names in EXPECTED_CORRECTIONS.values()):
        raise AnalysisAdjudicationError(
            f"unexpected number of applied corrections: {len(corrections)}"
        )
    if not corrected_audit_passes(current_audit):
        raise AnalysisAdjudicationError("corrected analysis input audit did not pass")
    payload = {
        "schema": "experiment_2d5c_analysis_order_adjudication_v1",
        "experiment": EXPERIMENT,
        "purpose": "correct one frozen serialized-mapping-order audit defect",
        "frozen_driver": file_identity(driver),
        "adjudicator": file_identity(Path(__file__).resolve()),
        "legacy_failed_audit": legacy_identity,
        "legacy_validation": legacy_validation,
        "corrected_analysis_input_audit": file_identity(current_audit),
        "correction_rule": {
            "ordered_controls_required": True,
            "exact_condition_name_set_required": True,
            "mapping_insertion_order_ignored_after_sorted_json_serialization": True,
        },
        "corrections": corrections,
        "checkpoint_or_measurement_mutation": False,
        "training_invoked": False,
        "created_at_unix": time.time(),
        "passed": True,
    }
    durable_json_exclusive(adjudication_output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-driver", type=Path, required=True)
    parser.add_argument("--failed-audit", type=Path, required=True)
    parser.add_argument("--preserved-failed-audit", type=Path, required=True)
    parser.add_argument("--adjudication-output", type=Path, required=True)
    parser.add_argument("driver_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run(build_parser().parse_args(argv))
    except (AnalysisAdjudicationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"EXPERIMENT_2D5C_ANALYSIS_ADJUDICATION_FAILURE: {error}", file=sys.stderr)
        return 1
    print(
        "EXPERIMENT_2D5C_ANALYSIS_ADJUDICATION_COMPLETE "
        + payload["corrected_analysis_input_audit"]["sha256"],
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
