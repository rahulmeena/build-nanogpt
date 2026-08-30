#!/usr/bin/env python3
"""Freeze a full-tensor 2D5C continuation envelope before official training."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import uuid
from pathlib import Path

import experiment_2d5c as driver


BASE_PROVENANCE_SHA256 = (
    "53a9324fcbf0f9196bfa27ee77c08d6ca7b571ba1e27007a472ab6c03f2f6e74"
)
PROBE_SCHEMA = "experiment_2d5c_continuation_full_tensor_probe_v1"
EVIDENCE_SCHEMA = "experiment_2d5c_continuation_tensor_envelope_evidence_v1"
MARGIN_FACTOR = 4.0
MINIMUM_TOLERANCES = {
    "gradients": {"max_abs": 1.0e-7, "l2_norm": 1.0e-5},
    "model_parameters": {"max_abs": 1.0e-8, "l2_norm": 1.0e-6},
    "optimizer_state": {"max_abs": 1.0e-7, "l2_norm": 1.0e-5},
}


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def identity(path: Path) -> dict:
    return driver.file_identity(require_file(path, str(path)))


def load_report(path: Path) -> dict:
    payload = json.loads(require_file(path, "probe report").read_text())
    run_uuid = payload.get("run_uuid")
    try:
        parsed_uuid = uuid.UUID(run_uuid) if isinstance(run_uuid, str) else None
    except ValueError:
        parsed_uuid = None
    checks = {
        "schema": payload.get("schema") == PROBE_SCHEMA,
        "experiment": payload.get("experiment") == driver.EXPERIMENT,
        "label": isinstance(payload.get("label"), str)
        and payload["label"].startswith("DISPOSABLE_"),
        "disposable": payload.get("disposable") is True,
        "before_official": payload.get("created_before_official_training")
        is True and payload.get("official_training_updates_executed") == 0,
        "official_run_root": payload.get("official_run_root")
        == "/workspace/exp2d5c_w2w2_100m",
        "official_state_before": bool(payload.get("official_state_checks_before"))
        and all(payload.get("official_state_checks_before", {}).values()),
        "official_state_after": bool(payload.get("official_state_checks_after"))
        and all(payload.get("official_state_checks_after", {}).values()),
        "source_unchanged": payload.get("source_unchanged") is True,
        "process": type(payload.get("process_id")) is int
        and payload["process_id"] > 0,
        "parent_process": type(payload.get("parent_process_id")) is int
        and payload["parent_process_id"] > 0,
        "run_uuid": parsed_uuid is not None and parsed_uuid.version == 4,
        "boundary": bool(payload.get("boundary_checks"))
        and all(payload.get("boundary_checks", {}).values()),
        "passed": payload.get("passed") is True,
    }
    if not all(checks.values()):
        raise SystemExit(f"probe report did not pass: {path}")
    return payload


def same_content(record: dict, actual: dict) -> bool:
    return (
        isinstance(record, dict)
        and record.get("sha256") == actual.get("sha256")
        and record.get("bytes") == actual.get("bytes")
    )


def expected_metadata(reference: dict) -> dict:
    summary = reference.get("snapshot_summary", {})
    if not (
        summary.get("schema") == driver.CONTINUATION_TENSOR_SNAPSHOT_SCHEMA
        and summary.get("label") == driver.CONTINUATION_TENSOR_SNAPSHOT_LABEL
        and summary.get("disposable") is True
        and summary.get("official_training_updates_executed") == 0
    ):
        raise SystemExit("reference snapshot summary is not explicitly disposable")
    sections = summary.get("sections", {})
    metadata = {}
    for section_name, spec in driver.CONTINUATION_TENSOR_SECTION_SPECS.items():
        section = sections.get(section_name, {})
        rows = section.get("tensors", [])
        if len(rows) != spec["tensor_count"]:
            raise SystemExit(
                f"reference {section_name} tensor count is {len(rows)}"
            )
        mapped = {
            row["key"]: {
                "shape": row["shape"],
                "dtype": row["dtype"],
            }
            for row in rows
        }
        if len(mapped) != spec["tensor_count"]:
            raise SystemExit(f"duplicate reference keys in {section_name}")
        metadata[section_name] = mapped
    return metadata


def build_envelope(reference: dict, comparisons: list[dict]) -> tuple[dict, dict]:
    metadata = expected_metadata(reference)
    maxima = {
        section: {
            key: {"max_abs": 0.0, "l2_norm": 0.0}
            for key in rows
        }
        for section, rows in metadata.items()
    }
    comparison_checks = []
    for index, report in enumerate(comparisons, start=1):
        pairwise = report.get("pairwise_against_reference")
        checks = report.get("comparison_checks", {})
        report_passed = (
            isinstance(pairwise, dict)
            and pairwise.get("passed") is True
            and checks
            and all(checks.values())
        )
        comparison_checks.append({
            "index": index,
            "label": report.get("label"),
            "passed": report_passed,
            "checks": checks,
        })
        if not report_passed:
            raise SystemExit(f"comparison {index} did not pass exact invariants")
        for section_name, expected in metadata.items():
            rows = pairwise.get("sections", {}).get(
                section_name, {}
            ).get("tensors", [])
            observed = {row["key"]: row for row in rows}
            if set(observed) != set(expected):
                raise SystemExit(
                    f"comparison {index} {section_name} key coverage mismatch"
                )
            for key, expected_row in expected.items():
                row = observed[key]
                if (
                    row.get("shape") != expected_row["shape"]
                    or row.get("dtype") != expected_row["dtype"]
                    or row.get("finite") is not True
                ):
                    raise SystemExit(
                        f"comparison {index} metadata mismatch for {section_name}:{key}"
                    )
                for metric in ("max_abs", "l2_norm"):
                    value = float(row[metric])
                    if not math.isfinite(value) or value < 0.0:
                        raise SystemExit(
                            f"comparison {index} nonfinite {metric} for {section_name}:{key}"
                        )
                    maxima[section_name][key][metric] = max(
                        maxima[section_name][key][metric], value
                    )

    sections = {}
    section_summaries = {}
    for section_name, spec in driver.CONTINUATION_TENSOR_SECTION_SPECS.items():
        rows = []
        exact_step_tensors = 0
        for key in sorted(metadata[section_name]):
            observed = maxima[section_name][key]
            if section_name == "optimizer_state" and key.endswith("::step"):
                if observed != {"max_abs": 0.0, "l2_norm": 0.0}:
                    raise SystemExit(f"optimizer step drifted during calibration: {key}")
                max_tolerance = 0.0
                l2_tolerance = 0.0
                exact_step_tensors += 1
            else:
                floor = MINIMUM_TOLERANCES[section_name]
                max_tolerance = max(
                    floor["max_abs"], MARGIN_FACTOR * observed["max_abs"]
                )
                l2_tolerance = max(
                    floor["l2_norm"], MARGIN_FACTOR * observed["l2_norm"]
                )
            if (
                max_tolerance > spec["max_abs_hard_cap"]
                or l2_tolerance > spec["l2_norm_hard_cap"]
            ):
                raise SystemExit(
                    f"calibrated tolerance exceeds hard cap for {section_name}:{key}"
                )
            rows.append({
                "key": key,
                "shape": metadata[section_name][key]["shape"],
                "dtype": metadata[section_name][key]["dtype"],
                "observed_max_abs": observed["max_abs"],
                "observed_l2_norm": observed["l2_norm"],
                "max_abs_tolerance": max_tolerance,
                "l2_norm_tolerance": l2_tolerance,
            })
        keys = [row["key"] for row in rows]
        sections[section_name] = {
            "tensor_count": len(rows),
            "key_sha256": driver.canonical_sha(keys),
            "tensors": rows,
        }
        section_summaries[section_name] = {
            "tensor_count": len(rows),
            "exact_step_tensors": exact_step_tensors,
            "largest_observed_max_abs": max(
                row["observed_max_abs"] for row in rows
            ),
            "largest_observed_l2_norm": max(
                row["observed_l2_norm"] for row in rows
            ),
            "largest_max_abs_tolerance": max(
                row["max_abs_tolerance"] for row in rows
            ),
            "largest_l2_norm_tolerance": max(
                row["l2_norm_tolerance"] for row in rows
            ),
        }
    envelope = {
        "schema": driver.CONTINUATION_TENSOR_ENVELOPE_SCHEMA,
        "sections": sections,
    }
    validated = driver.validate_continuation_tensor_envelope(envelope)
    if not validated["passed"]:
        raise SystemExit(f"constructed envelope failed validation: {validated}")
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "experiment": driver.EXPERIMENT,
        "disposable": True,
        "official_training_updates_executed": 0,
        "reference_label": reference.get("label"),
        "independent_fresh_process_comparisons": len(comparisons),
        "margin_policy": {
            "factor_over_largest_observed_reference_pair_drift": MARGIN_FACTOR,
            "minimum_tolerances": MINIMUM_TOLERANCES,
            "optimizer_step_tensors_exact": True,
            "hard_caps": copy.deepcopy(driver.CONTINUATION_TENSOR_SECTION_SPECS),
        },
        "comparison_checks": comparison_checks,
        "section_summaries": section_summaries,
        "envelope_sha256": driver.canonical_sha(envelope),
        "passed": all(row["passed"] for row in comparison_checks)
        and validated["passed"],
    }
    return envelope, evidence


def run(args) -> None:
    calibration_parent = args.output.parent
    diagnostics_root = Path(
        "/workspace/exp2d5c_w2w2_100m/diagnostics"
    ).resolve()
    all_paths = [
        args.base_provenance, args.reference_report, args.reference_snapshot,
        *args.comparison_report, args.probe_source, args.driver_source,
        args.builder_source, args.evidence_output, args.output,
    ]
    if diagnostics_root not in calibration_parent.parents:
        raise SystemExit("calibration directory must be beneath exact diagnostics root")
    if not calibration_parent.is_dir():
        raise SystemExit("calibration output directory does not exist")
    if any(path.parent != calibration_parent for path in all_paths):
        raise SystemExit("every calibration input and output must share one directory")
    if len(set(all_paths)) != len(all_paths):
        raise SystemExit("calibration input/output paths must be distinct")
    if args.output.name != "CALIBRATION_PROVENANCE.json":
        raise SystemExit("final calibration provenance filename is not exact")
    disposable_paths = [
        args.reference_report, args.reference_snapshot,
        *args.comparison_report, args.probe_source, args.driver_source,
        args.builder_source, args.evidence_output,
    ]
    if any("DISPOSABLE" not in path.name for path in disposable_paths):
        raise SystemExit("calibration disposable artifact names are not explicit")
    if args.output.exists() or args.evidence_output.exists():
        raise SystemExit("refusing to overwrite calibration output or evidence")
    implementation_commit = driver.git("rev-parse", "HEAD")
    origin_commit = driver.git("rev-parse", f"origin/{driver.BRANCH}")
    if driver.git("branch", "--show-current") != driver.BRANCH:
        raise SystemExit("calibration must run from the exact 2D5C branch")
    if driver.git("status", "--porcelain"):
        raise SystemExit("calibration implementation worktree must be clean")
    if implementation_commit != origin_commit:
        raise SystemExit("calibration implementation commit is not pushed")
    base_path = require_file(args.base_provenance, "base calibration provenance")
    if driver.sha256(base_path) != BASE_PROVENANCE_SHA256:
        raise SystemExit("base calibration provenance SHA-256 mismatch")
    base = json.loads(base_path.read_text())
    for name, digest in base.get("artifact_sha256", {}).items():
        inherited = calibration_parent / name
        if not inherited.is_file() or driver.sha256(inherited) != digest:
            raise SystemExit(f"missing inherited calibration artifact: {name}")
    disposable = base.get("disposable_update2_checkpoint", {})
    disposable_path = Path(disposable.get("path", ""))
    if not (
        disposable_path.is_file()
        and driver.sha256(disposable_path) == disposable.get("sha256")
        and disposable_path.stat().st_size == disposable.get("bytes")
    ):
        raise SystemExit("inherited disposable update-2 checkpoint is unavailable")
    reference_path = require_file(args.reference_report, "reference report")
    snapshot_path = require_file(args.reference_snapshot, "reference snapshot")
    if "DISPOSABLE" not in reference_path.name or "DISPOSABLE" not in snapshot_path.name:
        raise SystemExit("reference report and snapshot filenames must contain DISPOSABLE")
    reference = load_report(reference_path)
    reference_identity = identity(reference_path)
    snapshot_identity = identity(snapshot_path)
    if reference.get("raw_snapshot") != snapshot_identity:
        raise SystemExit("reference report does not bind the selected raw snapshot")
    raw_snapshot = driver.base.d0.torch_load(snapshot_path, mmap=False)
    raw_summary = driver.continuation_tensor_snapshot_summary(raw_snapshot)
    del raw_snapshot
    gc.collect()
    if raw_summary != reference.get("snapshot_summary"):
        raise SystemExit("raw snapshot content does not match the reference summary")
    comparison_paths = [require_file(path, "comparison report") for path in args.comparison_report]
    if len(comparison_paths) != 7:
        raise SystemExit("exactly seven independent fresh-process comparisons are required")
    if any("DISPOSABLE" not in path.name for path in comparison_paths):
        raise SystemExit("every comparison report filename must contain DISPOSABLE")
    comparisons = [load_report(path) for path in comparison_paths]
    all_reports = [reference, *comparisons]
    if len({report.get("label") for report in all_reports}) != len(all_reports):
        raise SystemExit("probe labels are not unique")
    if len({report.get("run_uuid") for report in all_reports}) != len(all_reports):
        raise SystemExit("probe run UUIDs are not unique")
    if len({report.get("process_id") for report in all_reports}) != len(all_reports):
        raise SystemExit("probe process IDs are not unique")
    uniform_fields = (
        "disposable_checkpoint", "disposable_verification", "source_checkpoint",
        "pod", "runtime", "boundary", "git", "official_run_root",
    )
    expected_git = {
        "branch": driver.BRANCH,
        "head": implementation_commit,
        "origin_branch": origin_commit,
        "clean": True,
    }
    if reference.get("git") != expected_git:
        raise SystemExit("reference probe does not bind the clean pushed implementation")
    for index, report in enumerate(comparisons, start=1):
        if any(report.get(field) != reference.get(field) for field in uniform_fields):
            raise SystemExit(f"comparison {index} provenance differs from reference")
        if report.get("reference_snapshot") != snapshot_identity:
            raise SystemExit(f"comparison {index} targets a different snapshot")
        if report.get("reference_report") != reference_identity:
            raise SystemExit(f"comparison {index} targets a different report")
    probe_source = identity(args.probe_source)
    driver_source = identity(args.driver_source)
    builder_source = identity(args.builder_source)
    live_probe_source = identity(
        Path(__file__).with_name("experiment_2d5c_continuation_probe.py")
    )
    live_driver_source = identity(Path(driver.__file__))
    live_builder_source = identity(Path(__file__))
    if not (
        same_content(probe_source, live_probe_source)
        and same_content(driver_source, live_driver_source)
        and same_content(builder_source, live_builder_source)
    ):
        raise SystemExit("archived calibration source differs from clean live Git source")
    if not all(
        same_content(report.get("probe_source"), probe_source)
        and same_content(report.get("driver_source"), driver_source)
        for report in all_reports
    ):
        raise SystemExit("probe/driver source identity differs across runs")
    fingerprints = [
        driver.canonical_sha(report["pairwise_against_reference"]["sections"])
        for report in comparisons
    ]
    distinct_fingerprint_count = len(set(fingerprints))
    envelope, evidence = build_envelope(reference, comparisons)
    evidence.update({
        "base_provenance": identity(base_path),
        "reference_report": reference_identity,
        "reference_snapshot": snapshot_identity,
        "comparison_reports": [identity(path) for path in comparison_paths],
        "probe_source": probe_source,
        "driver_source": driver_source,
        "builder_source": builder_source,
        "reference_process_id": reference["process_id"],
        "comparison_process_ids": [report["process_id"] for report in comparisons],
        "run_uuids": [report["run_uuid"] for report in all_reports],
        "pairwise_metric_fingerprints": fingerprints,
        "distinct_pairwise_metric_fingerprint_count": distinct_fingerprint_count,
        "uniform_provenance_fields": list(uniform_fields),
        "runtime": reference["runtime"],
        "implementation_commit": implementation_commit,
    })
    driver.durable_json(args.evidence_output, evidence)
    if not evidence["passed"]:
        raise SystemExit("full-tensor calibration evidence did not pass")

    payload = copy.deepcopy(base)
    payload["schema"] = driver.CONTINUATION_CALIBRATION_SCHEMA
    payload["implementation_commit"] = implementation_commit
    payload["runtime"] = reference["runtime"]
    payload["continuation_tensor_envelope"] = envelope
    payload["continuation_tensor_calibration"] = {
        "evidence": identity(args.evidence_output),
        "reference_snapshot": snapshot_identity,
        "reference_report": reference_identity,
        "comparison_count": len(comparison_paths),
        "comparison_report_sha256": [driver.sha256(path) for path in comparison_paths],
        "probe_source": probe_source,
        "driver_source": driver_source,
        "builder_source": builder_source,
        "reference_process_id": reference["process_id"],
        "comparison_process_ids": [report["process_id"] for report in comparisons],
        "run_uuids": [report["run_uuid"] for report in all_reports],
        "pairwise_metric_fingerprints": fingerprints,
        "distinct_pairwise_metric_fingerprint_count": distinct_fingerprint_count,
        "official_training_updates_executed": 0,
        "passed": True,
    }
    inventory = dict(payload.get("artifact_sha256", {}))
    for path in (
        base_path, reference_path, snapshot_path, args.evidence_output.resolve(),
        args.probe_source.resolve(), args.driver_source.resolve(),
        args.builder_source.resolve(), *comparison_paths,
    ):
        path = require_file(path, "calibration artifact")
        if path.parent != args.output.resolve().parent:
            raise SystemExit(
                f"calibration artifact must be beside final provenance: {path}"
            )
        inventory[path.name] = driver.sha256(path)
    payload["artifact_sha256"] = dict(sorted(inventory.items()))
    payload["interpretation"] = (
        "Exact serialized state, data, RNG, optimizer groups and forward losses; "
        "the inherited CUDA backward is non-bitwise before optimizer.step. "
        "A seven-repeat fresh-process calibration now freezes per-tensor "
        "max-absolute and L2 envelopes for all gradients, model parameters and "
        "optimizer state. These limits apply only to disposable continuation "
        "comparison and do not alter official training semantics."
    )
    payload["passed"] = True
    driver.durable_json(args.output, payload)
    print(
        f"EXPERIMENT_2D5C_CONTINUATION_CALIBRATION_V2_PASS {driver.sha256(args.output)}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-provenance", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, action="append", required=True)
    parser.add_argument("--probe-source", type=Path, required=True)
    parser.add_argument("--driver-source", type=Path, required=True)
    parser.add_argument("--builder-source", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "base_provenance", "reference_report", "reference_snapshot",
        "probe_source", "driver_source", "builder_source",
        "evidence_output", "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.comparison_report = [path.resolve() for path in args.comparison_report]
    run(args)


if __name__ == "__main__":
    main()
