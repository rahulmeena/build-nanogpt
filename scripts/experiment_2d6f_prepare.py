#!/usr/bin/env python3
"""Freeze the 2D6F fresh panel and recover prior paired evidence without PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPERIMENT = "2D6F"
PANEL_NAME = "fresh disjoint confirmation panel"
PANEL_SEED = 2_026_090_201
REUSED_BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
BATCH_SIZE = 64
SEQUENCE_LENGTH = 1024
PANEL_BATCHES = 32
PANEL_SEQUENCES = 2048
PANEL_TARGETS = 2_097_152
DATASET_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
FIXED_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
NEW_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
REUSED_ESTIMATE = 0.000060215693
REUSED_CI = (-0.000008212805, 0.000129402616)


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    return digest.hexdigest()


def aggregate_hashes(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def batch_arrays(tokens, start):
    count = BATCH_SIZE * SEQUENCE_LENGTH
    x = np.asarray(tokens[start : start + count], dtype=np.int64).reshape(
        BATCH_SIZE, SEQUENCE_LENGTH
    )
    y = np.asarray(tokens[start + 1 : start + count + 1], dtype=np.int64).reshape(
        BATCH_SIZE, SEQUENCE_LENGTH
    )
    return x, y


def batch_identity(x, y):
    return {
        "input_sha256": bytes_sha256(x),
        "target_sha256": bytes_sha256(y),
        "combined_sha256": bytes_sha256(x, y),
    }


def sequence_rows(x, y, dataset_start, batch_index=None):
    rows = []
    for sequence_index in range(BATCH_SIZE):
        start = int(dataset_start) + sequence_index * SEQUENCE_LENGTH
        row = {
            "sequence_index": sequence_index,
            "dataset_input_span": [start, start + SEQUENCE_LENGTH - 1],
            "dataset_target_span": [start + 1, start + SEQUENCE_LENGTH],
            "input_sha256": bytes_sha256(x[sequence_index]),
            "target_sha256": bytes_sha256(y[sequence_index]),
            "combined_sha256": bytes_sha256(x[sequence_index], y[sequence_index]),
            "targets": SEQUENCE_LENGTH,
        }
        if batch_index is not None:
            row["batch_index"] = int(batch_index)
        rows.append(row)
    return rows


def spans_intersect(left, right):
    return max(int(left[0]), int(right[0])) < min(int(left[1]), int(right[1]))


def canonical_panel(label, path, tokens):
    value = read_json(path)
    indices = value.get("batch_indices_in_evaluation_order")
    if indices is None:
        indices = list(range(int(value["start_batch"]), int(value["start_batch"]) + int(value["batches"])))
    identities, sequences, spans = [], [], []
    for index in indices:
        start = int(index) * BATCH_SIZE * SEQUENCE_LENGTH
        x, y = batch_arrays(tokens, start)
        identities.append(batch_identity(x, y))
        sequences.extend(sequence_rows(x, y, start, int(index)))
        spans.append([start + 1, start + BATCH_SIZE * SEQUENCE_LENGTH + 1])
    expected_identities = value["batch_identities"]
    expected_aggregate = value.get("panel_sha256", value.get("subset_sha256"))
    observed_aggregate = aggregate_hashes(row["combined_sha256"] for row in identities)
    recorded_sequences = value.get("sequence_identities")
    recorded_sequence_exact = True
    if recorded_sequences is not None:
        recorded_sequence_exact = [row["combined_sha256"] for row in recorded_sequences] == [
            row["combined_sha256"] for row in sequences
        ]
    checks = {
        "batch_identity_count": len(expected_identities) == len(identities),
        "batch_identities_replayed_exactly": expected_identities == identities,
        "aggregate_sha256_reproduced": expected_aggregate == observed_aggregate,
        "recorded_sequence_hashes_reproduced": recorded_sequence_exact,
    }
    if not all(checks.values()):
        raise SystemExit(f"historical panel identity failure for {label}: {checks}")
    return {
        "label": label,
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": file_sha256(path),
        "panel_sha256": expected_aggregate,
        "batch_indices": [int(index) for index in indices],
        "batch_identities": identities,
        "sequence_identities": sequences,
        "target_spans_half_open": spans,
        "identity_checks": checks,
        "identity_verified": True,
    }


def offset_panel(label, path, tokens):
    value = read_json(path)
    selection = value["selection"]
    start = int(selection["start_token_offset"])
    identities, sequences, spans = [], [], []
    for ordinal in range(int(selection["batch_count"])):
        batch_start = start + ordinal * BATCH_SIZE * SEQUENCE_LENGTH
        x, y = batch_arrays(tokens, batch_start)
        identities.append(batch_identity(x, y))
        sequences.extend(sequence_rows(x, y, batch_start))
        spans.append([batch_start + 1, batch_start + BATCH_SIZE * SEQUENCE_LENGTH + 1])
    checks = {
        "dataset_sha256": value["validation_shard_sha256"] == DATASET_SHA256,
        "batch_identities_replayed_exactly": identities == selection["batch_identities"],
        "aggregate_sha256_reproduced": aggregate_hashes(
            row["combined_sha256"] for row in identities
        ) == selection["subset_sha256"],
        "sequence_hashes_replayed_exactly": [row["combined_sha256"] for row in sequences]
        == [row["combined_sha256"] for row in selection["sequence_identities"]],
    }
    if not all(checks.values()):
        raise SystemExit(f"historical offset panel identity failure for {label}: {checks}")
    return {
        "label": label,
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": file_sha256(path),
        "panel_sha256": selection["subset_sha256"],
        "batch_indices": None,
        "batch_identities": identities,
        "sequence_identities": sequences,
        "target_spans_half_open": spans,
        "identity_checks": checks,
        "identity_verified": True,
    }


def paired_bootstrap(values, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != PANEL_SEQUENCES:
        raise SystemExit("bootstrap requires 2,048 paired sequences")
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(250, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(0, len(values), size=(count, len(values)), dtype=np.int32)
        distribution[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return {
        "estimate": float(values.mean()),
        "lower_95": float(np.percentile(distribution, 2.5)),
        "upper_95": float(np.percentile(distribution, 97.5)),
        "standard_error": float(distribution.std(ddof=1)),
    }


def recover_reused(repo, output):
    prior = repo / "results/experiment_2d6_b6_w1024_no_b7_recurrence_matched_100m"
    losses_path = prior / "LARGE_PAIRED_LOSSES.json"
    manifest_path = prior / "PANEL_MANIFEST_LARGE.json"
    losses = read_json(losses_path)
    manifest = read_json(manifest_path)
    fixed = np.asarray(losses["fixed_real"], dtype=np.float64)
    new = np.asarray(losses["new_real"], dtype=np.float64)
    result = paired_bootstrap(fixed - new, REUSED_BOOTSTRAP_SEED)
    checks = {
        "sequence_count": len(fixed) == len(new) == PANEL_SEQUENCES,
        "panel_sha_exact": losses["panel_sha256"] == manifest["panel_sha256"],
        "point_estimate_reproduced": abs(result["estimate"] - REUSED_ESTIMATE) < 5e-13,
        "lower_ci_reproduced": abs(result["lower_95"] - REUSED_CI[0]) < 5e-13,
        "upper_ci_reproduced": abs(result["upper_95"] - REUSED_CI[1]) < 5e-13,
    }
    if not all(checks.values()):
        raise SystemExit(f"reused-panel reproduction failed: {checks}, {result}")
    artifact = {
        "experiment": EXPERIMENT,
        "label": "reused sealed matched panel",
        "source_loss_artifact": str(losses_path.resolve()),
        "source_loss_artifact_sha256": file_sha256(losses_path),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "panel_sha256": manifest["panel_sha256"],
        "paired_sequences": PANEL_SEQUENCES,
        "targets_per_condition": PANEL_TARGETS,
        "fixed_checkpoint_sha256": FIXED_SHA256,
        "new_checkpoint_sha256": NEW_SHA256,
        "bootstrap": {
            "method": "paired per-sequence percentile bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": REUSED_BOOTSTRAP_SEED,
            **result,
        },
        "checks": checks,
        "passed": True,
    }
    durable_json(output / "REUSED_PANEL_IDENTITY.json", artifact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if file_sha256(dataset) != DATASET_SHA256:
        raise SystemExit("canonical validation dataset SHA mismatch")
    tokens = np.load(dataset, mmap_mode="r")

    specs = [
        ("immutable_mechanism_core", repo / "results/experiment_2d5c_fixed_writer_b3_b5_w2_matched_100m/PANEL_MANIFEST_CORE.json", "canonical"),
        ("2d2fg_recorded_validation_subset", repo / "configs/exp2d2fg_c1_validation_subset_manifest.json", "offset"),
        ("2d3a_500m_large", repo / "results/experiment_2d3a_alternating_integration_pyramid_500m/m500_large_confirmation_subset_manifest.json", "canonical"),
        ("2d3a_1b_large", repo / "results/experiment_2d3a_alternating_integration_pyramid_1b/m1000_fresh_subset_manifest.json", "canonical"),
        ("2d4a_100m_large", repo / "results/experiment_2d4a_matched_source_depth_routing/large_confirmation_subset_manifest.json", "canonical"),
        ("2d4a_250m_large", repo / "results/experiment_2d4a_matched_source_depth_routing_250m/large_250m_subset_manifest.json", "canonical"),
        ("2d5c_large", repo / "results/experiment_2d5c_fixed_writer_b3_b5_w2_matched_100m/PANEL_MANIFEST_LARGE.json", "canonical"),
        ("2d6_reused_large", repo / "results/experiment_2d6_b6_w1024_no_b7_recurrence_matched_100m/PANEL_MANIFEST_LARGE.json", "canonical"),
    ]
    historical = [
        offset_panel(label, path, tokens) if kind == "offset" else canonical_panel(label, path, tokens)
        for label, path, kind in specs
    ]
    forbidden_batch_hashes = {
        row["combined_sha256"] for panel in historical for row in panel["batch_identities"]
    }
    forbidden_sequence_hashes = {
        row["combined_sha256"] for panel in historical for row in panel["sequence_identities"]
    }
    forbidden_spans = [span for panel in historical for span in panel["target_spans_half_open"]]
    available_batches = (int(tokens.shape[0]) - 1) // (BATCH_SIZE * SEQUENCE_LENGTH)
    order = np.random.default_rng(PANEL_SEED).permutation(available_batches)
    selected, identities, sequences, spans = [], [], [], []
    selected_sequence_hashes = set()
    for raw_index in order:
        index = int(raw_index)
        start = index * BATCH_SIZE * SEQUENCE_LENGTH
        span = [start + 1, start + BATCH_SIZE * SEQUENCE_LENGTH + 1]
        if any(spans_intersect(span, old) for old in forbidden_spans):
            continue
        x, y = batch_arrays(tokens, start)
        identity = batch_identity(x, y)
        rows = sequence_rows(x, y, start, index)
        hashes = {row["combined_sha256"] for row in rows}
        if identity["combined_sha256"] in forbidden_batch_hashes:
            continue
        if hashes & forbidden_sequence_hashes or hashes & selected_sequence_hashes:
            continue
        selected.append(index)
        identities.append(identity)
        sequences.extend(rows)
        spans.append(span)
        selected_sequence_hashes.update(hashes)
        if len(selected) == PANEL_BATCHES:
            break
    if len(selected) != PANEL_BATCHES:
        raise SystemExit("could not construct exactly one 32-batch fresh panel")

    panel_hash = aggregate_hashes(row["combined_sha256"] for row in identities)
    manifest = {
        "experiment": EXPERIMENT,
        "panel_name": PANEL_NAME,
        "dataset": "edu_fineweb10B/edufineweb_val_000000.npy",
        "dataset_local_source": str(dataset),
        "dataset_sha256": DATASET_SHA256,
        "dataset_split": "validation",
        "token_count": int(tokens.shape[0]),
        "selection_seed": PANEL_SEED,
        "selection_algorithm": "numpy.default_rng(seed).permutation(all complete canonical 64x1024 validation batches); reject any prior target-span, batch-hash, or recovered sequence-hash overlap; accept first 32",
        "candidate_panels_constructed": 1,
        "checkpoint_losses_inspected_during_selection": False,
        "sealed_before_checkpoint_loading": True,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_indices_in_evaluation_order": selected,
        "canonical_target_spans_half_open": spans,
        "batch_identities": identities,
        "sequence_identities": sequences,
        "sequence_count": len(sequences),
        "targets_per_sequence": SEQUENCE_LENGTH,
        "targets_per_condition": len(sequences) * SEQUENCE_LENGTH,
        "panel_sha256": panel_hash,
    }
    prior_rows = {}
    for panel in historical:
        intersections = [
            {"fresh": fresh, "prior": prior}
            for fresh in spans
            for prior in panel["target_spans_half_open"]
            if spans_intersect(fresh, prior)
        ]
        prior_batch_hashes = {row["combined_sha256"] for row in panel["batch_identities"]}
        prior_sequence_hashes = {row["combined_sha256"] for row in panel["sequence_identities"]}
        row = {
            "manifest": panel["manifest"],
            "manifest_sha256": panel["manifest_sha256"],
            "panel_sha256": panel["panel_sha256"],
            "historical_identity_verified": panel["identity_verified"],
            "target_span_intersections": intersections,
            "batch_hash_intersection": sorted(
                {item["combined_sha256"] for item in identities} & prior_batch_hashes
            ),
            "sequence_hash_intersection": sorted(selected_sequence_hashes & prior_sequence_hashes),
        }
        row["verified_disjoint"] = (
            row["historical_identity_verified"]
            and not row["target_span_intersections"]
            and not row["batch_hash_intersection"]
            and not row["sequence_hash_intersection"]
        )
        prior_rows[panel["label"]] = row
    training_shards = read_json(
        repo / "results/experiment_2d6_b6_w1024_no_b7_recurrence_matched_100m/PANEL_MANIFEST_LARGE.json"
    )["training_shards"]
    internal_span_overlap = [
        [left, right]
        for i, left in enumerate(spans)
        for right in spans[i + 1 :]
        if spans_intersect(left, right)
    ]
    disjointness = {
        "experiment": EXPERIMENT,
        "panel_name": PANEL_NAME,
        "panel_sha256": panel_hash,
        "dataset_sha256": DATASET_SHA256,
        "training_split": {
            "accepted_training_loader_shards": training_shards,
            "validation_shard_absent_from_training_loader": all(
                Path(path).name != dataset.name for path in training_shards
            ),
            "dataset_span_identity": "validation shard SHA-256 plus exact half-open token offsets; training loader contains only train shards",
            "verified": all(Path(path).name != dataset.name for path in training_shards),
        },
        "prior_panels": prior_rows,
        "internal": {
            "unique_batch_indices": len(set(selected)) == PANEL_BATCHES,
            "unique_batch_hashes": len({row["combined_sha256"] for row in identities}) == PANEL_BATCHES,
            "unique_sequence_hashes": len(selected_sequence_hashes) == PANEL_SEQUENCES,
            "overlapping_target_spans": internal_span_overlap,
            "no_overlapping_target_spans": not internal_span_overlap,
        },
    }
    disjointness["all_required_disjointness_passed"] = (
        disjointness["training_split"]["verified"]
        and all(row["verified_disjoint"] for row in prior_rows.values())
        and all(
            value for key, value in disjointness["internal"].items()
            if key != "overlapping_target_spans"
        )
    )
    if not disjointness["all_required_disjointness_passed"]:
        raise SystemExit("fresh-panel disjointness audit failed")

    scope = {
        "experiment": EXPERIMENT,
        "description": "2d6-fresh-panel-zero-training-confirmation",
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameter_updates": 0,
        "scheduler_steps": 0,
        "training_targets": 0,
        "new_checkpoints": 0,
        "fresh_panels": 1,
        "authorized_fresh_conditions": ["FRESH_FIXED_REAL", "FRESH_NEW_REAL"],
        "additional_controls_prohibited": True,
        "additional_panels_prohibited": True,
        "training_prohibited": True,
    }
    durable_json(output / "SCOPE_LOCK.json", scope)
    durable_json(output / "FRESH_PANEL_MANIFEST.json", manifest)
    durable_json(output / "FRESH_PANEL_DISJOINTNESS_AUDIT.json", disjointness)
    recover_reused(repo, output)
    print("EXPERIMENT_2D6F_PANEL_FROZEN", panel_hash)
    print("batch_indices", selected)


if __name__ == "__main__":
    main()
