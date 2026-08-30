#!/usr/bin/env python3
"""Append-only post-training adjudication for the sealed Experiment 2D5C run.

The official 2D5C training implementation correctly advanced every Adam state
once per update, but its terminal audit incorrectly assumed that all Adam step
counters must equal the lineage-global update.  The accepted 2D3A source has
two intentional step populations: the three gates introduced by 2D3A track
the 2D3A update, while 149 inherited states began that lineage 478 steps ahead.

This tool does not train, mutate a checkpoint, or overwrite evidence.  It is
specific to the exact official artifacts listed below.  It first proves, from
the source and final checkpoint optimizer state dictionaries, that all 152
serialized optimizer states retain their identity and advance by exactly 191.
It independently proves the complete 191-row metric chain.  Only then can it:

* emit a downstream-compatible, distinctly named adjudicated training record;
* adjudicate the frozen driver's legacy failed seal after proving that its only
  false check is the same invalid singleton-step assertion.

PyTorch is imported lazily, after exact checkpoint hashes have been verified,
so the pure adjudication helpers remain testable on hosts without PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT = "2D5C"
BRANCH = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m"
PROTOCOL_SHA256 = "eae21daf3859eb70d3342261d2891fbb4b2114235b1a87450eb2b4d201183b70"
TRAINING_IMPLEMENTATION_COMMIT = "4df3cfaaa486a7162485862ea521e69c47d5147d"

SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
FINAL_CHECKPOINT_SHA256 = "f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe"
CONTROL_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
ARCHITECTURE_FINGERPRINT_C = "019d822dd89986c269e985fba8d1277a15d476dd73a0dac0d8c35e07e7315c12"
SOURCE_NEXT_BATCH = "61dd83544d83c0cf7b4d61005f5a9cf64e2cafa930af1819cba2aae4538e7e61"
SOURCE_NEXT_STREAM = "39f6599f552803150fad33d32aa9c4df5843b058410ff1ba38b5afa469046e97"
FINAL_NEXT_BATCH = "62800455f294aaf110fbfc024abaa601c30f45d96175acb795a4d162d53da097"
FINAL_NEXT_STREAM = "cdfd4afb20c268d69e3e3fbc1c39076af21719ba6d2a6636180f12b1afd5a157"

SOURCE_UPDATES = 1_908
SOURCE_TARGETS = 1_000_341_504
LOCAL_UPDATES = 191
TARGETS_PER_UPDATE = 524_288
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2_099
FINAL_CUMULATIVE_TARGETS = 1_100_480_512
RESTART_LOCAL_UPDATE = 96
PARAMETERS = 124_475_908
MILESTONES = (48, 96, 144, 191)

EXPECTED_SOURCE_STEP_COUNTS = {1_908: 3, 2_386: 149}
EXPECTED_FINAL_STEP_COUNTS = {2_099: 3, 2_577: 149}
EXPECTED_OPTIMIZER_STATE_ENTRIES = 152
EXPECTED_GROUP_COUNTS = {
    "base_decay": 50,
    "base_nodecay": 98,
    "gate": 1,
    "b3_gate": 1,
    "b5_gate": 1,
    "b6_gate": 1,
}

# These hashes identify the one official run being adjudicated.  Exact hashes
# are intentionally code constants rather than values learned from the failed
# record at runtime.
OFFICIAL_ARTIFACT_SHA256 = {
    "training_log": "47510ab50da4c0bcd4af2d859a881aac996b08620cec09a6bcaa0dd03012626c",
    "training_complete_original": "8fef8253f596d668462a8e4c313a762105c63a36bc23f7db3ee25fc9db04579c",
    "training_replay_actual": "6b3754868ee6fd1b2e64cca8bb4e487b2b0475a2232c141163ca80db40d61f35",
    "replay_ledger": "429f1d11b2af285fafab8aaf48341f6098a983b7f89598b1465974f4e969b6c0",
    "replay_audit": "35df41816120a565415b2cb4a5cba9e6bb26101e5340ef7855e6d91b163fc4ff",
    "milestone_manifest": "836a360b6713fa552158442718f35e9bcd15e0a6af0291d1d0b176b61cf7c6c0",
    "midpoint_restart_preexit": "37ff7f8eaa3cb1ad0426fbe02e4791f1b00c3b7b18c4c0dd4f50c07164c6a8a4",
    "midpoint_restart_audit": "e63eb9d7795cef17197ff71569cf25a5cd2ed31a0568fbe6cc22746e9b4affaf",
    "checkpoint_verification_sidecar": "5cc231e0baff11be14f037e8658a03c14846ad71c42a8dcced8a9739224619f8",
    "checkpoint_checksum_sidecar": "cb6a6d4ce9e4293ca83685ef6d2abc5b5f5abefa7a19c1d4b549cd6ab0ece33c",
}

FROZEN_IMPLEMENTATION_SHA256 = {
    "configs/exp2d5c_continuation_calibration_pin.json": "40b120169d493aad95afe66c8d8c6fcefaedee8eeeb4ca954cccb01ad2bcc712",
    "configs/exp2d5c_fixed_writer_b3_b5_w2_matched_100m.json": "9498a65614c2b829d8a64813a0e50acf6dfa13cfc97e6d74d4f2a99b82582c30",
    "scripts/experiment_2d5c.py": "204e589d4dfd68ed2136d70d2470c6bc06a85f5e89b706a7075706d039081b48",
    "scripts/experiment_2d5c_analysis.py": "97a82558057db8ea61feb80e7bf4243301415fe0e002ee529e76dcdd0751ba76",
    "scripts/experiment_2d5c_artifacts.py": "bed151e2dfb84d543baa0dfad4dad51f8193b20a8bf0627cb9e4fd082dc4f783",
    "scripts/experiment_2d5c_build_continuation_calibration.py": "bac58269354a663e884e1d79c6d211fece1b83425d70c02970c94bdddf2e8af6",
    "scripts/experiment_2d5c_complete.py": "a0b08e8aa903badbe031a96a86b2701600cbfeaf8dc891690786ba9bb1aa5a2c",
    "scripts/experiment_2d5c_continuation_probe.py": "2c2a510bd51284570af56be26284b2e4f642f2d07218c82604b2153aa82a32ae",
    "scripts/experiment_2d5c_core.py": "e559ae2fc0a6de880ccb32b1c11613d5ac49e19f8623448dc193f06f0ff9289d",
    "scripts/experiment_2d5c_finalizer.py": "5d220f061ee038360b50ee9dfb07394c7a77e1f35415fad66ab5fb3b7f28341a",
    "scripts/experiment_2d5c_runpod_guard.py": "1665eff37129e097a51b2d46927d1be9c65dadf4e3e43d778eb5e4266bfbadca",
    "scripts/experiment_2d5c_workflow.py": "b973f8c911d8cd5210ded0ed42633f1f445ed6a6759f938d005ef168fcdae305",
}


class AdjudicationError(RuntimeError):
    """Raised when append-only adjudication cannot be proven exactly."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise AdjudicationError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AdjudicationError(
                    f"JSONL row {line_number} is not an object: {path}"
                )
            rows.append(value)
    return rows


def durable_json_exclusive(path: str | Path, payload: Any) -> None:
    """Create one new JSON artifact; never overwrite an existing path."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False
    ) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as error:
        raise AdjudicationError(
            f"append-only output already exists: {destination}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        # The newly created file is evidence even if a later fsync fails.  Do
        # not remove or overwrite it automatically.
        raise


def _git(repo_root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if check and result.returncode:
        raise AdjudicationError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def frozen_implementation_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    filesystem = {
        relative: sha256(root / relative)
        for relative in FROZEN_IMPLEMENTATION_SHA256
    }
    commit_blobs = {}
    for relative in FROZEN_IMPLEMENTATION_SHA256:
        content = subprocess.check_output(
            ["git", "show", f"{TRAINING_IMPLEMENTATION_COMMIT}:{relative}"],
            cwd=root,
        )
        commit_blobs[relative] = hashlib.sha256(content).hexdigest()
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         TRAINING_IMPLEMENTATION_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    checks = {
        "head_is_training_commit": head == TRAINING_IMPLEMENTATION_COMMIT,
        "branch_exact": branch == BRANCH,
        "worktree_clean": status == "",
        "training_commit_is_ancestor": ancestor,
        "filesystem_map_exact": filesystem == FROZEN_IMPLEMENTATION_SHA256,
        "training_commit_map_exact": commit_blobs == FROZEN_IMPLEMENTATION_SHA256,
    }
    return {
        "repo_root": str(root),
        "head": head,
        "branch": branch,
        "worktree_status": status,
        "training_implementation_commit": TRAINING_IMPLEMENTATION_COMMIT,
        "frozen_implementation_sha256": filesystem,
        "training_commit_implementation_sha256": commit_blobs,
        "adjudicator": file_identity(Path(__file__).resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _step_value(value: Any) -> int:
    if hasattr(value, "numel") and callable(value.numel):
        if int(value.numel()) != 1:
            raise AdjudicationError("optimizer step tensor is not scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdjudicationError(f"optimizer step is not numeric: {type(value)}")
    integer = int(value)
    if float(value) != float(integer):
        raise AdjudicationError(f"optimizer step is nonintegral: {value}")
    return integer


def _json_counter(values: Counter[int]) -> dict[str, int]:
    return {str(key): int(values[key]) for key in sorted(values)}


def optimizer_progression_audit(
    source_optimizer: dict[str, Any], final_optimizer: dict[str, Any]
) -> dict[str, Any]:
    """Prove exact optimizer-state identity and +191 progression."""

    source_state = source_optimizer.get("state", {})
    final_state = final_optimizer.get("state", {})
    source_groups = source_optimizer.get("param_groups", [])
    final_groups = final_optimizer.get("param_groups", [])

    source_keys = set(source_state)
    final_keys = set(final_state)
    source_group_params = [
        parameter
        for group in source_groups for parameter in group.get("params", [])
    ]
    final_group_params = [
        parameter
        for group in final_groups for parameter in group.get("params", [])
    ]
    group_topology_exact = (
        len(source_groups) == len(final_groups)
        and all(
            source.get("name") == final.get("name")
            and source.get("params") == final.get("params")
            for source, final in zip(source_groups, final_groups)
        )
    )

    source_steps: dict[Any, int] = {}
    final_steps: dict[Any, int] = {}
    substate_keys_exact = True
    missing_steps = []
    for key in sorted(source_keys | final_keys, key=lambda value: repr(value)):
        source_row = source_state.get(key)
        final_row = final_state.get(key)
        if not isinstance(source_row, dict) or not isinstance(final_row, dict):
            substate_keys_exact = False
            continue
        substate_keys_exact = substate_keys_exact and (
            set(source_row) == set(final_row)
        )
        if "step" not in source_row or "step" not in final_row:
            missing_steps.append(repr(key))
            continue
        source_steps[key] = _step_value(source_row["step"])
        final_steps[key] = _step_value(final_row["step"])

    source_counter = Counter(source_steps.values())
    final_counter = Counter(final_steps.values())
    mismatches = [
        {
            "serialized_parameter_id": repr(key),
            "source_step": source_steps.get(key),
            "final_step": final_steps.get(key),
            "delta": (
                final_steps[key] - source_steps[key]
                if key in source_steps and key in final_steps else None
            ),
        }
        for key in sorted(source_keys | final_keys, key=lambda value: repr(value))
        if key not in source_steps or key not in final_steps
        or final_steps[key] != source_steps[key] + LOCAL_UPDATES
    ]

    per_group = []
    group_count_actual: dict[str, int] = {}
    for source_group, final_group in zip(source_groups, final_groups):
        name = source_group.get("name")
        parameters = source_group.get("params", [])
        delta_counter = Counter(
            final_steps[parameter] - source_steps[parameter]
            for parameter in parameters
            if parameter in source_steps and parameter in final_steps
        )
        group_count_actual[str(name)] = len(parameters)
        per_group.append({
            "name": name,
            "parameter_state_entries": len(parameters),
            "source_step_counts": _json_counter(Counter(
                source_steps[parameter] for parameter in parameters
                if parameter in source_steps
            )),
            "final_step_counts": _json_counter(Counter(
                final_steps[parameter] for parameter in parameters
                if parameter in final_steps
            )),
            "delta_counts": _json_counter(delta_counter),
        })

    checks = {
        "optimizer_dictionaries_present": bool(source_state)
        and bool(final_state) and bool(source_groups) and bool(final_groups),
        "state_entry_count_exact": len(source_keys)
        == len(final_keys) == EXPECTED_OPTIMIZER_STATE_ENTRIES,
        "state_keys_exact": source_keys == final_keys,
        "substate_keys_exact": substate_keys_exact,
        "all_states_have_scalar_steps": not missing_steps
        and len(source_steps) == len(final_steps) == EXPECTED_OPTIMIZER_STATE_ENTRIES,
        "source_group_state_coverage_exact": len(source_group_params)
        == len(set(source_group_params)) == EXPECTED_OPTIMIZER_STATE_ENTRIES
        and set(source_group_params) == source_keys,
        "final_group_state_coverage_exact": len(final_group_params)
        == len(set(final_group_params)) == EXPECTED_OPTIMIZER_STATE_ENTRIES
        and set(final_group_params) == final_keys,
        "group_topology_exact": group_topology_exact,
        "group_counts_exact": group_count_actual == EXPECTED_GROUP_COUNTS,
        "source_step_populations_exact": source_counter
        == Counter(EXPECTED_SOURCE_STEP_COUNTS),
        "final_step_populations_exact": final_counter
        == Counter(EXPECTED_FINAL_STEP_COUNTS),
        "every_state_advanced_exactly_191": not mismatches,
    }
    return {
        "source_step_counts": _json_counter(source_counter),
        "final_step_counts": _json_counter(final_counter),
        "expected_source_step_counts": _json_counter(Counter(
            EXPECTED_SOURCE_STEP_COUNTS
        )),
        "expected_final_step_counts": _json_counter(Counter(
            EXPECTED_FINAL_STEP_COUNTS
        )),
        "state_entry_count": len(source_keys),
        "state_key_sha256": hashlib.sha256(json.dumps(
            sorted(repr(key) for key in source_keys), separators=(",", ":")
        ).encode()).hexdigest(),
        "per_group": per_group,
        "missing_steps": missing_steps,
        "progression_mismatches": mismatches,
        "checks": checks,
        "passed": all(checks.values()),
    }


def training_row_chain_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_summary = sorted(EXPECTED_SOURCE_STEP_COUNTS)
    final_summary = sorted(EXPECTED_FINAL_STEP_COUNTS)
    shape_exact = len(rows) == LOCAL_UPDATES
    local_updates_exact = shape_exact and [
        row.get("local_update") for row in rows
    ] == list(range(1, LOCAL_UPDATES + 1))

    expected_summaries = shape_exact and all(
        row.get("optimizer_steps_before_summary")
        == [step + index for step in source_summary]
        and row.get("optimizer_steps_after_summary")
        == [step + index + 1 for step in source_summary]
        for index, row in enumerate(rows)
    )
    adjacency = shape_exact and all(
        left.get("optimizer_steps_after_summary")
        == right.get("optimizer_steps_before_summary")
        for left, right in zip(rows, rows[1:])
    )
    checks = {
        "rows_exact": shape_exact,
        "local_update_sequence_exact": local_updates_exact,
        "global_update_sequence_exact": shape_exact and all(
            row.get("global_update") == SOURCE_UPDATES + index
            for index, row in enumerate(rows, 1)
        ),
        "target_arithmetic_exact": shape_exact and all(
            row.get("new_targets") == index * TARGETS_PER_UPDATE
            and row.get("cumulative_targets")
            == SOURCE_TARGETS + index * TARGETS_PER_UPDATE
            for index, row in enumerate(rows, 1)
        ),
        "one_optimizer_update_each": shape_exact and all(
            row.get("optimizer_steps") == 1 for row in rows
        ) and sum(row.get("optimizer_steps", 0) for row in rows) == LOCAL_UPDATES,
        "optimizer_success_each": shape_exact and all(
            row.get("optimizer_step_success") is True for row in rows
        ),
        "full_state_increment_flag_each": shape_exact and all(
            row.get("optimizer_step_increment_exact") is True for row in rows
        ),
        "no_scheduler_steps": shape_exact and all(
            row.get("scheduler_steps") == 0 for row in rows
        ),
        "every_row_summary_exact": bool(expected_summaries),
        "row_summary_chain_exact": bool(adjacency),
        "first_before_matches_source": shape_exact
        and rows[0].get("optimizer_steps_before_summary") == source_summary,
        "last_after_matches_final": shape_exact
        and rows[-1].get("optimizer_steps_after_summary") == final_summary,
        "terminal_counts_exact": shape_exact
        and rows[-1].get("global_update") == FINAL_GLOBAL_UPDATE
        and rows[-1].get("cumulative_targets") == FINAL_CUMULATIVE_TARGETS,
        "mandatory_process_boundary_exact": shape_exact
        and len({row.get("process_id") for row in rows[:RESTART_LOCAL_UPDATE]}) == 1
        and len({row.get("process_id") for row in rows[RESTART_LOCAL_UPDATE:]}) == 1
        and rows[RESTART_LOCAL_UPDATE - 1].get("process_id")
        != rows[RESTART_LOCAL_UPDATE].get("process_id"),
    }
    return {
        "rows": len(rows),
        "first_row": rows[0] if rows else None,
        "last_row": rows[-1] if rows else None,
        "source_step_summary": source_summary,
        "final_step_summary": final_summary,
        "checks": checks,
        "passed": all(checks.values()),
    }


def correct_original_training_payload(
    original: dict[str, Any], adjudication: dict[str, Any]
) -> dict[str, Any]:
    false_checks = sorted(
        key for key, value in original.get("checks", {}).items()
        if value is not True
    )
    if original.get("schema") != "experiment_2d5c_training_complete_v1":
        raise AdjudicationError("original training-complete schema is not exact")
    if original.get("experiment") != EXPERIMENT or original.get("passed") is not False:
        raise AdjudicationError("original training-complete outcome is not the known failure")
    if false_checks != ["optimizer_terminal_step_exact"]:
        raise AdjudicationError(
            f"original training failure set is not the adjudicable singleton: {false_checks}"
        )
    if adjudication.get("passed") is not True:
        raise AdjudicationError("optimizer adjudication has not passed")
    result = copy.deepcopy(original)
    result["legacy_checks"] = copy.deepcopy(original["checks"])
    del result["checks"]["optimizer_terminal_step_exact"]
    result["checks"]["optimizer_step_lineage_exact"] = True
    result["passed"] = all(result["checks"].values())
    result["posttrain_adjudication"] = copy.deepcopy(adjudication)
    if result["passed"] is not True:
        raise AdjudicationError("corrected training record is not fully passing")
    return result


def correct_legacy_seal_payload(
    legacy: dict[str, Any], adjudication: dict[str, Any]
) -> dict[str, Any]:
    false_checks = sorted(
        key for key, value in legacy.get("checks", {}).items()
        if value is not True
    )
    if legacy.get("schema") != "experiment_2d5c_final_checkpoint_provenance_v1":
        raise AdjudicationError("legacy seal schema is not exact")
    if legacy.get("experiment") != EXPERIMENT or legacy.get("sealed") is not False:
        raise AdjudicationError("legacy seal is not the known failed seal")
    if false_checks != ["optimizer_step_exact"]:
        raise AdjudicationError(
            f"legacy seal failure set is not the adjudicable singleton: {false_checks}"
        )
    if adjudication.get("passed") is not True:
        raise AdjudicationError("seal adjudication has not passed")
    result = copy.deepcopy(legacy)
    result["legacy_checks"] = copy.deepcopy(legacy["checks"])
    del result["checks"]["optimizer_step_exact"]
    result["checks"]["optimizer_step_lineage_exact"] = True
    result["sealed"] = all(result["checks"].values())
    result["posttrain_adjudication"] = copy.deepcopy(adjudication)
    result["legacy_failed_seal_attempted_at_unix"] = legacy.get("sealed_at_unix")
    result["adjudication_process_id"] = os.getpid()
    result["adjudicated_at_unix"] = time.time()
    result["sealed_at_unix"] = result["adjudicated_at_unix"]
    if result["sealed"] is not True:
        raise AdjudicationError("adjudicated final seal is not fully passing")
    return result


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise AdjudicationError("PyTorch is required to adjudicate checkpoints") from error
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise AdjudicationError(f"checkpoint payload is not a dictionary: {path}")
    return payload


def _official_input_identities(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "source_checkpoint": args.source_checkpoint,
        "final_checkpoint": args.final_checkpoint,
        "training_log": args.training_log,
        "training_complete_original": args.training_complete_original,
        "training_replay_actual": args.training_replay_actual,
        "replay_ledger": args.replay_ledger,
        "replay_audit": args.replay_audit,
        "milestone_manifest": args.milestone_manifest,
        "midpoint_restart_preexit": args.midpoint_restart_preexit,
        "midpoint_restart_audit": args.midpoint_restart_audit,
    }
    identities = {name: file_identity(path) for name, path in paths.items()}
    checkpoint = Path(args.final_checkpoint).resolve()
    identities["checkpoint_verification_sidecar"] = file_identity(
        checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
    )
    identities["checkpoint_checksum_sidecar"] = file_identity(
        checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    )
    return identities


def _official_hash_checks(identities: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "source_checkpoint_sha256": identities["source_checkpoint"]["sha256"]
        == SOURCE_SHA256,
        "final_checkpoint_sha256": identities["final_checkpoint"]["sha256"]
        == FINAL_CHECKPOINT_SHA256,
    }
    checks.update({
        name: identities[name]["sha256"] == expected
        for name, expected in OFFICIAL_ARTIFACT_SHA256.items()
    })
    return checks


def _training_artifact_identity_checks(
    original: dict[str, Any], identities: dict[str, Any]
) -> dict[str, bool]:
    expected = {
        "final_checkpoint": "final_checkpoint",
        "training_log": "training_log",
        "training_replay_actual": "training_replay_actual",
        "replay_ledger": "replay_ledger",
        "replay_audit": "replay_audit",
        "milestone_manifest": "milestone_manifest",
        "midpoint_restart_preexit": "midpoint_restart_preexit",
        "midpoint_restart_audit": "midpoint_restart_audit",
    }
    artifacts = original.get("artifact_identity", {})
    checks = {
        "artifact_key_set_exact": set(artifacts) == set(expected),
    }
    checks.update({
        f"{artifact_name}_identity_exact": artifacts.get(artifact_name)
        == identities[identity_name]
        for artifact_name, identity_name in expected.items()
    })
    return checks


def _checkpoint_payload_checks(
    source: dict[str, Any], final: dict[str, Any]
) -> dict[str, bool]:
    return {
        "source_schema": source.get("schema")
        == "exp2d3a_alternating_integration_pyramid_checkpoint_v1",
        "source_experiment": source.get("experiment_name") == "2D3A-1B",
        "source_updates": source.get("d3a_completed_updates") == SOURCE_UPDATES,
        "source_targets": source.get("d3a_processed_targets") == SOURCE_TARGETS,
        "source_next_batch": source.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "source_next_stream": source.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "final_schema": final.get("schema")
        == "exp2d5c_fixed_writer_b3_b5_w2_checkpoint_v1",
        "final_experiment": final.get("experiment_name") == EXPERIMENT,
        "final_arm": final.get("arm") == "C",
        "parent_sha": final.get("parent_checkpoint_sha256") == SOURCE_SHA256,
        "control_sha": final.get("control_checkpoint_sha256") == CONTROL_SHA256,
        "local_updates": final.get("local_updates") == LOCAL_UPDATES,
        "new_targets": final.get("new_targets") == LOCAL_TARGETS,
        "global_update": final.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": final.get("cumulative_targets")
        == FINAL_CUMULATIVE_TARGETS,
        "architecture": final.get("architecture_fingerprint")
        == ARCHITECTURE_FINGERPRINT_C,
        "parameters": final.get("parameter_count") == PARAMETERS,
        "next_batch": final.get("next_global_batch_sha256") == FINAL_NEXT_BATCH,
        "next_stream": final.get("next_global_batch_stream_sha256") == FINAL_NEXT_STREAM,
        "training_commit": final.get("git_implementation_commit")
        == TRAINING_IMPLEMENTATION_COMMIT,
        "metadata_training_commit": final.get("metadata", {}).get(
            "git_implementation_commit"
        ) == TRAINING_IMPLEMENTATION_COMMIT,
        "model_state_key_set_preserved": set(source.get("model", {}))
        == set(final.get("model", {})),
    }


def _milestone_checks(
    milestones: dict[str, Any], identities: dict[str, Any]
) -> dict[str, bool]:
    final = milestones.get(str(LOCAL_UPDATES), {})
    sidecar = read_json(identities["checkpoint_verification_sidecar"]["path"])
    checksum = Path(
        identities["checkpoint_checksum_sidecar"]["path"]
    ).read_text(encoding="utf-8")
    expected_summaries = {
        "48": [1_956, 2_434],
        "96": [2_004, 2_482],
        "144": [2_052, 2_530],
        "191": [2_099, 2_577],
    }
    return {
        "milestone_keys_exact": set(milestones) == {str(value) for value in MILESTONES},
        "milestone_step_summaries_exact": all(
            milestones.get(update, {}).get("optimizer_step_summary") == summary
            for update, summary in expected_summaries.items()
        ),
        "final_checkpoint_identity": final.get("sha256")
        == identities["final_checkpoint"]["sha256"]
        and final.get("bytes") == identities["final_checkpoint"]["bytes"],
        "final_counts": final.get("local_update") == LOCAL_UPDATES
        and final.get("global_update") == FINAL_GLOBAL_UPDATE
        and final.get("cumulative_targets") == FINAL_CUMULATIVE_TARGETS,
        "final_strict_reopen": final.get("strict_reopen", {}).get("passed") is True,
        "sidecar_exact": sidecar == final,
        "checksum_exact": checksum
        == f"{FINAL_CHECKPOINT_SHA256}  {Path(identities['final_checkpoint']['path']).name}\n",
    }


def _replay_and_restart_checks(
    original: dict[str, Any], identities: dict[str, Any]
) -> dict[str, bool]:
    replay = read_json(identities["replay_audit"]["path"])
    preexit = read_json(identities["midpoint_restart_preexit"]["path"])
    restart = read_json(identities["midpoint_restart_audit"]["path"])
    return {
        "replay_audit_passed": replay.get("passed") is True,
        "replay_rows": replay.get("rows") == LOCAL_UPDATES,
        "replay_ledger_sha": replay.get("ledger_sha256")
        == identities["replay_ledger"]["sha256"],
        "actual_replay_rows": len(read_jsonl(
            identities["training_replay_actual"]["path"]
        )) == LOCAL_UPDATES,
        "midpoint_preexit_summary": preexit.get("optimizer_step_summary")
        == [2_004, 2_482],
        "midpoint_restart_passed": restart.get("passed") is True,
        "midpoint_optimizer_step_check": restart.get("checks", {}).get(
            "optimizer_steps"
        ) is True,
        "midpoint_optimizer_digest_check": restart.get("checks", {}).get(
            "optimizer_digest"
        ) is True,
        "process_evidence_exact": original.get("process_evidence", {}).get(
            "pre_restart_process_id"
        ) == preexit.get("saved_process_id")
        and original.get("process_evidence", {}).get("post_restart_process_id")
        == restart.get("resumed_process_id"),
    }


def _base_training_adjudication(args: argparse.Namespace) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    identities = _official_input_identities(args)
    hash_checks = _official_hash_checks(identities)
    if not all(hash_checks.values()):
        raise AdjudicationError(f"official artifact hash mismatch: {hash_checks}")

    # Unpickle only after the checkpoint bytes match immutable constants.
    source = _load_checkpoint(identities["source_checkpoint"]["path"])
    final = _load_checkpoint(identities["final_checkpoint"]["path"])
    original = read_json(identities["training_complete_original"]["path"])
    rows = read_jsonl(identities["training_log"]["path"])
    milestones = read_json(identities["milestone_manifest"]["path"])

    optimizer = optimizer_progression_audit(
        source.get("optimizer", {}), final.get("optimizer", {})
    )
    row_chain = training_row_chain_audit(rows)
    implementation = frozen_implementation_provenance(args.repo_root)
    identity_checks = _training_artifact_identity_checks(original, identities)
    payload_checks = _checkpoint_payload_checks(source, final)
    milestone_checks = _milestone_checks(milestones, identities)
    restart_checks = _replay_and_restart_checks(original, identities)
    false_checks = sorted(
        key for key, value in original.get("checks", {}).items()
        if value is not True
    )
    original_checks = {
        "schema": original.get("schema")
        == "experiment_2d5c_training_complete_v1",
        "experiment": original.get("experiment") == EXPERIMENT,
        "original_failed": original.get("passed") is False,
        "only_false_check_is_invalid_terminal_step_check": false_checks
        == ["optimizer_terminal_step_exact"],
        "recorded_final_step_summary_is_heterogeneous": original.get(
            "optimizer_evidence", {}
        ).get("step_summary") == [FINAL_GLOBAL_UPDATE, 2_577],
        "recorded_optimizer_digest_matches_milestone": original.get(
            "optimizer_evidence", {}
        ).get("state_sha256") == milestones.get("191", {}).get(
            "optimizer_state_sha256"
        ),
        "recorded_model_digest_matches_milestone": original.get(
            "optimizer_evidence", {}
        ).get("model_state_sha256") == milestones.get("191", {}).get(
            "model_state_sha256"
        ),
    }
    groups = {
        "official_artifact_hashes": {"checks": hash_checks,
                                      "passed": all(hash_checks.values())},
        "frozen_implementation": implementation,
        "original_training_record": {"checks": original_checks,
                                      "passed": all(original_checks.values())},
        "original_artifact_identity": {"checks": identity_checks,
                                       "passed": all(identity_checks.values())},
        "checkpoint_payloads": {"checks": payload_checks,
                                "passed": all(payload_checks.values())},
        "optimizer_progression": optimizer,
        "training_row_chain": row_chain,
        "milestones": {"checks": milestone_checks,
                       "passed": all(milestone_checks.values())},
        "replay_and_restart": {"checks": restart_checks,
                               "passed": all(restart_checks.values())},
    }
    passed = all(group.get("passed") is True for group in groups.values())
    adjudication = {
        "schema": "experiment_2d5c_optimizer_terminal_adjudication_v1",
        "experiment": EXPERIMENT,
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "post-training audit correction only; no training, checkpoint mutation, evaluation, or state restoration",
        "finding": "the singleton [2099] terminal expectation is false for the inherited two-population Adam lineage; every one of 152 states advanced exactly +191",
        "training_implementation_commit": TRAINING_IMPLEMENTATION_COMMIT,
        "original_training_complete": identities["training_complete_original"],
        "input_artifact_identity": identities,
        "groups": groups,
        "passed": passed,
        "adjudicated_at_unix": time.time(),
        "adjudication_process_id": os.getpid(),
    }
    return original, adjudication, identities


def run_adjudicate_training(args: argparse.Namespace) -> None:
    original, adjudication, _ = _base_training_adjudication(args)
    if adjudication["passed"] is not True:
        failures = [
            name for name, row in adjudication["groups"].items()
            if row.get("passed") is not True
        ]
        raise AdjudicationError(f"training adjudication failed: {failures}")
    result = correct_original_training_payload(original, adjudication)
    durable_json_exclusive(args.output, result)
    print("EXPERIMENT_2D5C_TRAINING_ADJUDICATED", flush=True)


def run_adjudicate_seal(args: argparse.Namespace) -> None:
    original, fresh_adjudication, identities = _base_training_adjudication(args)
    training = read_json(args.training_adjudication)
    legacy = read_json(args.legacy_failed_seal)
    training_identity = file_identity(args.training_adjudication)
    legacy_identity = file_identity(args.legacy_failed_seal)

    embedded = training.get("posttrain_adjudication", {})
    legacy_false = sorted(
        key for key, value in legacy.get("checks", {}).items()
        if value is not True
    )
    checks = {
        "fresh_training_adjudication_passed": fresh_adjudication.get("passed") is True,
        "training_record_passed": training.get("passed") is True
        and all(training.get("checks", {}).values()),
        "training_schema_downstream_compatible": training.get("schema")
        == "experiment_2d5c_training_complete_v1",
        "embedded_adjudication_passed": embedded.get("passed") is True,
        "embedded_original_identity_exact": embedded.get(
            "original_training_complete"
        ) == identities["training_complete_original"],
        "embedded_optimizer_progression_exact": embedded.get("groups", {}).get(
            "optimizer_progression", {}
        ).get("passed") is True,
        "legacy_schema": legacy.get("schema")
        == "experiment_2d5c_final_checkpoint_provenance_v1",
        "legacy_unsealed": legacy.get("sealed") is False,
        "legacy_only_false_optimizer_step": legacy_false
        == ["optimizer_step_exact"],
        "legacy_training_identity": legacy.get("artifact_identity", {}).get(
            "training_complete"
        ) == training_identity,
        "legacy_checkpoint_identity": legacy.get("checkpoint_sha256")
        == FINAL_CHECKPOINT_SHA256
        and legacy.get("checkpoint_bytes")
        == identities["final_checkpoint"]["bytes"],
        "legacy_step_summary": legacy.get("optimizer_step_summary")
        == [FINAL_GLOBAL_UPDATE, 2_577],
        "legacy_optimizer_evidence": legacy.get("optimizer_evidence")
        == training.get("optimizer_evidence"),
        "legacy_milestone_binding": legacy.get("milestone_binding", {}).get(
            "passed"
        ) is True,
        "legacy_strict_reopen": legacy.get("checks", {}).get("strict_reopen")
        is True,
        "original_record_still_exact": file_identity(
            args.training_complete_original
        )["sha256"] == OFFICIAL_ARTIFACT_SHA256["training_complete_original"],
    }
    seal_adjudication = {
        "schema": "experiment_2d5c_final_seal_adjudication_v1",
        "experiment": EXPERIMENT,
        "scope": "append-only adjudication of the one known false optimizer-step seal check",
        "training_implementation_commit": TRAINING_IMPLEMENTATION_COMMIT,
        "training_adjudication": training_identity,
        "legacy_failed_seal": legacy_identity,
        "original_training_complete": identities["training_complete_original"],
        "fresh_optimizer_and_row_adjudication": fresh_adjudication,
        "checks": checks,
        "passed": all(checks.values()),
        "adjudicated_at_unix": time.time(),
        "adjudication_process_id": os.getpid(),
    }
    if seal_adjudication["passed"] is not True:
        raise AdjudicationError(
            f"seal adjudication failed: {checks}"
        )
    result = correct_legacy_seal_payload(legacy, seal_adjudication)
    result.setdefault("artifact_identity", {}).update({
        "original_training_complete": identities["training_complete_original"],
        "training_adjudication": training_identity,
        "legacy_failed_seal": legacy_identity,
    })
    durable_json_exclusive(args.output, result)
    print("EXPERIMENT_2D5C_FINAL_CHECKPOINT_ADJUDICATED_AND_SEALED", flush=True)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--training-complete-original", required=True)
    parser.add_argument("--training-replay-actual", required=True)
    parser.add_argument("--replay-ledger", required=True)
    parser.add_argument("--replay-audit", required=True)
    parser.add_argument("--milestone-manifest", required=True)
    parser.add_argument("--midpoint-restart-preexit", required=True)
    parser.add_argument("--midpoint-restart-audit", required=True)
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("adjudicate-training")
    _add_common_arguments(training)
    training.set_defaults(function=run_adjudicate_training)
    seal = commands.add_parser("adjudicate-seal")
    _add_common_arguments(seal)
    seal.add_argument("--training-adjudication", required=True)
    seal.add_argument("--legacy-failed-seal", required=True)
    seal.set_defaults(function=run_adjudicate_seal)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.function(args)
    except AdjudicationError as error:
        print(f"EXPERIMENT_2D5C_POSTTRAIN_ADJUDICATION_FAILED: {error}",
              file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
