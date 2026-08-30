#!/usr/bin/env python3
"""Deterministic artifact helpers for Experiment 2D5C.

This module is intentionally independent of the training driver and of Torch.
It contains the small, auditable transformations used to freeze the replay
ledger, recurrent-control permutation, and validation-panel manifest.  Tensor
inputs are accepted by duck typing so the same code is easy to exercise in
CPU-only tests.

Intervals in this module are half open: ``[start, end)``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


REPLAY_CHAIN_DOMAIN = "experiment-2d5c/replay-ledger/v1"
DERANGEMENT_DOMAIN = "experiment-2d5c/path-consistent-cyclic-derangement/v1"
PANEL_SELECTION_DOMAIN = "experiment-2d5c/validation-panel/v1"
MANIFEST_HASH_FIELD = "manifest_sha256"
GENESIS_CHAIN_SHA256 = "0" * 64


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 JSON encoding used by 2D5C artifacts.

    The encoder rejects NaN and infinities rather than permitting their
    non-standard spellings.  A trailing newline is deliberately absent so a
    hash always covers exactly the canonical JSON value.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    value = str(value)
    if len(value) != 64:
        raise ValueError(f"{label} is not a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a SHA-256 hex digest") from exc
    return value.lower()


def _without_hash_fields(value: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for field in fields:
        result.pop(field, None)
    return result


def manifest_sha256(
    manifest: Mapping[str, Any], *, hash_field: str = MANIFEST_HASH_FIELD
) -> str:
    """Hash a manifest, excluding its top-level self-hash field."""

    return canonical_json_sha256(_without_hash_fields(manifest, (hash_field,)))


def with_manifest_sha256(
    manifest: Mapping[str, Any], *, hash_field: str = MANIFEST_HASH_FIELD
) -> dict[str, Any]:
    """Return a deep copy with a reproducible top-level self-hash."""

    result = _without_hash_fields(manifest, (hash_field,))
    result[hash_field] = manifest_sha256(result, hash_field=hash_field)
    return result


def verify_manifest_sha256(
    manifest: Mapping[str, Any], *, hash_field: str = MANIFEST_HASH_FIELD
) -> bool:
    observed = manifest.get(hash_field)
    if not isinstance(observed, str):
        return False
    try:
        observed = _require_sha256(observed, hash_field)
    except ValueError:
        return False
    return observed == manifest_sha256(manifest, hash_field=hash_field)


def rolling_replay_chain_sha256(
    previous_sha256: str | None,
    row: Mapping[str, Any],
    *,
    domain: str = REPLAY_CHAIN_DOMAIN,
) -> str:
    """Extend the replay chain with one row using explicit domain separation."""

    previous = GENESIS_CHAIN_SHA256 if previous_sha256 is None else previous_sha256
    previous = _require_sha256(previous, "previous replay-chain hash")
    payload = canonical_json_bytes(
        _without_hash_fields(row, ("previous_chain_sha256", "chain_sha256"))
    )
    digest = hashlib.sha256()
    digest.update(str(domain).encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(previous))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def chain_replay_rows(
    rows: Iterable[Mapping[str, Any]], *, domain: str = REPLAY_CHAIN_DOMAIN
) -> tuple[list[dict[str, Any]], str]:
    """Copy rows, attach their links, and return rows plus the terminal hash."""

    chained: list[dict[str, Any]] = []
    previous = GENESIS_CHAIN_SHA256
    for source_row in rows:
        row = _without_hash_fields(
            source_row, ("previous_chain_sha256", "chain_sha256")
        )
        row["previous_chain_sha256"] = previous
        row["chain_sha256"] = rolling_replay_chain_sha256(
            previous, row, domain=domain
        )
        previous = row["chain_sha256"]
        chained.append(row)
    return chained, previous


def verify_replay_chain(
    rows: Iterable[Mapping[str, Any]],
    terminal_sha256: str | None = None,
    *,
    domain: str = REPLAY_CHAIN_DOMAIN,
) -> bool:
    previous = GENESIS_CHAIN_SHA256
    for row in rows:
        if row.get("previous_chain_sha256") != previous:
            return False
        expected = rolling_replay_chain_sha256(previous, row, domain=domain)
        if row.get("chain_sha256") != expected:
            return False
        previous = expected
    if terminal_sha256 is not None:
        try:
            terminal_sha256 = _require_sha256(
                terminal_sha256, "terminal replay-chain hash"
            )
        except ValueError:
            return False
        if previous != terminal_sha256:
            return False
    return True


def seeded_path_consistent_cyclic_derangement(
    size: int,
    seed: int,
    *,
    paths: Sequence[str] = ("B3", "B5"),
    domain: str = DERANGEMENT_DOMAIN,
) -> dict[str, Any]:
    """Build one seeded, single-cycle donor permutation shared by all paths.

    Rows are ordered by a domain-separated digest and connected into one
    directed cycle.  ``donor_permutation[recipient]`` is the donor row.  One
    shared list is recorded for B3 and B5, preventing link-specific shuffles.
    """

    size = int(size)
    seed = int(seed)
    if size < 2:
        raise ValueError("a derangement requires at least two rows")
    path_names = tuple(str(path) for path in paths)
    if not path_names or len(set(path_names)) != len(path_names):
        raise ValueError("paths must be a non-empty sequence of unique names")

    def rank(row: int) -> bytes:
        material = canonical_json_bytes(
            {"domain": str(domain), "seed": seed, "size": size, "row": row}
        )
        return hashlib.sha256(material).digest()

    cycle = sorted(range(size), key=lambda row: (rank(row), row))
    donors = [-1] * size
    for position, recipient in enumerate(cycle):
        donors[recipient] = cycle[(position + 1) % size]

    if sorted(donors) != list(range(size)):
        raise AssertionError("internal error: donor mapping is not bijective")
    if any(recipient == donor for recipient, donor in enumerate(donors)):
        raise AssertionError("internal error: donor mapping has a fixed point")

    permutation_hash = canonical_json_sha256(donors)
    path_permutations = {path: list(donors) for path in path_names}
    result = {
        "schema": "exp2d5c_path_consistent_cyclic_derangement_v1",
        "algorithm": "sha256-ranked-single-cycle",
        "domain": str(domain),
        "seed": seed,
        "size": size,
        "cycle_order": cycle,
        "donor_permutation": donors,
        "donor_permutation_sha256": permutation_hash,
        "paths": list(path_names),
        "path_permutations": path_permutations,
        "path_consistent": all(
            path_permutations[path] == donors for path in path_names
        ),
        "bijective": True,
        "no_fixed_points": True,
        "single_cycle": True,
    }
    return with_manifest_sha256(result)


# A short alias is convenient at call sites while the long name remains
# unambiguous in frozen scientific artifacts.
seeded_cyclic_derangement = seeded_path_consistent_cyclic_derangement


def _tensor_bytes(value: Any) -> bytes:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    contiguous = getattr(current, "contiguous", None)
    if callable(contiguous):
        current = contiguous()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    tobytes = getattr(current, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    if isinstance(current, (bytes, bytearray, memoryview)):
        return bytes(current)
    raise TypeError("microbatch values must expose deterministic raw bytes")


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dimension) for dimension in shape]


def _tensor_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _tensor_numel(value: Any) -> int | None:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    shape = _tensor_shape(value)
    if shape is None:
        return None
    return math.prod(shape)


def microbatch_identity(inputs: Any, targets: Any) -> dict[str, Any]:
    """Return the project-compatible raw-byte identity for one microbatch."""

    input_bytes = _tensor_bytes(inputs)
    target_bytes = _tensor_bytes(targets)
    combined = hashlib.sha256()
    combined.update(input_bytes)
    combined.update(target_bytes)
    return {
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "combined_sha256": combined.hexdigest(),
        "input_shape": _tensor_shape(inputs),
        "target_shape": _tensor_shape(targets),
        "input_dtype": _tensor_dtype(inputs),
        "target_dtype": _tensor_dtype(targets),
        "target_count": _tensor_numel(targets),
    }


def aggregate_payload_hashes(values: Iterable[str]) -> str:
    """Match the project's aggregate hash: SHA-256 over binary digests."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(_require_sha256(value, "payload hash")))
    return digest.hexdigest()


def _cursor_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a stable logical cursor without discarding shard identity."""

    cursor: dict[str, Any] = {}
    for key in (
        "current_shard",
        "current_shard_filename",
        "current_position",
        "process_rank",
        "num_processes",
    ):
        if key in state:
            cursor[key] = copy.deepcopy(state[key])
    shards = state.get("shards", state.get("paths"))
    if shards is not None:
        shard_list = list(shards)
        cursor["shard_count"] = len(shard_list)
        cursor["shard_manifest_sha256"] = canonical_json_sha256(shard_list)
        shard_index = cursor.get("current_shard")
        if isinstance(shard_index, int) and 0 <= shard_index < len(shard_list):
            cursor["current_shard_source"] = shard_list[shard_index]
    if "current_shard" not in cursor or "current_position" not in cursor:
        # A custom loader may use different cursor fields.  Retaining the whole
        # state is safer than manufacturing an incomplete provenance record.
        cursor["loader_state"] = copy.deepcopy(dict(state))
    return cursor


def _lookup_expected(
    values: Sequence[str] | Mapping[int, str] | None,
    local_update: int,
    label: str,
) -> str | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        key: int | str
        if local_update in values:
            key = local_update
        elif str(local_update) in values:
            key = str(local_update)
        else:
            raise ValueError(f"missing {label} for local update {local_update}")
        return _require_sha256(values[key], label)
    index = local_update - 1
    if index < 0 or index >= len(values):
        raise ValueError(f"missing {label} for local update {local_update}")
    return _require_sha256(values[index], label)


def _default_pass_count(global_update: int) -> int:
    return 3 if int(global_update) % 32 == 0 else 2


def build_replay_ledger(
    loader: Any,
    *,
    accumulation: int,
    updates: int = 191,
    inherited_global_update: int = 1_908,
    inherited_cumulative_targets: int = 1_000_341_504,
    expected_targets_per_update: int | None = 524_288,
    pass_count_fn: Callable[[int], int] | None = None,
    expected_batch_hashes: Sequence[str] | Mapping[int, str] | None = None,
    expected_stream_hashes: Sequence[str] | Mapping[int, str] | None = None,
    chain_domain: str = REPLAY_CHAIN_DOMAIN,
) -> dict[str, Any]:
    """Replay an ``ExplicitShardLoader`` clone into an enriched ledger.

    ``pass_count_fn`` receives the restored *global* update.  The original
    loader is never advanced.  Logical-batch and stream hashes exactly match
    the accepted project's definitions:

    * batch: SHA-256 over the binary per-microbatch combined digests;
    * stream: SHA-256 over ``x_stream_sha_hex + y_stream_sha_hex`` as ASCII.
    """

    accumulation = int(accumulation)
    updates = int(updates)
    if accumulation < 1 or updates < 1:
        raise ValueError("accumulation and updates must be positive")
    clone_method = getattr(loader, "clone", None)
    state_method = getattr(loader, "state_dict", None)
    if not callable(clone_method) or not callable(state_method):
        raise TypeError("loader must provide clone(), state_dict(), and next_batch()")

    original_state = copy.deepcopy(state_method())
    replay = clone_method()
    if replay is loader:
        raise ValueError("loader.clone() returned the original loader")
    if copy.deepcopy(replay.state_dict()) != original_state:
        raise ValueError("loader clone does not begin at the original cursor")

    pass_count_fn = pass_count_fn or _default_pass_count
    rows: list[dict[str, Any]] = []
    cumulative_targets = int(inherited_cumulative_targets)

    for local_update in range(1, updates + 1):
        global_update = int(inherited_global_update) + local_update
        start_state = copy.deepcopy(replay.state_dict())
        microbatches: list[dict[str, Any]] = []
        input_stream_digest = hashlib.sha256()
        target_stream_digest = hashlib.sha256()
        payload_hashes: list[str] = []
        observed_targets = 0
        targets_known = True

        for microbatch_index in range(accumulation):
            micro_start = _cursor_from_state(copy.deepcopy(replay.state_dict()))
            inputs, targets = replay.next_batch()
            micro_end = _cursor_from_state(copy.deepcopy(replay.state_dict()))
            input_bytes = _tensor_bytes(inputs)
            target_bytes = _tensor_bytes(targets)
            identity = microbatch_identity(inputs, targets)
            payload_hashes.append(identity["combined_sha256"])
            input_stream_digest.update(input_bytes)
            target_stream_digest.update(target_bytes)
            target_count = identity["target_count"]
            if target_count is None:
                targets_known = False
            else:
                observed_targets += int(target_count)
            microbatches.append(
                {
                    "microbatch_index": microbatch_index,
                    "order": microbatch_index + 1,
                    "start_cursor": micro_start,
                    "end_cursor": micro_end,
                    **identity,
                }
            )

        batch_hash = aggregate_payload_hashes(payload_hashes)
        stream_hash = hashlib.sha256(
            (
                input_stream_digest.hexdigest()
                + target_stream_digest.hexdigest()
            ).encode("ascii")
        ).hexdigest()
        expected_batch = _lookup_expected(
            expected_batch_hashes, local_update, "logical global-batch hash"
        )
        expected_stream = _lookup_expected(
            expected_stream_hashes, local_update, "logical global-batch stream hash"
        )
        if expected_batch is not None and batch_hash != expected_batch:
            raise ValueError(
                f"logical global-batch hash mismatch at local update {local_update}: "
                f"{batch_hash} != {expected_batch}"
            )
        if expected_stream is not None and stream_hash != expected_stream:
            raise ValueError(
                f"logical stream hash mismatch at local update {local_update}: "
                f"{stream_hash} != {expected_stream}"
            )

        target_count = (
            observed_targets
            if targets_known
            else (
                None
                if expected_targets_per_update is None
                else int(expected_targets_per_update)
            )
        )
        if target_count is None:
            raise ValueError(
                "target count is not inferable; provide expected_targets_per_update"
            )
        if (
            expected_targets_per_update is not None
            and target_count != int(expected_targets_per_update)
        ):
            raise ValueError(
                f"target-count mismatch at local update {local_update}: "
                f"{target_count} != {int(expected_targets_per_update)}"
            )
        cumulative_targets += target_count
        pass_count = int(pass_count_fn(global_update))
        if pass_count not in (2, 3):
            raise ValueError(
                f"invalid pass count {pass_count} at global update {global_update}"
            )

        rows.append(
            {
                "local_update": local_update,
                "global_update": global_update,
                "start_cursor": _cursor_from_state(start_state),
                "end_cursor": _cursor_from_state(
                    copy.deepcopy(replay.state_dict())
                ),
                "logical_global_batch_sha256": batch_hash,
                "logical_global_batch_stream_sha256": stream_hash,
                "microbatch_count": accumulation,
                "microbatches": microbatches,
                "target_count": target_count,
                "pass_count": pass_count,
                "cumulative_target_count": cumulative_targets,
            }
        )

    if copy.deepcopy(loader.state_dict()) != original_state:
        raise AssertionError("replay-ledger generation advanced the source loader")

    chained_rows, terminal_chain = chain_replay_rows(rows, domain=chain_domain)
    result = {
        "schema": "exp2d5c_enriched_replay_ledger_v1",
        "hash_definitions": {
            "canonical_json": "utf8/sorted-keys/no-whitespace/no-trailing-newline",
            "logical_global_batch": "sha256(binary per-microbatch combined sha256 digests)",
            "logical_global_batch_stream": "sha256(ascii(sha256(x stream).hex + sha256(y stream).hex))",
            "rolling_chain_domain": str(chain_domain),
        },
        "updates": updates,
        "accumulation": accumulation,
        "inherited_global_update": int(inherited_global_update),
        "final_global_update": int(inherited_global_update) + updates,
        "inherited_cumulative_targets": int(inherited_cumulative_targets),
        "final_cumulative_targets": cumulative_targets,
        "expected_targets_per_update": (
            None
            if expected_targets_per_update is None
            else int(expected_targets_per_update)
        ),
        "initial_loader_state": original_state,
        "terminal_loader_state": copy.deepcopy(replay.state_dict()),
        "rows": chained_rows,
        "replay_chain_sha256": terminal_chain,
    }
    return with_manifest_sha256(result)


def validate_replay_ledger(manifest: Mapping[str, Any]) -> bool:
    """Validate self-hash, row ordering, cursor continuity, and replay chain."""

    try:
        if not verify_manifest_sha256(manifest):
            return False
        rows = manifest.get("rows")
        if not isinstance(rows, list) or len(rows) != manifest.get("updates"):
            return False
        if not verify_replay_chain(
            rows,
            manifest.get("replay_chain_sha256"),
            domain=manifest.get("hash_definitions", {}).get(
                "rolling_chain_domain", REPLAY_CHAIN_DOMAIN
            ),
        ):
            return False
        cumulative = int(manifest["inherited_cumulative_targets"])
        inherited_global = int(manifest["inherited_global_update"])
        previous_end = None
        for index, row in enumerate(rows, start=1):
            if row.get("local_update") != index:
                return False
            if row.get("global_update") != inherited_global + index:
                return False
            if previous_end is not None and row.get("start_cursor") != previous_end:
                return False
            if row.get("microbatch_count") != len(row.get("microbatches", ())):
                return False
            for microbatch_index, microbatch in enumerate(row.get("microbatches", ())):
                if microbatch.get("microbatch_index") != microbatch_index:
                    return False
                if microbatch.get("order") != microbatch_index + 1:
                    return False
            cumulative += int(row["target_count"])
            if row.get("cumulative_target_count") != cumulative:
                return False
            previous_end = row.get("end_cursor")
        return (
            cumulative == manifest.get("final_cumulative_targets")
            and (not rows or rows[-1].get("end_cursor") == _cursor_from_state(
                manifest.get("terminal_loader_state", {})
            ))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _normalise_interval(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("span_start"))
        end = value.get("end", value.get("span_end"))
    else:
        try:
            start, end = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid half-open interval: {value!r}") from exc
    start, end = int(start), int(end)
    if start < 0 or end <= start:
        raise ValueError(f"invalid half-open interval [{start}, {end})")
    return start, end


def _lookup_by_batch(source: Any, batch: Any, position: int) -> Any:
    if source is None:
        return None
    if callable(source):
        return source(batch)
    if isinstance(source, Mapping):
        try:
            return source[batch]
        except (KeyError, TypeError):
            return source[position]
    return source[position]


def _identity_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        preferred = (
            "identity_sha256",
            "token_span_sha256",
            "sequence_sha256",
            "combined_sha256",
        )
        rows = [str(value[key]) for key in preferred if value.get(key) is not None]
        if rows:
            return rows
        return [canonical_json_sha256(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def select_validation_panel_batches(
    candidates: int | Iterable[Any],
    panel_size: int,
    *,
    seed: int = 2_026_083_001,
    forbidden_intervals: Iterable[Any] = (),
    forbidden_identities: Iterable[str] = (),
    batch_intervals: Mapping[Any, Any] | Sequence[Any] | Callable[[Any], Any] | None = None,
    batch_identities: Mapping[Any, Any] | Sequence[Any] | Callable[[Any], Any] | None = None,
    require_selected_interval_disjointness: bool = True,
    domain: str = PANEL_SELECTION_DOMAIN,
) -> dict[str, Any]:
    """Select a seeded panel while excluding historical spans and identities.

    ``candidates`` may be a count (yielding integer batch ids) or explicit ids.
    Intervals are half open.  Selection uses SHA-256 ranking instead of the
    process-global RNG, making results stable across Python versions and calls.
    """

    panel_size = int(panel_size)
    seed = int(seed)
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    if isinstance(candidates, int):
        if candidates < 1:
            raise ValueError("candidate count must be positive")
        candidate_values = list(range(candidates))
    else:
        candidate_values = list(candidates)
    if not candidate_values:
        raise ValueError("no candidate batches")

    forbidden_spans = sorted(_normalise_interval(row) for row in forbidden_intervals)
    forbidden_identity_set = {str(value) for value in forbidden_identities}
    seen_batch_keys: set[str] = set()
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for position, batch in enumerate(candidate_values):
        batch_key = canonical_json_sha256(batch)
        if batch_key in seen_batch_keys:
            raise ValueError(f"duplicate candidate batch at position {position}")
        seen_batch_keys.add(batch_key)
        supplied_interval = _lookup_by_batch(batch_intervals, batch, position)
        if supplied_interval is None:
            if isinstance(batch, Mapping) and (
                "start" in batch or "span_start" in batch
            ):
                supplied_interval = batch
            elif isinstance(batch, int):
                supplied_interval = (batch, batch + 1)
            else:
                supplied_interval = (position, position + 1)
        interval = _normalise_interval(supplied_interval)

        supplied_identities = _lookup_by_batch(batch_identities, batch, position)
        if supplied_identities is None and isinstance(batch, Mapping):
            supplied_identities = batch
        identities = sorted(set(_identity_values(supplied_identities)))
        overlaps = [
            [start, end]
            for start, end in forbidden_spans
            if interval[0] < end and start < interval[1]
        ]
        identity_collisions = sorted(forbidden_identity_set.intersection(identities))
        record = {
            "candidate_position": position,
            "batch": copy.deepcopy(batch),
            "interval": [interval[0], interval[1]],
            "identities": identities,
        }
        if overlaps or identity_collisions:
            record["excluded_by_intervals"] = overlaps
            record["excluded_by_identities"] = identity_collisions
            excluded.append(record)
            continue
        rank_material = {
            "domain": str(domain),
            "seed": seed,
            "candidate": record,
        }
        record["selection_rank_sha256"] = canonical_json_sha256(rank_material)
        eligible.append(record)

    eligible.sort(
        key=lambda row: (row["selection_rank_sha256"], row["candidate_position"])
    )
    selected: list[dict[str, Any]] = []
    for record in eligible:
        start, end = record["interval"]
        if require_selected_interval_disjointness and any(
            start < chosen["interval"][1] and chosen["interval"][0] < end
            for chosen in selected
        ):
            continue
        chosen = copy.deepcopy(record)
        chosen["selection_order"] = len(selected) + 1
        selected.append(chosen)
        if len(selected) == panel_size:
            break
    if len(selected) != panel_size:
        raise ValueError(
            f"only {len(selected)} eligible disjoint batches for panel of {panel_size}"
        )

    result = {
        "schema": "exp2d5c_validation_panel_selection_v1",
        "algorithm": "domain-separated-sha256-ranking",
        "domain": str(domain),
        "seed": seed,
        "interval_semantics": "half-open [start,end)",
        "candidate_count": len(candidate_values),
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "panel_size": panel_size,
        "forbidden_intervals": [[start, end] for start, end in forbidden_spans],
        "forbidden_identities": sorted(forbidden_identity_set),
        "selected_batches": selected,
        "selected_batch_values": [row["batch"] for row in selected],
        "selected_intervals": [row["interval"] for row in selected],
        "selected_identities": [row["identities"] for row in selected],
        "excluded_candidates": excluded,
        "selected_intervals_pairwise_disjoint": True,
        "verified_forbidden_interval_disjointness": True,
        "verified_forbidden_identity_disjointness": True,
    }
    return with_manifest_sha256(result)


__all__ = [
    "DERANGEMENT_DOMAIN",
    "GENESIS_CHAIN_SHA256",
    "MANIFEST_HASH_FIELD",
    "PANEL_SELECTION_DOMAIN",
    "REPLAY_CHAIN_DOMAIN",
    "aggregate_payload_hashes",
    "build_replay_ledger",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "chain_replay_rows",
    "manifest_sha256",
    "microbatch_identity",
    "rolling_replay_chain_sha256",
    "seeded_cyclic_derangement",
    "seeded_path_consistent_cyclic_derangement",
    "select_validation_panel_batches",
    "validate_replay_ledger",
    "verify_manifest_sha256",
    "verify_replay_chain",
    "with_manifest_sha256",
]
