import copy
import hashlib
import math
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c_artifacts as artifacts  # noqa: E402


class FakeTensor:
    def __init__(self, values, shape):
        self.values = tuple(int(value) for value in values)
        self.shape = tuple(shape)
        self.dtype = "int64"
        assert math.prod(self.shape) == len(self.values)

    def tobytes(self):
        return b"".join(value.to_bytes(8, "little", signed=True) for value in self.values)

    def numel(self):
        return len(self.values)


class FakeExplicitShardLoader:
    def __init__(self, microbatches, *, position=0):
        self.microbatches = tuple(microbatches)
        self.current_position = int(position)
        self.current_shard = 0
        self.shards = ("/dataset/train_000001.npy",)

    def state_dict(self):
        return {
            "shards": list(self.shards),
            "batch_size": 1,
            "sequence_length": 2,
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def clone(self):
        return FakeExplicitShardLoader(
            self.microbatches, position=self.current_position
        )

    def next_batch(self):
        row = self.microbatches[self.current_position % len(self.microbatches)]
        self.current_position += 1
        return row


def fake_microbatches(count=16):
    result = []
    for index in range(count):
        x = FakeTensor((index, index + 1), (1, 2))
        y = FakeTensor((index + 1, index + 2), (1, 2))
        result.append((x, y))
    return result


def test_canonical_json_and_manifest_hash_are_stable_and_strict():
    left = {"z": [3, 2, 1], "unicode": "λ", "a": {"b": True}}
    right = {"a": {"b": True}, "unicode": "λ", "z": [3, 2, 1]}
    assert artifacts.canonical_json_bytes(left) == artifacts.canonical_json_bytes(right)
    assert artifacts.canonical_json_sha256(left) == artifacts.canonical_json_sha256(right)
    assert not artifacts.canonical_json_bytes(left).endswith(b"\n")
    with pytest.raises(ValueError):
        artifacts.canonical_json_bytes({"bad": float("nan")})

    frozen = artifacts.with_manifest_sha256(left)
    assert artifacts.verify_manifest_sha256(frozen)
    frozen["z"].append(0)
    assert not artifacts.verify_manifest_sha256(frozen)


def test_rolling_chain_detects_reordering_and_tampering():
    rows, terminal = artifacts.chain_replay_rows(
        [{"local_update": 1, "value": "a"}, {"local_update": 2, "value": "b"}]
    )
    assert artifacts.verify_replay_chain(rows, terminal)
    assert rows[0]["previous_chain_sha256"] == artifacts.GENESIS_CHAIN_SHA256
    assert rows[1]["previous_chain_sha256"] == rows[0]["chain_sha256"]

    tampered = copy.deepcopy(rows)
    tampered[0]["value"] = "changed"
    assert not artifacts.verify_replay_chain(tampered, terminal)
    assert not artifacts.verify_replay_chain(list(reversed(rows)), terminal)


def test_seeded_derangement_is_one_shared_cycle_without_fixed_points():
    first = artifacts.seeded_path_consistent_cyclic_derangement(
        23, 2_026_083_002
    )
    second = artifacts.seeded_cyclic_derangement(23, 2_026_083_002)
    changed = artifacts.seeded_cyclic_derangement(23, 2_026_083_003)
    donors = first["donor_permutation"]

    assert first == second
    assert donors != changed["donor_permutation"]
    assert sorted(donors) == list(range(23))
    assert all(recipient != donor for recipient, donor in enumerate(donors))
    assert first["path_permutations"]["B3"] == donors
    assert first["path_permutations"]["B5"] == donors
    assert first["donor_permutation_sha256"] == artifacts.canonical_json_sha256(donors)
    assert artifacts.verify_manifest_sha256(first)

    visited = []
    row = 0
    for _ in range(23):
        visited.append(row)
        row = donors[row]
    assert row == 0
    assert len(set(visited)) == 23
    with pytest.raises(ValueError):
        artifacts.seeded_cyclic_derangement(1, 2_026_083_002)


def test_replay_ledger_matches_project_hashes_and_preserves_source_loader():
    loader = FakeExplicitShardLoader(fake_microbatches())
    before = copy.deepcopy(loader.state_dict())
    ledger = artifacts.build_replay_ledger(
        loader,
        accumulation=2,
        updates=13,
        inherited_global_update=1_908,
        inherited_cumulative_targets=1_000,
        expected_targets_per_update=4,
    )

    assert loader.state_dict() == before
    assert ledger["updates"] == 13
    assert ledger["final_global_update"] == 1_921
    assert ledger["final_cumulative_targets"] == 1_052
    assert artifacts.validate_replay_ledger(ledger)
    assert artifacts.verify_manifest_sha256(ledger)

    first = ledger["rows"][0]
    assert first["local_update"] == 1
    assert first["global_update"] == 1_909
    assert first["start_cursor"]["current_position"] == 0
    assert first["end_cursor"]["current_position"] == 2
    assert [row["order"] for row in first["microbatches"]] == [1, 2]
    assert first["target_count"] == 4
    assert first["pass_count"] == 2
    assert ledger["rows"][11]["global_update"] == 1_920
    assert ledger["rows"][11]["pass_count"] == 3

    first_payloads = [
        artifacts.microbatch_identity(x, y)["combined_sha256"]
        for x, y in fake_microbatches()[:2]
    ]
    assert first["logical_global_batch_sha256"] == artifacts.aggregate_payload_hashes(
        first_payloads
    )
    x_digest, y_digest = hashlib.sha256(), hashlib.sha256()
    for x, y in fake_microbatches()[:2]:
        x_digest.update(x.tobytes())
        y_digest.update(y.tobytes())
    expected_stream = hashlib.sha256(
        (x_digest.hexdigest() + y_digest.hexdigest()).encode("ascii")
    ).hexdigest()
    assert first["logical_global_batch_stream_sha256"] == expected_stream


def test_replay_ledger_can_build_all_191_rows_and_fails_closed_on_anchors():
    loader = FakeExplicitShardLoader(fake_microbatches(32))
    ledger = artifacts.build_replay_ledger(
        loader,
        accumulation=1,
        updates=191,
        expected_targets_per_update=2,
    )
    assert len(ledger["rows"]) == 191
    assert ledger["rows"][-1]["local_update"] == 191
    assert artifacts.validate_replay_ledger(ledger)

    wrong = ["0" * 64]
    with pytest.raises(ValueError, match="logical global-batch hash mismatch"):
        artifacts.build_replay_ledger(
            loader,
            accumulation=1,
            updates=1,
            expected_targets_per_update=2,
            expected_batch_hashes=wrong,
        )


def test_validation_panel_selection_excludes_intervals_and_identities():
    identities = {index: f"sequence-{index}" for index in range(40)}
    kwargs = {
        "seed": 2_026_083_001,
        "forbidden_intervals": [(3, 8), (20, 23)],
        "forbidden_identities": {"sequence-12", "sequence-31"},
        "batch_identities": identities,
    }
    first = artifacts.select_validation_panel_batches(40, 8, **kwargs)
    second = artifacts.select_validation_panel_batches(40, 8, **kwargs)
    changed = artifacts.select_validation_panel_batches(
        40, 8, **{**kwargs, "seed": 2_026_083_002}
    )

    assert first == second
    assert artifacts.verify_manifest_sha256(first)
    selected = first["selected_batch_values"]
    assert len(selected) == len(set(selected)) == 8
    assert not set(selected).intersection(range(3, 8))
    assert not set(selected).intersection(range(20, 23))
    assert 12 not in selected and 31 not in selected
    assert selected != changed["selected_batch_values"]
    assert first["verified_forbidden_interval_disjointness"]
    assert first["verified_forbidden_identity_disjointness"]


def test_validation_panel_selection_handles_exact_span_records_and_shortage():
    candidates = [
        {
            "batch_index": index,
            "span_start": index * 10,
            "span_end": index * 10 + 10,
            "token_span_sha256": f"token-{index}",
        }
        for index in range(10)
    ]
    panel = artifacts.select_validation_panel_batches(
        candidates,
        4,
        forbidden_intervals=[{"span_start": 20, "span_end": 40}],
        forbidden_identities={"token-7"},
    )
    selected_indices = {
        row["batch"]["batch_index"] for row in panel["selected_batches"]
    }
    assert not selected_indices.intersection({2, 3, 7})

    with pytest.raises(ValueError, match="eligible disjoint batches"):
        artifacts.select_validation_panel_batches(
            5,
            4,
            forbidden_intervals=[(0, 3)],
        )
