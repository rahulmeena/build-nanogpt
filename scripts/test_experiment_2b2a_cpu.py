#!/usr/bin/env python3
"""CPU-only contract tests for Experiment 2B2A distributed arithmetic."""

import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2b2a as candidate  # noqa: E402


def test_config():
    config = candidate.load_config()
    assert config["global_targets_per_update"] == 524_288
    assert config["world_size"] * config["rank_targets_per_update"] == 524_288
    assert (
        config["world_size"]
        * config["microsteps_per_rank"]
        * config["batch_sequences"]
        * config["sequence_length"]
        == 524_288
    )


def test_canonical_hash_order():
    payloads = [[hashlib.sha256(f"r{rank}m{micro}".encode()).hexdigest()
                 for micro in range(2)] for rank in range(4)]
    expected = hashlib.sha256()
    for micro in range(2):
        for rank in range(4):
            expected.update(bytes.fromhex(payloads[rank][micro]))
    assert candidate.canonical_batch_hash(payloads) == expected.hexdigest()


def test_global_mean_gradient_scaling():
    # Each of eight equal-sized slices contributes loss_sum / global_targets.
    local = []
    for rank in range(4):
        parameter = torch.tensor([1.0, -2.0], requires_grad=True)
        for micro in range(2):
            value = float(2 * rank + micro + 1)
            (parameter.sum() * value / 8.0).backward()
        local.append(parameter.grad)
    distributed_sum = torch.stack(local).sum(0)
    reference_parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    for value in range(1, 9):
        (reference_parameter.sum() * value / 8.0).backward()
    assert torch.equal(distributed_sum, reference_parameter.grad)


def test_comparison_metrics():
    reference = torch.linspace(-1.0, 1.0, 128)
    same = candidate.comparison(reference, reference.clone())
    assert same["cosine_similarity"] >= 0.999999999
    assert same["relative_l2_difference"] == 0.0
    perturbed = candidate.comparison(reference, reference + 1e-7)
    assert perturbed["relative_l2_difference"] < 1e-4


def test_single_tensor_digest_wrapper():
    value = torch.arange(8, dtype=torch.float32)
    digest = candidate.tensor_digest([("gradient", value)])
    assert isinstance(digest, str) and len(digest) == 64


def main():
    test_config()
    test_canonical_hash_order()
    test_global_mean_gradient_scaling()
    test_comparison_metrics()
    test_single_tensor_digest_wrapper()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2B2A CPU contract tests: PASS")


if __name__ == "__main__":
    main()
