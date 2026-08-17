#!/usr/bin/env python3
"""CPU contract tests for Experiment 2B5 decomposition semantics."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2b5 as candidate  # noqa: E402


def test_config():
    config = candidate.load_config()
    assert config["training"] == "forbidden"
    assert config["hellaswag"] == "forbidden"
    assert config["alphas"] == list(candidate.ALPHAS)
    assert config["checkpoint_labels"] == list(candidate.CHECKPOINT_LABELS)


def test_permutations_reuse_2b4():
    coherent = candidate.coherent_permutation(64, torch.device("cpu"))
    independent = candidate.independent_source_permutations(64, torch.device("cpu"))
    rows = torch.arange(64)
    assert torch.equal(coherent, rows.roll(1))
    assert not torch.any(coherent == rows)
    assert len(independent) == 4
    assert all(torch.equal(value, rows.roll(index + 1)) for index, value in enumerate(independent))


def test_decomposition_controls_and_identities():
    torch.manual_seed(7)
    memory = torch.randn(4, 8, 1, 16, dtype=torch.bfloat16)
    mean = torch.randn(4, 16, dtype=torch.float32)
    zero = torch.zeros_like(memory)

    assert torch.equal(candidate.controlled_memory("mu", zero, mean, 0), zero)
    mu = candidate.controlled_memory("mu", memory, mean, 1)
    expected_mu = mean[:, None, None].expand_as(memory).to(memory.dtype)
    assert torch.equal(mu, expected_mu)

    real_1x, identity = candidate.controlled_memory(
        "alpha_real_1", memory, mean, 1, return_identity=True
    )
    assert torch.equal(real_1x, memory)
    assert identity["max_absolute_difference"] == 0.0

    shuffled_1x, shuffled_identity = candidate.controlled_memory(
        "alpha_shuffle_1", memory, mean, 1, return_identity=True
    )
    direct = memory[:, candidate.coherent_permutation(8, memory.device)]
    assert torch.equal(shuffled_1x, direct)
    assert shuffled_identity["max_absolute_difference"] == 0.0

    residual = candidate.controlled_memory("residual", memory, mean, 1)
    expected_residual = (
        memory.float() - mean[:, None, None]
    ).to(memory.dtype)
    assert torch.equal(residual, expected_residual)

    independent_1x, independent_identity = candidate.controlled_memory(
        "independent_shuffle", memory, mean, 1, return_identity=True
    )
    permutations = candidate.independent_source_permutations(8, memory.device)
    direct_independent = torch.stack(
        [memory[index, permutation] for index, permutation in enumerate(permutations)]
    )
    assert torch.equal(independent_1x, direct_independent)
    assert independent_identity["max_absolute_difference"] == 0.0


def test_row_coupling_is_exact():
    memory = torch.zeros(4, 8, 1, 4, dtype=torch.bfloat16)
    mean = torch.zeros(4, 4)
    memory[:, 0] = 1
    coherent = candidate.controlled_memory("alpha_shuffle_1", memory, mean, 1)
    changed = torch.nonzero(coherent.abs().sum(dim=(0, 2, 3))).flatten().tolist()
    assert changed == [1]

    independent = candidate.controlled_memory("independent_shuffle", memory, mean, 1)
    receivers = [
        torch.nonzero(independent[source].abs().sum(dim=(1, 2))).flatten().tolist()
        for source in range(4)
    ]
    assert receivers == [[1], [2], [3], [4]]


def main():
    test_config()
    test_permutations_reuse_2b4()
    test_decomposition_controls_and_identities()
    test_row_coupling_is_exact()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2B5 CPU contract tests: PASS")


if __name__ == "__main__":
    main()
