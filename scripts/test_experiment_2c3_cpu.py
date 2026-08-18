#!/usr/bin/env python3
"""CPU contracts for the evaluation-only 2C3 per-reader shuffle selector."""

import copy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from train_gpt2 import GPT, GPTConfig  # noqa: E402


def model():
    torch.manual_seed(1234)
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            residual_mode="full_attnres",
            enable_topdown_feedback=True,
            topdown_feedback_destinations=(0, 1),
        )
    ).eval()


def expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), (fragment, str(error))
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def main():
    network = model()
    state_before = copy.deepcopy(network.state_dict())
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    x = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    memory = torch.randn(4, 4, 4, 8)
    permutation = torch.tensor([1, 2, 3, 0])
    captured = {0: [], 1: []}

    def capture(destination):
        def hook(_module, inputs):
            captured[destination].append(torch.stack(tuple(inputs[0]), dim=0))

        return hook

    handles = [
        network.transformer.topdown_attnres_by_destination[str(destination)].register_forward_pre_hook(
            capture(destination)
        )
        for destination in (0, 1)
    ]
    with torch.no_grad():
        reference_a, _ = network(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
        )
        reference_b, _ = network(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
        )
        network(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_permutation_by_destination={1: permutation},
        )
    for handle in handles:
        handle.remove()
    assert torch.equal(reference_a, reference_b)
    assert torch.equal(captured[0][-1], memory)
    assert torch.equal(captured[1][-1], memory[:, permutation])
    assert sum(parameter.numel() for parameter in network.parameters()) == parameter_count
    assert set(network.state_dict()) == set(state_before)
    assert all(
        torch.equal(value, state_before[name])
        for name, value in network.state_dict().items()
    )
    expect_value_error(
        lambda: network(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_permutation_by_destination={2: permutation},
        ),
        "configured readers",
    )
    expect_value_error(
        lambda: network(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_permutation_by_destination={1: torch.arange(4)},
        ),
        "fixed-point-free",
    )
    expect_value_error(
        lambda: network(
            x,
            mode="masked_cumulative_shuffled_feedback",
            feedback_sources=memory,
            feedback_permutation=permutation,
            feedback_permutation_by_destination={1: permutation},
        ),
        "mutually exclusive",
    )
    print("EXPERIMENT_2C3_CPU_CONTRACTS_PASS")


if __name__ == "__main__":
    main()
