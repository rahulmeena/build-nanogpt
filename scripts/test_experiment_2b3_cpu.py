#!/usr/bin/env python3
"""CPU contract tests for Experiment 2B3 joint optimization."""

import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2b3 as candidate  # noqa: E402
from train_gpt2 import GPT, GPTConfig  # noqa: E402


def test_config():
    config = candidate.load_config()
    assert config["trainable_parameters"] == 50_689
    assert config["writer_parameters"] + config["reader_parameters"] == 50_689
    assert (
        config["world_size"]
        * config["microsteps_per_rank"]
        * config["batch_sequences"]
        * config["sequence_length"]
        == 524_288
    )
    assert config["source_writer_updates"] + config["joint_updates"] == 38
    assert config["forced_restart_after_joint_update"] == 5


def test_joint_freeze_contract():
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=16,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=8,
        memory_writer_init_seed=7,
    )
    model = GPT(config)
    model.freeze_for_joint_writer_reader_training()
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    assert len(trainable) == 11
    assert all(
        name.startswith("transformer.memory_writers.")
        or name.startswith("transformer.topdown_attnres.")
        for name in trainable
    )
    assert sum(value.numel() for value in model.parameters() if value.requires_grad) == 1_057


def test_canonical_hash_order():
    payloads = [
        [hashlib.sha256(f"r{rank}m{micro}".encode()).hexdigest() for micro in range(2)]
        for rank in range(4)
    ]
    expected = hashlib.sha256()
    for micro in range(2):
        for rank in range(4):
            expected.update(bytes.fromhex(payloads[rank][micro]))
    assert candidate.b2a.canonical_batch_hash(payloads) == expected.hexdigest()


def test_reader_optimizer_steps():
    parameters = [
        torch.nn.Parameter(torch.ones(4)),
        torch.nn.Parameter(torch.ones(4)),
        torch.nn.Parameter(torch.ones(())),
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    fresh = candidate.reader_optimizer_report(optimizer, 0)
    assert fresh["state_entries"] == 0
    sum(value.sum() for value in parameters).backward()
    optimizer.step()
    stepped = candidate.reader_optimizer_report(optimizer, 1)
    assert stepped["steps"] == [1, 1, 1]


def test_separate_clipping():
    writer = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    reader = torch.nn.Parameter(torch.tensor([0.3, 0.4]))
    writer.grad = writer.detach().clone()
    reader.grad = reader.detach().clone()
    writer_pre = torch.nn.utils.clip_grad_norm_([writer], 1.0)
    reader_pre = torch.nn.utils.clip_grad_norm_([reader], 1.0)
    assert torch.isclose(writer_pre, torch.tensor(5.0))
    assert torch.isclose(reader_pre, torch.tensor(0.5))
    assert torch.isclose(writer.grad.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(reader.grad.norm(), torch.tensor(0.5), atol=1e-6)


def main():
    test_config()
    test_joint_freeze_contract()
    test_canonical_hash_order()
    test_reader_optimizer_steps()
    test_separate_clipping()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2B3 CPU contract tests: PASS")


if __name__ == "__main__":
    main()
