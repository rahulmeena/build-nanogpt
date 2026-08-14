#!/usr/bin/env python3
"""Correctness, causality, and state tests for Experiment 2B0."""

import io
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


torch.set_num_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]


def definition_prefix(source):
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    if marker not in source:
        raise RuntimeError("could not find training launch marker")
    return source.split(marker)[0]


def load_symbols(source, filename, name):
    namespace = {"__name__": name, "__file__": filename, "master_process": True}
    sys.path.insert(0, str(REPO_ROOT))
    exec(compile(definition_prefix(source), filename, "exec"), namespace)
    return namespace


CURRENT = load_symbols(
    (REPO_ROOT / "train_gpt2.py").read_text(),
    str(REPO_ROOT / "train_gpt2.py"),
    "experiment_2b0_under_test",
)


def tiny_model(block_size=8):
    config = CURRENT["GPTConfig"](
        block_size=block_size,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
    )
    model = CURRENT["GPT"](config).eval()
    with torch.no_grad():
        model.transformer.topdown_attnres.query.copy_(
            torch.linspace(-0.2, 0.2, config.n_embd)
        )
        model.transformer.topdown_attnres.gate.fill_(0.25)
    return model


def incremental(model, tokens, mode, teacher_feedback=None, reset_positions=()):
    state = model.init_recurrent_state(tokens.size(0), mode)
    logits = []
    memories = []
    for position in range(tokens.size(1)):
        kwargs = {"reset_feedback": position in reset_positions}
        if teacher_feedback is not None:
            kwargs["feedback_sources"] = teacher_feedback[:, :, position : position + 1]
        step_logits, state = model.forward_step(
            tokens[:, position : position + 1], state, **kwargs
        )
        logits.append(step_logits)
        memories.append(state.feedback_memory.clone())
    return torch.cat(logits, dim=1), state, memories


class IncrementalEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260814)
        self.model = tiny_model()
        self.tokens = torch.randint(0, self.model.config.vocab_size, (2, 8))
        self.targets = torch.randint(0, self.model.config.vocab_size, (2, 8))

    def assert_logits_and_loss_close(self, parallel, incremental_logits):
        torch.testing.assert_close(incremental_logits, parallel, rtol=2e-5, atol=2e-6)
        parallel_loss = F.cross_entropy(
            parallel.reshape(-1, parallel.size(-1)), self.targets.reshape(-1)
        )
        incremental_loss = F.cross_entropy(
            incremental_logits.reshape(-1, incremental_logits.size(-1)),
            self.targets.reshape(-1),
        )
        torch.testing.assert_close(incremental_loss, parallel_loss, rtol=2e-6, atol=2e-7)

    def test_full_context_parallel_incremental_equivalence(self):
        with torch.no_grad():
            parallel, _ = self.model(self.tokens, mode="full_context")
            stepwise, state, _ = incremental(self.model, self.tokens, "full_context")
        self.assert_logits_and_loss_close(parallel, stepwise)
        self.assertEqual(state.position, 8)
        self.assertTrue(all(cache.length == 8 for cache in state.kv_caches))

    def test_masked_l1_parallel_incremental_equivalence(self):
        with torch.no_grad():
            parallel, _ = self.model(self.tokens, mode="masked_l1_no_feedback")
            stepwise, state, _ = incremental(
                self.model, self.tokens, "masked_l1_no_feedback"
            )
        self.assert_logits_and_loss_close(parallel, stepwise)
        self.assertIsNone(state.kv_caches[0])
        self.assertTrue(all(cache.length == 8 for cache in state.kv_caches[1:]))

    def test_teacher_parallel_incremental_equivalence(self):
        with torch.no_grad():
            sources = self.model.capture_residual_sources(self.tokens)
            feedback = CURRENT["shift_teacher_sources"](sources)
            parallel, _ = self.model(
                self.tokens,
                mode="masked_l1_topdown_teacher",
                feedback_sources=feedback,
            )
            stepwise, _, _ = incremental(
                self.model,
                self.tokens,
                "masked_l1_topdown_teacher",
                teacher_feedback=feedback,
            )
        self.assert_logits_and_loss_close(parallel, stepwise)


class RecurrentCausalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.model = tiny_model()

    def test_future_suffix_invariance_includes_memory_and_caches(self):
        first = torch.randint(0, self.model.config.vocab_size, (2, 8))
        second = first.clone()
        second[:, 5:] = (second[:, 5:] + 7) % self.model.config.vocab_size
        with torch.no_grad():
            first_logits, first_state, first_memories = incremental(
                self.model, first[:, :5], "masked_l1_topdown_self"
            )
            second_logits, second_state, second_memories = incremental(
                self.model, second[:, :5], "masked_l1_topdown_self"
            )
        self.assertTrue(torch.equal(first_logits, second_logits))
        for left, right in zip(first_memories, second_memories):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(first_state.kv_caches, second_state.kv_caches):
            if left is not None:
                self.assertTrue(torch.equal(left.prefix()[0], right.prefix()[0]))
                self.assertTrue(torch.equal(left.prefix()[1], right.prefix()[1]))

    def test_batch_rows_are_isolated(self):
        first = torch.randint(0, self.model.config.vocab_size, (2, 8))
        second = first.clone()
        second[1] = (second[1] + 11) % self.model.config.vocab_size
        with torch.no_grad():
            first_logits, first_state, _ = incremental(
                self.model, first, "masked_l1_topdown_self"
            )
            second_logits, second_state, _ = incremental(
                self.model, second, "masked_l1_topdown_self"
            )
        self.assertTrue(torch.equal(first_logits[0], second_logits[0]))
        self.assertTrue(
            torch.equal(first_state.feedback_memory[:, 0], second_state.feedback_memory[:, 0])
        )
        for left, right in zip(first_state.kv_caches[1:], second_state.kv_caches[1:]):
            self.assertTrue(torch.equal(left.prefix()[0][0], right.prefix()[0][0]))
            self.assertTrue(torch.equal(left.prefix()[1][0], right.prefix()[1][0]))

    def test_fresh_sequence_reset_has_no_history(self):
        prefix = torch.randint(0, self.model.config.vocab_size, (2, 8))
        fresh = torch.randint(0, self.model.config.vocab_size, (2, 1))
        with torch.no_grad():
            incremental(self.model, prefix, "masked_l1_topdown_self")
            first_logits, first_state, _ = incremental(
                self.model, fresh, "masked_l1_topdown_self"
            )
            second_logits, second_state, _ = incremental(
                self.model, fresh, "masked_l1_topdown_self"
            )
        self.assertTrue(torch.equal(first_logits, second_logits))
        self.assertTrue(torch.equal(first_state.feedback_memory, second_state.feedback_memory))
        initial = self.model.init_recurrent_state(2, "masked_l1_topdown_self")
        self.assertEqual(initial.position, 0)
        self.assertEqual(initial.feedback_memory.count_nonzero().item(), 0)
        self.assertIsNone(initial.kv_caches[0])
        self.assertTrue(all(cache.length == 0 for cache in initial.kv_caches[1:]))

    def test_memory_only_reset_preserves_kv_and_removes_feedback(self):
        tokens = torch.randint(0, self.model.config.vocab_size, (2, 4))
        with torch.no_grad():
            _, state, _ = incremental(
                self.model, tokens[:, :3], "masked_l1_topdown_self"
            )
            reset = self.model.reset_recurrent_memory(state)
        self.assertEqual(reset.position, state.position)
        self.assertEqual(reset.feedback_memory.count_nonzero().item(), 0)
        for before, after in zip(state.kv_caches, reset.kv_caches):
            self.assertIs(before, after)

    def test_per_example_reset_mask_is_row_local(self):
        tokens = torch.randint(0, self.model.config.vocab_size, (2, 4))
        state = self.model.init_recurrent_state(2, "masked_l1_topdown_self")
        with torch.no_grad():
            for position in range(3):
                _, state = self.model.forward_step(tokens[:, position], state)
            payload = state.state_dict()
            mixed = self.model.load_recurrent_state(payload)
            all_reset = self.model.load_recurrent_state(payload)
            no_reset = self.model.load_recurrent_state(payload)
            mixed_logits, _ = self.model.forward_step(
                tokens[:, 3], mixed, reset_feedback=torch.tensor([True, False])
            )
            reset_logits, _ = self.model.forward_step(
                tokens[:, 3], all_reset, reset_feedback=torch.tensor([True, True])
            )
            normal_logits, _ = self.model.forward_step(
                tokens[:, 3], no_reset, reset_feedback=torch.tensor([False, False])
            )
        self.assertTrue(torch.equal(mixed_logits[0], reset_logits[0]))
        self.assertTrue(torch.equal(mixed_logits[1], normal_logits[1]))


class RecurrentSerializationTests(unittest.TestCase):
    def test_compact_state_roundtrip_resume_is_exact(self):
        torch.manual_seed(909)
        model = tiny_model()
        tokens = torch.randint(0, model.config.vocab_size, (2, 8))
        state = model.init_recurrent_state(2, "masked_l1_topdown_self")
        with torch.no_grad():
            for position in range(3):
                _, state = model.forward_step(tokens[:, position], state)
            buffer = io.BytesIO()
            torch.save(state.state_dict(), buffer)
            buffer.seek(0)
            restored = model.load_recurrent_state(
                torch.load(buffer, weights_only=False)
            )
            direct_logits = []
            restored_logits = []
            for position in range(3, 8):
                direct, state = model.forward_step(tokens[:, position], state)
                resumed, restored = model.forward_step(tokens[:, position], restored)
                direct_logits.append(direct)
                restored_logits.append(resumed)
        self.assertTrue(torch.equal(torch.cat(direct_logits, 1), torch.cat(restored_logits, 1)))
        self.assertTrue(torch.equal(state.feedback_memory, restored.feedback_memory))
        for direct, resumed in zip(state.kv_caches[1:], restored.kv_caches[1:]):
            self.assertTrue(torch.equal(direct.prefix()[0], resumed.prefix()[0]))
            self.assertTrue(torch.equal(direct.prefix()[1], resumed.prefix()[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
