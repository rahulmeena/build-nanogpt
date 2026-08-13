#!/usr/bin/env python3
"""Correctness and causality tests for Experiment 2A0."""

import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_1_COMMIT = "abecd3e91e89e1259f7198d72d15664943ad48bf"


def definition_prefix(source):
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    if marker not in source:
        raise RuntimeError("could not find training launch marker")
    return source.split(marker)[0]


def load_symbols(source, filename, name):
    namespace = {
        "__name__": name,
        "__file__": filename,
        "master_process": True,
    }
    sys.path.insert(0, str(REPO_ROOT))
    exec(compile(definition_prefix(source), filename, "exec"), namespace)
    namespace["master_process"] = True
    return namespace


CURRENT = load_symbols(
    (REPO_ROOT / "train_gpt2.py").read_text(),
    str(REPO_ROOT / "train_gpt2.py"),
    "experiment_2a0_under_test",
)


def tiny_config(enable_topdown=True, n_layer=12):
    return CURRENT["GPTConfig"](
        block_size=8,
        vocab_size=32,
        n_layer=n_layer,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=enable_topdown,
    )


class TopDownRouterTests(unittest.TestCase):
    def test_initialization_and_depth_only_softmax(self):
        router = CURRENT["TopDownAttnRes"](8, (16, 17, 20, 24))
        values = [torch.randn(2, 3, 8) for _ in range(4)]
        output, weights = router(values, return_weights=True)
        expected_weights = torch.full_like(weights, 0.25)
        expected_output = sum(values) / 4
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
        self.assertEqual(router.query.count_nonzero().item(), 0)
        self.assertEqual(router.gate.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(router.norm.weight, torch.ones_like(router.norm.weight)))
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(2, 3), rtol=0, atol=0)

    def test_source_mask_renormalizes_remaining_sources(self):
        router = CURRENT["TopDownAttnRes"](8, (16, 17, 20, 24))
        router.masked_source = 1
        values = [torch.randn(2, 3, 8) for _ in range(4)]
        _, weights = router(values, return_weights=True)
        self.assertEqual(weights[1].count_nonzero().item(), 0)
        expected = torch.full_like(weights[0], 1 / 3)
        for index in (0, 2, 3):
            torch.testing.assert_close(weights[index], expected, rtol=1e-6, atol=1e-6)

    def test_shift_is_detached_zero_filled_and_has_no_wraparound(self):
        sources = torch.arange(4 * 2 * 5 * 3, dtype=torch.float32).view(4, 2, 5, 3)
        sources.requires_grad_(True)
        shifted = CURRENT["shift_teacher_sources"](sources)
        self.assertFalse(shifted.requires_grad)
        self.assertEqual(shifted[:, :, 0].count_nonzero().item(), 0)
        torch.testing.assert_close(shifted[:, :, 1:], sources.detach()[:, :, :-1], rtol=0, atol=0)

    def test_fixed_derangement(self):
        with self.assertRaises(ValueError):
            CURRENT["fixed_derangement"](1)
        permutation = CURRENT["fixed_derangement"](7)
        self.assertTrue(torch.equal(torch.sort(permutation).values, torch.arange(7)))
        self.assertFalse(torch.any(permutation == torch.arange(7)))

    def test_nonuniform_routing_is_independent_at_each_token(self):
        router = CURRENT["TopDownAttnRes"](8, (16, 17, 20, 24))
        with torch.no_grad():
            router.query.copy_(torch.arange(1, 9, dtype=torch.float32) / 8)
        values = [torch.randn(2, 5, 8) for _ in range(4)]
        _, original = router(values, return_weights=True)
        changed = [value.clone() for value in values]
        changed[2][:, 4] += 100
        _, perturbed = router(changed, return_weights=True)
        torch.testing.assert_close(perturbed[:, :, :4], original[:, :, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(perturbed[:, :, 4], original[:, :, 4]))


class ArchitectureModeTests(unittest.TestCase):
    def test_self_only_attention_is_exact_current_value_projection(self):
        config = tiny_config(n_layer=12)
        attention = CURRENT["CausalSelfAttention"](config).eval()
        x = torch.randn(2, 5, config.n_embd)
        with torch.no_grad():
            qkv = attention.c_attn(x)
            _, _, value = qkv.split(config.n_embd, dim=2)
            expected = attention.c_proj(value)
            actual = attention(x, self_only=True)
        # The sliced V-only GEMM and the V slice of the fused QKV GEMM are
        # mathematically identical but may select different CPU kernels.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_full_context_is_exact_experiment1_regression(self):
        reference_source = subprocess.check_output(
            ["git", "show", f"{EXPERIMENT_1_COMMIT}:train_gpt2.py"],
            cwd=REPO_ROOT,
            text=True,
        )
        reference = load_symbols(
            reference_source,
            f"{EXPERIMENT_1_COMMIT}:train_gpt2.py",
            "experiment_1_frozen_reference",
        )
        kwargs = dict(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            residual_mode="full_attnres",
        )
        torch.manual_seed(2026)
        expected_model = reference["GPT"](reference["GPTConfig"](**kwargs)).eval()
        actual_model = CURRENT["GPT"](
            CURRENT["GPTConfig"](**kwargs, enable_topdown_feedback=True)
        ).eval()
        actual_model.load_experiment1_full_attnres_state(expected_model.state_dict())
        idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [7, 6, 5, 4, 3, 2, 1, 0]])
        with torch.no_grad():
            expected_logits, expected_loss = expected_model(idx, idx)
            actual_logits, actual_loss = actual_model(idx, idx, mode="full_context")
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)

    def test_zero_gate_is_exact_masked_no_feedback_equivalence(self):
        model = CURRENT["GPT"](tiny_config()).eval()
        idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        feedback = torch.randn(4, 2, model.config.block_size, model.config.n_embd)
        feedback[:, :, 0] = 0
        with torch.no_grad():
            no_feedback_logits, no_feedback_loss = model(
                idx, idx, mode="masked_l1_no_feedback"
            )
            feedback_logits, feedback_loss = model(
                idx,
                idx,
                mode="masked_l1_topdown_teacher",
                feedback_sources=feedback,
            )
        torch.testing.assert_close(feedback_logits, no_feedback_logits, rtol=0, atol=0)
        torch.testing.assert_close(feedback_loss, no_feedback_loss, rtol=0, atol=0)

    def test_only_block1_receives_self_only_flag(self):
        model = CURRENT["GPT"](tiny_config()).eval()
        calls = []
        for block_index, block in enumerate(model.transformer.h):
            original = block.attn.forward

            def wrapper(module_self, x, self_only=False, *, _index=block_index, _original=original):
                calls.append((_index, self_only))
                return _original(x, self_only=self_only)

            block.attn.forward = types.MethodType(wrapper, block.attn)
        idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        with torch.no_grad():
            model(idx, mode="masked_l1_no_feedback")
        self.assertEqual(calls, [(0, True)] + [(index, False) for index in range(1, 12)])

    def test_shuffled_mode_uses_fixed_point_free_sequence_memory(self):
        model = CURRENT["GPT"](tiny_config()).eval()
        with torch.no_grad():
            model.transformer.topdown_attnres.gate.fill_(0.5)
            model.transformer.topdown_attnres.query.fill_(0.05)
        idx = torch.randint(0, model.config.vocab_size, (3, model.config.block_size))
        feedback = torch.randn(4, 3, model.config.block_size, model.config.n_embd)
        feedback[:, :, 0] = 0
        permutation = CURRENT["fixed_derangement"](3)
        with torch.no_grad():
            actual, _ = model(
                idx,
                mode="masked_l1_shuffled_feedback",
                feedback_sources=feedback,
                feedback_permutation=permutation,
            )
            expected, _ = model(
                idx,
                mode="masked_l1_topdown_teacher",
                feedback_sources=feedback[:, permutation],
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class CausalityAndGradientTests(unittest.TestCase):
    def test_future_tokens_cannot_change_memory_at_current_position(self):
        teacher = CURRENT["GPT"](tiny_config(enable_topdown=False)).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]) % 32
        second = first.clone()
        t = 4
        second[:, t:] = torch.tensor([[9, 10, 11, 12], [12, 11, 10, 9]])
        with torch.no_grad():
            _, _, first_sources = teacher(
                first,
                mode="full_context",
                return_source_depths=CURRENT["EXPERIMENT_2A0_SOURCE_DEPTHS"],
            )
            _, _, second_sources = teacher(
                second,
                mode="full_context",
                return_source_depths=CURRENT["EXPERIMENT_2A0_SOURCE_DEPTHS"],
            )
        first_memory = CURRENT["shift_teacher_sources"](first_sources)
        second_memory = CURRENT["shift_teacher_sources"](second_sources)
        self.assertTrue(torch.equal(first_memory[:, :, t], second_memory[:, :, t]))
        self.assertEqual(first_memory[:, :, 0].count_nonzero().item(), 0)

    def test_requested_source_depths_are_raw_full_attnres_values(self):
        model = CURRENT["GPT"](tiny_config(enable_topdown=False)).eval()
        idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        with torch.no_grad():
            pos = torch.arange(idx.size(1))
            x = model.transformer.wte(idx) + model.transformer.wpe(pos)
            values = [x]
            destination = 0
            for block in model.transformer.h:
                h = model.transformer.attnres[destination](values)
                values.append(block.attn(block.ln_1(h)))
                destination += 1
                h = model.transformer.attnres[destination](values)
                values.append(block.mlp(block.ln_2(h)))
                destination += 1
            _, _, captured = model(
                idx,
                return_source_depths=CURRENT["EXPERIMENT_2A0_SOURCE_DEPTHS"],
            )
            early_exit_captured = model.capture_residual_sources(idx)
        expected = torch.stack(
            [values[depth] for depth in CURRENT["EXPERIMENT_2A0_SOURCE_DEPTHS"]]
        )
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(early_exit_captured, expected, rtol=0, atol=0)

    def test_freeze_boundary_and_new_parameter_gradients(self):
        model = CURRENT["GPT"](tiny_config())
        model.freeze_for_topdown_training()
        trainable = {
            name: parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {
                "transformer.topdown_attnres.query": 8,
                "transformer.topdown_attnres.norm.weight": 8,
                "transformer.topdown_attnres.gate": 1,
            },
        )
        with torch.no_grad():
            model.transformer.topdown_attnres.query.fill_(0.05)
            model.transformer.topdown_attnres.gate.fill_(0.1)
        idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        feedback = torch.randn(4, 2, model.config.block_size, model.config.n_embd)
        feedback[:, :, 0] = 0
        _, loss = model(
            idx,
            idx,
            mode="masked_l1_topdown_teacher",
            feedback_sources=feedback,
        )
        loss.backward()
        router = model.transformer.topdown_attnres
        for parameter in (router.gate, router.query, router.norm.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.count_nonzero().item(), 0)
        for name, parameter in model.named_parameters():
            if not name.startswith("transformer.topdown_attnres."):
                self.assertIsNone(parameter.grad, name)

    def test_exact_initialization_gradient_stages(self):
        model = CURRENT["GPT"](tiny_config())
        model.freeze_for_topdown_training()
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.05,
        )
        idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        feedback = torch.randn(4, 2, model.config.block_size, model.config.n_embd)
        feedback[:, :, 0] = 0
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not name.startswith("transformer.topdown_attnres.")
        }
        stages = []
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(
                idx,
                idx,
                mode="masked_l1_topdown_teacher",
                feedback_sources=feedback,
            )
            loss.backward()
            router = model.transformer.topdown_attnres
            stages.append({
                "gate": router.gate.grad is not None and bool(router.gate.grad.count_nonzero()),
                "query": router.query.grad is not None and bool(router.query.grad.count_nonzero()),
                "rmsnorm": (
                    router.norm.weight.grad is not None
                    and bool(router.norm.weight.grad.count_nonzero())
                ),
            })
            optimizer.step()
        self.assertEqual(stages[0], {"gate": True, "query": False, "rmsnorm": False})
        self.assertTrue(any(stage["query"] for stage in stages[1:]))
        self.assertTrue(any(stage["rmsnorm"] for stage in stages[2:]))
        for name, expected in frozen_before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], expected), name)

    def test_teacher_sources_are_detached_end_to_end(self):
        teacher = CURRENT["GPT"](tiny_config(enable_topdown=False)).eval()
        student = CURRENT["GPT"](tiny_config())
        student.freeze_for_topdown_training()
        idx = torch.randint(0, teacher.config.vocab_size, (2, teacher.config.block_size))
        with torch.no_grad():
            raw_sources = teacher.capture_residual_sources(
                idx, CURRENT["EXPERIMENT_2A0_SOURCE_DEPTHS"]
            )
        memory = CURRENT["shift_teacher_sources"](raw_sources)
        self.assertFalse(memory.requires_grad)
        _, loss = student(
            idx,
            idx,
            mode="masked_l1_topdown_teacher",
            feedback_sources=memory,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
