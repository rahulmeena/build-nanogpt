#!/usr/bin/env python3
"""Detached-state, gradient-scope, and protocol tests for Experiment 2B1."""

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


torch.set_num_threads(1)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2b1 as b1  # noqa: E402


def definition_prefix(source):
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    return source.split(marker)[0]


CURRENT = {"__name__": "experiment_2b1_under_test", "master_process": True}
exec(
    compile(
        definition_prefix((REPO_ROOT / "train_gpt2.py").read_text()),
        str(REPO_ROOT / "train_gpt2.py"),
        "exec",
    ),
    CURRENT,
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
        router = model.transformer.topdown_attnres
        router.query.copy_(torch.linspace(-0.3, 0.3, config.n_embd))
        router.norm.weight.copy_(torch.linspace(0.8, 1.2, config.n_embd))
        router.gate.fill_(0.25)
    model.freeze_for_topdown_training()
    return model


def reader_gradients(model, tokens, targets, chunk):
    model.zero_grad(set_to_none=True)
    state = model.init_recurrent_state(
        tokens.size(0), "masked_l1_topdown_self", dtype=torch.float32
    )
    pending = None
    for position in range(tokens.size(1)):
        logits, state = model.forward_step(tokens[:, position], state)
        loss = F.cross_entropy(logits[:, 0], targets[:, position], reduction="sum")
        loss = loss / targets.numel()
        pending = loss if pending is None else pending + loss
        if (position + 1) % chunk == 0 or position + 1 == tokens.size(1):
            pending.backward()
            pending = None
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return gradients, state


class DetachedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260815)
        self.tokens = torch.randint(0, 32, (2, 8))
        self.targets = torch.randint(0, 32, (2, 8))

    def test_chunked_backward_matches_per_token_reference(self):
        reference = tiny_model()
        chunked = tiny_model()
        chunked.load_state_dict(reference.state_dict(), strict=True)
        reference_gradients, _ = reader_gradients(
            reference, self.tokens, self.targets, chunk=1
        )
        chunked_gradients, state = reader_gradients(
            chunked, self.tokens, self.targets, chunk=8
        )
        self.assertEqual(set(reference_gradients), b1.TRAINABLE_NAMES)
        for name in reference_gradients:
            torch.testing.assert_close(
                chunked_gradients[name],
                reference_gradients[name],
                rtol=2e-5,
                atol=2e-6,
            )
        self.assertIsNone(state.feedback_memory.grad_fn)
        self.assertFalse(state.feedback_memory.requires_grad)
        for cache in state.kv_caches[1:]:
            key, value = cache.prefix()
            self.assertIsNone(key.grad_fn)
            self.assertIsNone(value.grad_fn)
            self.assertFalse(key.requires_grad)
            self.assertFalse(value.requires_grad)

    def test_later_loss_does_not_reach_prior_memory_or_kv(self):
        model = tiny_model(block_size=3)
        tokens = self.tokens[:, :3]
        targets = self.targets[:, :3]
        marker = {"position": -1}
        captured = {}

        def hook(name):
            def capture(_module, _inputs, output):
                if marker["position"] == 1:
                    tensor = output[0] if isinstance(output, tuple) else output
                    tensor.retain_grad()
                    captured[name] = tensor
            return capture

        modules = {
            "v16": model.transformer.h[7].mlp,
            "v17": model.transformer.h[8].attn.c_proj,
            "v20": model.transformer.h[9].mlp,
            "v24": model.transformer.h[11].mlp,
            "kv": model.transformer.h[1].attn.c_attn,
        }
        handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
        try:
            state = model.init_recurrent_state(2, "masked_l1_topdown_self")
            for position in range(3):
                marker["position"] = position
                logits, state = model.forward_step(tokens[:, position], state)
                self.assertIsNone(state.feedback_memory.grad_fn)
                for cache in state.kv_caches[1:]:
                    self.assertIsNone(cache.key.grad_fn)
                    self.assertIsNone(cache.value.grad_fn)
            F.cross_entropy(logits[:, 0], targets[:, 2]).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(set(captured), set(modules))
        self.assertTrue(all(tensor.grad is None for tensor in captured.values()))
        for name, parameter in model.named_parameters():
            if name in b1.TRAINABLE_NAMES:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
            else:
                self.assertIsNone(parameter.grad)

    def test_training_prefix_clone_preserves_forward_values(self):
        inference = tiny_model()
        training = tiny_model()
        training.load_state_dict(inference.state_dict(), strict=True)
        inference_state = inference.init_recurrent_state(2, "masked_l1_topdown_self")
        training_state = training.init_recurrent_state(2, "masked_l1_topdown_self")
        inference_rows = []
        training_rows = []
        for position in range(8):
            with torch.no_grad():
                logits, inference_state = inference.forward_step(
                    self.tokens[:, position], inference_state
                )
            inference_rows.append(logits)
            logits, training_state = training.forward_step(
                self.tokens[:, position], training_state
            )
            training_rows.append(logits.detach())
        self.assertTrue(torch.equal(torch.cat(inference_rows, 1), torch.cat(training_rows, 1)))


class ProtocolTests(unittest.TestCase):
    def test_classification_uses_both_gains(self):
        self.assertEqual(
            b1.classification(0.02, 0.001),
            "SELF-ADAPTATION IMPROVES SEQUENCE MEMORY",
        )
        self.assertEqual(
            b1.classification(0.02, 0.0),
            "SELF-ADAPTATION IMPROVES GENERIC COMPENSATION ONLY",
        )
        self.assertEqual(
            b1.classification(0.005, 0.1), "SELF-ADAPTATION IS NEUTRAL"
        )
        self.assertEqual(
            b1.classification(-0.02, 0.1), "SELF-ADAPTATION DEGRADES"
        )
        self.assertEqual(
            b1.classification(0.02, 0.1, invariants_passed=False),
            "SELF-ADAPTATION IS UNSTABLE",
        )

    def test_frozen_constants(self):
        config = b1.validate_config()
        self.assertEqual(config["source_depths"], [16, 17, 20, 24])
        self.assertEqual(config["trainable_parameters"], 1537)
        self.assertEqual(config["processed_2b1_tokens"], 5_242_880)
        self.assertEqual(config["teacher_training_forward_calls"], 0)
        self.assertEqual(config["hellaswag"], "not run without separate approval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
