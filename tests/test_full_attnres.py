#!/usr/bin/env python3
"""Correctness tests for Experiment 1 Full Attention Residuals."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_TAG = "baseline-gpt2-124m-10b"


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
    "experiment_under_test",
)


class FullAttnResTests(unittest.TestCase):
    def test_output_shape_dtype_and_device(self):
        router = CURRENT["FullAttnRes"](8)
        values = [torch.randn(2, 3, 8) for _ in range(4)]
        output = router(values)
        self.assertEqual(output.shape, (2, 3, 8))
        self.assertEqual(output.dtype, values[0].dtype)
        self.assertEqual(output.device, values[0].device)

    def test_one_source_is_exact_identity(self):
        router = CURRENT["FullAttnRes"](8)
        value = torch.randn(2, 3, 8)
        output = router([value])
        self.assertTrue(torch.equal(output, value))

    def test_zero_query_is_uniform(self):
        router = CURRENT["FullAttnRes"](8)
        values = [torch.randn(2, 3, 8) for _ in range(5)]
        _, weights = router(values, return_weights=True)
        expected = torch.full_like(weights, 1.0 / len(values))
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)

    def test_softmax_is_depth_only(self):
        router = CURRENT["FullAttnRes"](4)
        with torch.no_grad():
            router.query.copy_(torch.tensor([1.0, -0.5, 0.25, 0.75]))
        values = [torch.randn(2, 3, 4) + index for index in range(4)]
        _, weights = router(values, return_weights=True)
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(2, 3), rtol=1e-6, atol=1e-6)
        self.assertFalse(torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1))))
        self.assertFalse(torch.allclose(weights.sum(dim=2), torch.ones_like(weights.sum(dim=2))))

    def test_gradient_flow_and_finite_values(self):
        router = CURRENT["FullAttnRes"](8)
        with torch.no_grad():
            router.query.fill_(0.05)
        values = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(4)]
        output = router(values)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))
        for value in values:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(torch.count_nonzero(value.grad).item(), 0)
        for parameter in (router.query, router.norm.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

        config = CURRENT["GPTConfig"](
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=4,
            n_embd=16,
            residual_mode="full_attnres",
        )
        model = CURRENT["GPT"](config)
        with torch.no_grad():
            for depth, layer in enumerate(model.transformer.attnres):
                if depth > 0:
                    layer.query.fill_(0.05)
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        logits, model_loss = model(idx, idx)
        model_loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(model_loss))
        checked = [
            model.transformer.h[0].attn.c_attn.weight,
            model.transformer.h[0].mlp.c_fc.weight,
            model.transformer.attnres[1].query,
            model.transformer.attnres[1].norm.weight,
        ]
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_standard_mode_exactly_matches_frozen_baseline(self):
        baseline_source = subprocess.check_output(
            ["git", "show", f"{BASELINE_TAG}:train_gpt2.py"],
            cwd=REPO_ROOT,
            text=True,
        )
        baseline = load_symbols(baseline_source, f"{BASELINE_TAG}:train_gpt2.py", "frozen_regression_baseline")
        config_kwargs = dict(block_size=8, vocab_size=32, n_layer=2, n_head=4, n_embd=16)
        torch.manual_seed(1337)
        expected_model = baseline["GPT"](baseline["GPTConfig"](**config_kwargs)).eval()
        actual_model = CURRENT["GPT"](CURRENT["GPTConfig"](**config_kwargs, residual_mode="standard")).eval()
        actual_model.load_state_dict(expected_model.state_dict(), strict=True)
        idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [7, 6, 5, 4, 3, 2, 1, 0]])
        with torch.no_grad():
            expected_logits, expected_loss = expected_model(idx, idx)
            actual_logits, actual_loss = actual_model(idx, idx)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
