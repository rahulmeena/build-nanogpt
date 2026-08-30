import inspect
import json
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d5c as driver  # noqa: E402


def tensor_with_gradient(gradient):
    value = torch.zeros_like(gradient, requires_grad=True)
    value.grad = gradient.clone()
    return value


def reference_gradient_rows(kind, gradients, lags, heads):
    rows = {
        name: {
            "opportunities": 0,
            "sum_pair_norm": 0.0,
            "sum_squared_gradient": 0.0,
            "nonzero_pairs": 0,
            "per_head": [
                {
                    "opportunities": 0,
                    "sum_pair_norm": 0.0,
                    "sum_squared_gradient": 0.0,
                    "nonzero_pairs": 0,
                }
                for _ in range(heads)
            ] if kind in ("key", "value") else None,
        }
        for name, _, _ in driver.RECURRENT_BINS
    }
    for gradient in gradients:
        gradient = gradient.detach().float().cpu()
        for name, low, high in driver.RECURRENT_BINS:
            mask = (lags >= low) & (lags <= high)
            if not int(mask.sum()):
                continue
            row = rows[name]
            if kind == "source":
                selected = gradient[:, mask, :]
                pair_norm = selected.norm(dim=-1)
                row["opportunities"] += int(pair_norm.numel())
                row["sum_pair_norm"] += float(pair_norm.sum())
                row["sum_squared_gradient"] += float(selected.square().sum())
                row["nonzero_pairs"] += int((pair_norm > 0).sum())
            else:
                selected = gradient[:, :, mask, :]
                pair_norm = selected.norm(dim=-1)
                row["opportunities"] += int(pair_norm.numel())
                row["sum_pair_norm"] += float(pair_norm.sum())
                row["sum_squared_gradient"] += float(selected.square().sum())
                row["nonzero_pairs"] += int((pair_norm > 0).sum())
                for head in range(pair_norm.size(1)):
                    head_row = row["per_head"][head]
                    head_norm = pair_norm[:, head]
                    head_selected = selected[:, head]
                    head_row["opportunities"] += int(head_norm.numel())
                    head_row["sum_pair_norm"] += float(head_norm.sum())
                    head_row["sum_squared_gradient"] += float(
                        head_selected.square().sum()
                    )
                    head_row["nonzero_pairs"] += int((head_norm > 0).sum())
    return rows


def assert_raw_row_matches(observed, expected):
    assert isinstance(observed["opportunities"], int)
    assert isinstance(observed["nonzero_pairs"], int)
    assert isinstance(observed["sum_pair_norm"], float)
    assert isinstance(observed["sum_squared_gradient"], float)
    assert observed["opportunities"] == expected["opportunities"]
    assert observed["nonzero_pairs"] == expected["nonzero_pairs"]
    assert math.isclose(
        observed["sum_pair_norm"],
        expected["sum_pair_norm"],
        rel_tol=1e-6,
        abs_tol=1e-8,
    )
    assert math.isclose(
        observed["sum_squared_gradient"],
        expected["sum_squared_gradient"],
        rel_tol=1e-6,
        abs_tol=1e-8,
    )


def test_gradient_bins_reduce_on_device_and_finalize_to_equivalent_json_scalars():
    heads = 2
    positions = torch.tensor([[0, 2, 5, 9, 17]], dtype=torch.long)
    query_position = 20
    lags = query_position - positions.reshape(-1)
    source_gradients = [
        torch.tensor(
            [[[1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0],
              [-1.0, 2.0, -2.0], [5.0, 0.0, 12.0]]]
        ),
        torch.tensor(
            [[[0.5, 1.5, -2.0], [2.0, 0.0, 0.0], [0.0, -3.0, 4.0],
              [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]
        ),
    ]
    key_gradients = [
        torch.arange(1, 21, dtype=torch.float32).reshape(1, heads, 5, 2),
        torch.arange(20, 0, -1, dtype=torch.float32).reshape(1, heads, 5, 2)
        / 3.0,
    ]
    value_gradients = [gradient.neg() / 2.0 for gradient in key_gradients]
    accumulator = driver.diagnostic_link_accumulator(heads)
    for source, key, value in zip(
        source_gradients, key_gradients, value_gradients
    ):
        driver.accumulate_gradient_bins(accumulator, {
            "query_position": query_position,
            "recurrent_positions": positions,
            "recurrent_source_reads": tensor_with_gradient(source),
            "recurrent_key_reads": tensor_with_gradient(key),
            "recurrent_value_reads": tensor_with_gradient(value),
        })

    covered = accumulator["gradient"]["source"]["2-7"]
    assert torch.is_tensor(covered["sum_pair_norm"])
    assert covered["sum_pair_norm"].device == source_gradients[0].device
    assert torch.is_tensor(covered["sum_squared_gradient"])
    assert torch.is_tensor(covered["nonzero_pairs"])

    driver.finalize_diagnostic_link(accumulator, "C", "b3")
    references = {
        "source": reference_gradient_rows(
            "source", source_gradients, lags, heads
        ),
        "key": reference_gradient_rows("key", key_gradients, lags, heads),
        "value": reference_gradient_rows(
            "value", value_gradients, lags, heads
        ),
    }
    for kind, expected_bins in references.items():
        for name, expected in expected_bins.items():
            observed = accumulator["gradient"][kind][name]
            assert_raw_row_matches(observed, expected)
            if expected["per_head"] is not None:
                for observed_head, expected_head in zip(
                    observed["per_head"], expected["per_head"]
                ):
                    assert_raw_row_matches(observed_head, expected_head)

    assert driver.finite_numeric_tree(accumulator)
    json.dumps(accumulator, allow_nan=False)
    reduction_source = inspect.getsource(driver.accumulate_gradient_bins)
    assert ".cpu()" not in reduction_source
    assert "float(pair_norm" not in reduction_source
    assert "float(selected" not in reduction_source


def test_writer_gradients_reduce_on_device_then_preserve_schema_and_finiteness():
    accumulator = driver.diagnostic_link_accumulator(heads=2)
    writer = accumulator["actual_writer_gradient"]
    missing = torch.zeros((1, 1, 4), requires_grad=True)
    zero = tensor_with_gradient(torch.zeros((1, 1, 4)))
    nonzero_gradient = torch.tensor([[[3.0, 4.0, 0.0, 12.0]]])
    nonzero = tensor_with_gradient(nonzero_gradient)
    for value in (missing, zero, nonzero):
        driver.accumulate_actual_writer_gradient(writer, value)

    assert writer["positions"] == 3
    assert writer["positions_with_gradient"] == 2
    assert torch.is_tensor(writer["sum_norm"])
    assert torch.is_tensor(writer["sum_squared_gradient"])
    assert torch.is_tensor(writer["positions_with_nonzero_gradient"])

    driver.finalize_diagnostic_link(accumulator, "C", "b5")
    assert writer["positions"] == 3
    assert writer["positions_with_gradient"] == 2
    assert writer["positions_with_nonzero_gradient"] == 1
    assert math.isclose(
        writer["sum_norm"], float(nonzero_gradient.norm()), rel_tol=1e-6
    )
    assert math.isclose(
        writer["sum_squared_gradient"],
        float(nonzero_gradient.square().sum()),
        rel_tol=1e-6,
    )
    assert math.isclose(
        writer["l2_norm_of_all_elements"],
        float(nonzero_gradient.norm()),
        rel_tol=1e-6,
    )
    assert writer["nonzero_back_to_actual_writer"]
    assert driver.finite_numeric_tree(accumulator)
    json.dumps(accumulator, allow_nan=False)


def test_nonfinite_device_reduction_remains_fail_closed_after_finalization():
    accumulator = driver.diagnostic_link_accumulator(heads=2)
    writer = accumulator["actual_writer_gradient"]
    value = tensor_with_gradient(
        torch.tensor([[[float("inf"), 0.0, 0.0, 0.0]]])
    )
    driver.accumulate_actual_writer_gradient(writer, value)
    driver.finalize_diagnostic_link(accumulator, "C", "b3")
    assert not driver.finite_numeric_tree(accumulator)


def load_tests(_loader, _tests, _pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(function)
        for function in (
            test_gradient_bins_reduce_on_device_and_finalize_to_equivalent_json_scalars,
            test_writer_gradients_reduce_on_device_then_preserve_schema_and_finiteness,
            test_nonfinite_device_reduction_remains_fail_closed_after_finalization,
        )
    )
