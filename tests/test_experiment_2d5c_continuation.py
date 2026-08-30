import copy
import math
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c as driver  # noqa: E402


def _small_specs(monkeypatch):
    specs = {
        section: {
            "tensor_count": 1,
            "max_abs_hard_cap": 10.0,
            "l2_norm_hard_cap": 10.0,
        }
        for section in ("gradients", "model_parameters", "optimizer_state")
    }
    monkeypatch.setattr(driver, "CONTINUATION_TENSOR_SECTION_SPECS", specs)


def _snapshot(offset=0.0):
    values = {
        "gradients": ("weight", torch.tensor([1.0 + offset, 2.0])),
        "model_parameters": ("weight", torch.tensor([3.0 + offset, 4.0])),
        "optimizer_state": (
            "weight::exp_avg", torch.tensor([5.0 + offset, 6.0]),
        ),
    }
    sections = {}
    for section, (key, tensor) in values.items():
        sections[section] = {
            key: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor": tensor,
            }
        }
    return {
        "schema": driver.CONTINUATION_TENSOR_SNAPSHOT_SCHEMA,
        "label": driver.CONTINUATION_TENSOR_SNAPSHOT_LABEL,
        "disposable": True,
        "official_training_updates_executed": 0,
        "sections": sections,
    }


def _envelope(left, right):
    observed = driver.continuation_tensor_pairwise_metrics(left, right)
    assert observed["passed"] is True
    sections = {}
    for section_name, section in observed["sections"].items():
        rows = []
        for row in section["tensors"]:
            rows.append({
                "key": row["key"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "observed_max_abs": row["max_abs"],
                "observed_l2_norm": row["l2_norm"],
                "max_abs_tolerance": max(1e-6, 4.0 * row["max_abs"]),
                "l2_norm_tolerance": max(1e-6, 4.0 * row["l2_norm"]),
            })
        sections[section_name] = {
            "tensor_count": len(rows),
            "key_sha256": driver.canonical_sha([row["key"] for row in rows]),
            "tensors": rows,
        }
    return {
        "schema": driver.CONTINUATION_TENSOR_ENVELOPE_SCHEMA,
        "sections": sections,
    }


def test_scalar_drift_requires_finite_bounded_same_sign_when_requested():
    passed = driver.scalar_drift(
        -0.04836, -0.04839,
        driver.CONTINUATION_GATE_GRADIENT_ABS_TOL,
        driver.CONTINUATION_GATE_GRADIENT_REL_TOL,
        require_same_sign=True,
    )
    assert passed["passed"] is True
    assert passed["same_sign"] is True

    sign_flip = driver.scalar_drift(
        -1e-8, 1e-8, 1.0, 1.0, require_same_sign=True
    )
    assert sign_flip["same_sign"] is False
    assert sign_flip["passed"] is False

    nonfinite = driver.scalar_drift(float("inf"), 1.0, 1.0, 1.0)
    assert nonfinite["passed"] is False


def test_sentinel_drift_enforces_both_maximum_and_mean_limits():
    reference = {
        "selected_logits_sha256": "a" * 64,
        "selected_logits": [[0.0] * 10],
    }
    natural = {
        "selected_logits_sha256": "b" * 64,
        "selected_logits": [[0.05] * 9 + [0.3]],
    }
    row = driver.sentinel_drift(reference, natural)
    assert row["bitwise_exact"] is False
    assert math.isclose(row["max_abs"], 0.3)
    assert math.isclose(row["mean_abs"], 0.075)
    assert row["passed"] is True

    excessive_maximum = {
        **natural,
        "selected_logits": [[0.0] * 9 + [0.51]],
    }
    assert driver.sentinel_drift(reference, excessive_maximum)["passed"] is False

    excessive_mean = {
        **natural,
        "selected_logits": [[0.11] * 10],
    }
    assert driver.sentinel_drift(reference, excessive_mean)["passed"] is False


def test_full_tensor_envelope_validates_and_comparison_passes(monkeypatch):
    _small_specs(monkeypatch)
    left, right = _snapshot(), _snapshot(1e-4)
    envelope = _envelope(left, right)

    validated = driver.validate_continuation_tensor_envelope(envelope)
    assert validated["passed"] is True
    assert validated["envelope"] == envelope

    comparison = driver.compare_continuation_tensor_snapshots(
        left, right, validated["envelope"]
    )
    assert comparison["passed"] is True
    assert all(
        section["tensor_count"] == 1 and section["passed"]
        for section in comparison["sections"].values()
    )


@pytest.mark.parametrize(
    "tamper",
    ("missing_tensor", "wrong_key_sha", "negative", "nonfinite", "over_cap"),
)
def test_full_tensor_envelope_rejects_tampering(monkeypatch, tamper):
    _small_specs(monkeypatch)
    envelope = _envelope(_snapshot(), _snapshot(1e-4))
    section = envelope["sections"]["gradients"]
    row = section["tensors"][0]
    if tamper == "missing_tensor":
        section["tensors"] = []
    elif tamper == "wrong_key_sha":
        section["key_sha256"] = "0" * 64
    elif tamper == "negative":
        row["max_abs_tolerance"] = -1.0
    elif tamper == "nonfinite":
        row["l2_norm_tolerance"] = float("inf")
    elif tamper == "over_cap":
        row["max_abs_tolerance"] = 10.01
    assert driver.validate_continuation_tensor_envelope(envelope)["passed"] is False


def test_full_tensor_comparison_rejects_missing_and_excessive_tensor(monkeypatch):
    _small_specs(monkeypatch)
    left, natural = _snapshot(), _snapshot(1e-4)
    envelope = _envelope(left, natural)

    missing = copy.deepcopy(natural)
    del missing["sections"]["optimizer_state"]["weight::exp_avg"]
    missing_result = driver.compare_continuation_tensor_snapshots(
        left, missing, envelope
    )
    assert missing_result["passed"] is False
    assert missing_result["sections"]["optimizer_state"]["keys_exact"] is False

    excessive = _snapshot(1.0)
    excessive_result = driver.compare_continuation_tensor_snapshots(
        left, excessive, envelope
    )
    assert excessive_result["passed"] is False
    assert all(
        section["tensors"][0]["passed"] is False
        for section in excessive_result["sections"].values()
    )


def test_disposable_smoke_requires_calibration_before_any_gpu_work():
    with pytest.raises(SystemExit, match="calibration"):
        driver.disposable_smoke(None, None, None, None, None, calibration=None)


def test_full_state_calibration_schema_and_frozen_hash_are_explicit():
    assert driver.CONTINUATION_CALIBRATION_SCHEMA.endswith("_v2")
    assert driver.CONTINUATION_TENSOR_ENVELOPE_SCHEMA.endswith("_v1")
    assert driver.CONTINUATION_CALIBRATION_SHA256 == (
        "4d6c15d927780dbf6597d38c054581761f31456a4b7256f47fca5c71eaa4740f"
    )
    assert driver.CONTINUATION_CALIBRATION_IMPLEMENTATION_COMMIT == (
        "f826e54c81243aab95fc9262b6e79167d985e5da"
    )
    assert driver.CONTINUATION_TENSOR_SECTION_SPECS["gradients"]["tensor_count"] == 152
    assert driver.CONTINUATION_TENSOR_SECTION_SPECS["model_parameters"]["tensor_count"] == 152
    assert driver.CONTINUATION_TENSOR_SECTION_SPECS["optimizer_state"]["tensor_count"] == 456
