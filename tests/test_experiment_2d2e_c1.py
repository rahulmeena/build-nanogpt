import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "experiment_2d2e_c1", ROOT / "scripts" / "experiment_2d2e_c1.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confirmatory_geometry_and_token_disjoint_offset():
    assert MODULE.TARGETS_PER_CONTROL == 1_048_576
    assert MODULE.EXPECTED_SEQUENCES == 1024
    old_last = MODULE.ORIGINAL_BATCHES * MODULE.B * MODULE.T
    assert MODULE.CONFIRM_START_TOKEN_OFFSET == old_last + 1


def test_paired_stats_use_comparator_minus_real():
    row = MODULE.paired_stats([1.0, 2.0, 3.0], [2.0, 1.0, 3.0])
    assert row["wins"] == 1
    assert row["losses"] == 1
    assert row["ties"] == 1
    assert row["mean_comparator_minus_real"] == 0.0


def test_bootstrap_is_reproducible_and_classification_preregistered():
    effects = np.linspace(0.01, 0.02, 32)
    first = MODULE.bootstrap_mean_ci(effects, seed=17, resamples=200)
    second = MODULE.bootstrap_mean_ci(effects, seed=17, resamples=200)
    assert first == second
    assert first["lower"] > 0
    assert MODULE.classify_confirmation(0.1, 0.2, first, second) == "STRONG CONFIRMATION"
    includes_zero = {"lower": -0.1, "upper": 0.1}
    assert MODULE.classify_confirmation(0.1, 0.2, first, includes_zero) == "DIRECTIONAL CONFIRMATION"
    assert MODULE.classify_confirmation(0.0, 0.2, first, second) == "NOT CONFIRMED"


def test_frozen_incremental_step_uses_two_value_no_diagnostics_contract():
    source = inspect.getsource(MODULE.evaluate_controls)
    assert "logits, state = model.incremental_step(" in source
    assert "logits, state, _ = model.incremental_step(" not in source


def test_cuda_uuid_manifest_is_stable_and_json_serializable():
    class FakeCUuuid:
        bytes = tuple(range(16))

    expected = {
        "format": "uuid-16-byte",
        "byte_count": 16,
        "hex": "000102030405060708090a0b0c0d0e0f",
        "canonical": "00010203-0405-0607-0809-0a0b0c0d0e0f",
    }
    first = MODULE.cuda_uuid_manifest(FakeCUuuid())
    second = MODULE.cuda_uuid_manifest(FakeCUuuid())
    assert first == expected
    assert second == expected
    assert json.loads(json.dumps(first, sort_keys=True)) == expected


def test_require_visible_a100_normalizes_cuda_uuid(monkeypatch):
    class FakeCUuuid:
        bytes = (255,) * 16

    class FakeProperties:
        name = "NVIDIA A100-SXM4-80GB"
        total_memory = 80 * 1024**3
        uuid = FakeCUuuid()

    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(MODULE.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        MODULE.torch.cuda, "get_device_properties", lambda _index: FakeProperties()
    )
    monkeypatch.setattr(MODULE.torch.cuda, "set_device", lambda _index: None)

    hardware = MODULE.require_visible_a100()
    assert hardware["uuid"] == {
        "format": "uuid-16-byte",
        "byte_count": 16,
        "hex": "ff" * 16,
        "canonical": "ffffffff-ffff-ffff-ffff-ffffffffffff",
    }
    json.dumps(hardware, sort_keys=True)
