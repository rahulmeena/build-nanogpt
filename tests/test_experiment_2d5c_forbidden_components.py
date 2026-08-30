from copy import deepcopy
from types import SimpleNamespace
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d5c as driver  # noqa: E402


class AuditModel(nn.Module):
    def __init__(self, **config_overrides):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        config = {
            "residual_mode": "standard",
            # GPTConfig owns this field in standard mode, where it is inert.
            "attnres_rms_eps": 1e-5,
        }
        config.update(config_overrides)
        self.config = SimpleNamespace(**config)


class ExecutedAttnResProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class ModelWithExecutedAttnRes(AuditModel):
    def __init__(self):
        super().__init__()
        self.attnres_probe = ExecutedAttnResProbe()

    def forward(self, value):
        return self.attnres_probe(value) * self.weight


def audit(monkeypatch, model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    production_read_json = driver.read_json

    def compact_test_manifest(path):
        payload = deepcopy(production_read_json(path))
        payload["architecture_c"]["parameters"] = parameter_count
        return payload

    monkeypatch.setattr(
        driver,
        "PARAMETERS",
        parameter_count,
    )
    monkeypatch.setattr(driver, "read_json", compact_test_manifest)
    return driver.forbidden_component_audit(model)


def test_standard_mode_ignores_inert_attnres_rms_epsilon(monkeypatch):
    report = audit(monkeypatch, AuditModel())

    assert report["forbidden_runtime_config_hits"] == []
    assert report["module_checks"]["forbidden_runtime_config_absent"]
    assert report["passed"]


def test_active_full_attnres_mode_is_forbidden(monkeypatch):
    report = audit(monkeypatch, AuditModel(residual_mode="full_attnres"))

    assert not report["module_checks"]["forbidden_runtime_config_absent"]
    assert "residual_mode" in report["forbidden_runtime_config_hits"]
    assert not report["passed"]


def test_other_truthy_forbidden_runtime_config_is_rejected(monkeypatch):
    report = audit(monkeypatch, AuditModel(router=True))

    assert "router" in report["forbidden_runtime_config_hits"]
    assert not report["module_checks"]["forbidden_runtime_config_absent"]
    assert not report["passed"]


def test_executed_named_attnres_module_is_rejected(monkeypatch):
    model = ModelWithExecutedAttnRes()
    output = model(torch.tensor(2.0))
    assert output.item() == 2.0
    assert model.attnres_probe.calls == 1

    report = audit(monkeypatch, model)

    assert (
        "attnres_probe:executedattnresprobe"
        in report["forbidden_registered_name_hits"]
    )
    assert not report["module_checks"]["forbidden_registered_names_absent"]
    assert not report["passed"]
