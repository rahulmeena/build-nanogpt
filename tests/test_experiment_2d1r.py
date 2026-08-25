import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d1 as d1  # noqa: E402
import experiment_2d1r as d1r  # noqa: E402


def test_frozen_resume_budget_and_geometry():
    assert d1r.START_UPDATE == 954
    assert d1r.FINAL_UPDATE == 4769
    assert d1r.ADDITIONAL_UPDATES == 3815
    assert d1r.ADDITIONAL_TARGETS == 2_000_158_720
    assert d1r.FINAL_TOTAL_TARGETS == 2_500_329_472
    assert d1r.SCIENTIFIC_UPDATES == (1000, 1100, 1200, 1908, 2862, 3815, 4769)
    assert d1r.STAGES is d1.STAGES
    assert d1r.TARGET_WINDOWS is d1.TARGET_WINDOWS


def test_exact_projection_is_identity_below_cap_and_bounded_above():
    weight = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.5, 0.25])))
    below = d1r.project_weight_(weight, 1.0)
    assert below["projection_scale"] == 1.0
    assert below["projection_applied"] is False
    assert torch.equal(weight, torch.diag(torch.tensor([1.0, 0.5, 0.25])))

    with torch.no_grad():
        weight.mul_(2.0)
    above = d1r.project_weight_(weight, 1.0)
    assert above["projection_applied"] is True
    assert abs(above["sigma_raw"] - 2.0) < 1e-6
    assert above["sigma_post"] <= 1.0 * (1.0 + d1r.PROJECTION_RELATIVE_TOLERANCE)
    assert abs(above["projection_scale"] - 0.5) < 1e-6


def test_projection_only_mutates_requested_weight():
    first = torch.nn.Parameter(torch.eye(4) * 2)
    untouched = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).view(4, 4))
    before = untouched.detach().clone()
    d1r.project_weight_(first, 1.0)
    assert torch.equal(untouched, before)


def test_projection_corrects_an_inconsistent_post_svd_verification():
    weight = torch.nn.Parameter(torch.eye(2) * 2)
    observed = iter((2.0, 1.00005, 0.99994))
    original = d1r.exact_spectral_norm
    d1r.exact_spectral_norm = lambda unused: next(observed)
    try:
        report = d1r.project_weight_(weight, 1.0)
    finally:
        d1r.exact_spectral_norm = original
    assert report["primary_projection_scale"] == 0.5
    assert report["corrective_projection_count"] == 1
    assert report["sigma_post"] <= 1.0 * (1.0 + d1r.PROJECTION_RELATIVE_TOLERANCE)
    assert report["projection_scale"] < report["primary_projection_scale"]


def test_schedule_and_pass_cadence_remain_frozen():
    assert d1r.stage_for_update(955) == d1.stage_for_update(955)
    assert d1r.stage_for_update(1909) == d1.stage_for_update(1909)
    assert d1r.stage_for_update(2863) == d1.stage_for_update(2863)
    assert d1r.pass_count_for_update(992) == 3
    assert d1r.pass_count_for_update(991) == 2
