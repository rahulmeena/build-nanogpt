#!/usr/bin/env python3
"""GPU evaluator for the zero-training Experiment 2D6F confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d3a as base  # noqa: E402
import experiment_2d4a as d4a  # noqa: E402
import experiment_2d6 as d6  # noqa: E402
import experiment_2d6_core as core  # noqa: E402


EXPERIMENT = "2D6F"
BRANCH = "experiment-2d6-fresh-panel-zero-training-confirmation"
FIXED_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
NEW_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
PANEL_NAME = "fresh disjoint confirmation panel"
PANEL_SEQUENCES = 2048
PANEL_TARGETS = 2_097_152
SENTINEL_SEQUENCES = 32
PARAMETERS = 124_475_908


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    d6.durable_json(Path(path), value)


def file_sha256(path):
    return d6.sha256(Path(path))


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def require_sealed_panel(panel_path):
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit("wrong 2D6F branch")
    if git("status", "--porcelain"):
        raise SystemExit("2D6F evaluator requires a clean worktree")
    relative = str(Path(panel_path).resolve().relative_to(REPO_ROOT))
    subprocess.check_call(["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT)
    panel = read_json(panel_path)
    checks = {
        "name": panel.get("panel_name") == PANEL_NAME,
        "sequence_count": panel.get("sequence_count") == PANEL_SEQUENCES,
        "targets": panel.get("targets_per_condition") == PANEL_TARGETS,
        "one_candidate": panel.get("candidate_panels_constructed") == 1,
        "seed": panel.get("selection_seed") == 2_026_090_201,
        "sealed_before_checkpoint_loading": panel.get("sealed_before_checkpoint_loading") is True,
        "no_loss_inspection": panel.get("checkpoint_losses_inspected_during_selection") is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"fresh panel scope failure: {checks}")
    return panel, {
        "git_commit": git("rev-parse", "HEAD"),
        "manifest_file_sha256": file_sha256(panel_path),
        "checks": checks,
        "passed": True,
    }


def require_checkpoint(path, expected, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise SystemExit(f"{label} SHA mismatch: {observed}")
    return path


def load_model(family, checkpoint, device):
    expected = FIXED_SHA256 if family == "fixed" else NEW_SHA256
    path = require_checkpoint(checkpoint, expected, family)
    payload = base.d0.torch_load(path, mmap=True)
    _, foundation = base.instantiate_base(device)
    if family == "fixed":
        model = base.AlternatingIntegrationRecurrentPyramidGPT(foundation).to(device)
        payload_checks = {
            "schema": payload.get("schema") == d4a.SCHEMA,
            "arm": payload.get("arm") == "fixed",
            "parent_sha": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
            "local_updates": payload.get("d4a_local_updates") == 191,
            "global_update": payload.get("inherited_global_update") == 2099,
            "cumulative_targets": payload.get("inherited_total_targets") == 1_100_480_512,
        }
        architecture_fingerprint = "accepted-2d3a-fixed-w512-b7-to-b6-real"
    elif family == "new":
        model = core.B6NativeNoB7RecurrenceGPT(foundation).to(device)
        payload_checks = {
            "schema": payload.get("schema") == d6.SCHEMA,
            "arm": payload.get("arm") == "NEW",
            "parent_sha": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
            "fixed_control_sha": payload.get("fixed_control_checkpoint_sha256") == FIXED_SHA256,
            "local_updates": payload.get("local_updates") == 191,
            "global_update": payload.get("global_update") == 2099,
            "cumulative_targets": payload.get("cumulative_targets") == 1_100_480_512,
            "architecture_fingerprint": payload.get("architecture_fingerprint") == core.ARCHITECTURE_FINGERPRINT,
        }
        architecture_fingerprint = core.ARCHITECTURE_FINGERPRINT
    else:
        raise SystemExit("family must be fixed or new")
    incompatible = model.load_state_dict(payload["model"], strict=True)
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    if family == "fixed":
        architecture_checks = {
            "b6_window_512": base.LOCAL_WINDOWS[5] == 512,
            "b7_ring_present": hasattr(state, "h7_ring"),
            "b7_to_b6_source": base.SOURCE_KEYS["b6"] == "h7",
            "b1_b3_b5_sources": {key: base.SOURCE_KEYS[key] for key in ("b1", "b3", "b5")}
            == {"b1": "h12", "b3": "h10", "b5": "h8"},
        }
    else:
        audit = model.incremental_cache_audit(state)
        architecture_checks = {
            "b6_window_1024": next(
                row[2] for row in core.BLOCK_GEOMETRY if row[0] == 6
            ) == 1024,
            "b7_ring_absent": not hasattr(state, "h7_ring"),
            "b7_ring_audit_absent": "h7" not in audit.get("ring_lengths", {}),
            "b6_recurrent_computation_absent": core.ARCHITECTURE_MANIFEST[
                "b7_to_b6_computational_link"
            ] is False,
            "b1_b3_b5_unchanged": core.ARCHITECTURE_MANIFEST[
                "active_writers"
            ] == {"B1": "B12", "B3": "B10", "B5": "B8"},
            "architecture_method": model.architecture_fingerprint() == core.ARCHITECTURE_FINGERPRINT,
        }
    checks = {
        **payload_checks,
        "checkpoint_sha": file_sha256(path) == expected,
        "missing_tensors": list(incompatible.missing_keys) == [],
        "unexpected_tensors": list(incompatible.unexpected_keys) == [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS,
        **architecture_checks,
    }
    if not all(checks.values()):
        raise SystemExit(f"{family} checkpoint/architecture validation failed: {checks}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    del payload, state
    gc.collect()
    torch.cuda.empty_cache()
    return model, {
        "path": str(path),
        "sha256": expected,
        "bytes": path.stat().st_size,
        "architecture_fingerprint": architecture_fingerprint,
        "checks": checks,
        "passed": True,
    }


def parameter_digest(model):
    digest = hashlib.sha256()
    seen = set()
    tensors = 0
    parameters = 0
    for name, parameter in model.named_parameters():
        identity = (parameter.device.type, parameter.data_ptr(), parameter.numel(), parameter.element_size())
        if identity in seen:
            continue
        seen.add(identity)
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
        tensors += 1
        parameters += value.numel()
    return {"sha256": digest.hexdigest(), "unique_tensors": tensors, "parameters": parameters}


def batch_for_panel(val_path, panel, ordinal, device=None):
    index = int(panel["batch_indices_in_evaluation_order"][ordinal])
    x, y = d6.d5c.batch_at_index(val_path, index)
    observed = base.batch_identity(x, y)
    if observed != panel["batch_identities"][ordinal]:
        raise SystemExit(f"panel batch identity mismatch at ordinal {ordinal}")
    if device is not None:
        x, y = x.to(device), y.to(device)
    return index, x, y, observed


def sentinel(family, checkpoint, reused_panel, reused_losses, val_path, device):
    model, model_identity = load_model(family, checkpoint, device)
    _, x, y, observed = batch_for_panel(val_path, reused_panel, 0, device)
    # Preserve the accepted evaluator's batch-of-64 BF16 kernel geometry, then
    # compare only the preregistered first 32 rows.  Slicing the input batch
    # changes kernel numerics and is not a reproduction of the stored panel.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = d6.incremental_condition(model, x, y, control="all_real", audit=False)
    expected_key = "fixed_real" if family == "fixed" else "new_real"
    expected = np.asarray(reused_losses[expected_key][:SENTINEL_SEQUENCES], dtype=np.float64)
    observed_losses = np.asarray(
        result["per_sequence_ce"][:SENTINEL_SEQUENCES], dtype=np.float64
    )
    delta = np.abs(observed_losses - expected)
    row = {
        "family": family,
        "checkpoint_sha256": model_identity["sha256"],
        "sequences": SENTINEL_SEQUENCES,
        "targets": SENTINEL_SEQUENCES * base.T,
        "reused_panel_sha256": reused_panel["panel_sha256"],
        "first_batch_identity": observed,
        "max_abs_ce": float(delta.max()),
        "mean_abs_ce": float(delta.mean()),
        "tolerance": 1e-12,
        "cache_audit_passed": result["final_cache_audit"]["passed"],
        "passed": float(delta.max()) <= 1e-12 and result["final_cache_audit"]["passed"],
    }
    del model, x, y, result
    gc.collect()
    torch.cuda.empty_cache()
    if not row["passed"]:
        raise SystemExit(f"{family} evaluator sentinel failed: {row}")
    return row, model_identity


def empty_resume(condition, checkpoint_sha, panel, panel_file_sha, commit):
    return {
        "schema": "experiment_2d6f_condition_resume_v1",
        "condition": condition,
        "checkpoint_sha256": checkpoint_sha,
        "panel_name": PANEL_NAME,
        "panel_sha256": panel["panel_sha256"],
        "panel_manifest_sha256": panel_file_sha,
        "evaluator_git_commit": commit,
        "batch_indices_in_evaluation_order": panel["batch_indices_in_evaluation_order"],
        "completed_batch_indices": [],
        "batch_identities": [],
        "aggregate_nll": 0.0,
        "targets": 0,
        "per_sequence_nll": [],
        "per_sequence_ce": [],
        "per_sequence_targets": [],
        "wall_seconds": 0.0,
        "status": "running",
    }


def evaluate_condition(family, checkpoint, panel, panel_file_sha, val_path, output, device):
    condition = "FRESH_FIXED_REAL" if family == "fixed" else "FRESH_NEW_REAL"
    expected_sha = FIXED_SHA256 if family == "fixed" else NEW_SHA256
    resume_path = output / f".{condition}.resume.json"
    commit = git("rev-parse", "HEAD")
    if resume_path.exists():
        state = read_json(resume_path)
        identity_ok = (
            state.get("condition") == condition
            and state.get("checkpoint_sha256") == expected_sha
            and state.get("panel_sha256") == panel["panel_sha256"]
            and state.get("panel_manifest_sha256") == panel_file_sha
            and state.get("evaluator_git_commit") == commit
        )
        if not identity_ok:
            raise SystemExit(f"{condition} resume identity mismatch")
    else:
        state = empty_resume(condition, expected_sha, panel, panel_file_sha, commit)
    model, model_identity = load_model(family, checkpoint, device)
    before = parameter_digest(model)
    completed = set(state["completed_batch_indices"])
    started = time.monotonic()
    with torch.inference_mode():
        for ordinal, expected_index in enumerate(panel["batch_indices_in_evaluation_order"]):
            index = int(expected_index)
            if index in completed:
                continue
            _, x, y, observed = batch_for_panel(val_path, panel, ordinal, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = d6.incremental_condition(model, x, y, control="all_real", audit=False)
            state["aggregate_nll"] += row["nll_sum"]
            state["targets"] += row["targets"]
            state["per_sequence_nll"].extend(row["per_sequence_nll"])
            state["per_sequence_ce"].extend(row["per_sequence_ce"])
            state["per_sequence_targets"].extend([base.T] * x.size(0))
            state["completed_batch_indices"].append(index)
            state["batch_identities"].append(observed)
            state["final_cache_audit"] = row["final_cache_audit"]
            state["wall_seconds"] += time.monotonic() - started
            started = time.monotonic()
            durable_json(resume_path, state)
            print(f"2D6F {condition} batch {ordinal + 1}/32", flush=True)
            del x, y, row
            torch.cuda.empty_cache()
    after = parameter_digest(model)
    state["aggregate_ce"] = state["aggregate_nll"] / state["targets"]
    state["paired_sequences"] = len(state["per_sequence_ce"])
    state["model_parameter_digest_before"] = before
    state["model_parameter_digest_after"] = after
    state["parameter_state_unchanged"] = before == after
    state["gradients_all_none"] = all(parameter.grad is None for parameter in model.parameters())
    state["status"] = "complete"
    state["passed"] = (
        len(state["completed_batch_indices"]) == 32
        and state["paired_sequences"] == PANEL_SEQUENCES
        and state["targets"] == PANEL_TARGETS
        and len(state["per_sequence_targets"]) == PANEL_SEQUENCES
        and set(state["per_sequence_targets"]) == {base.T}
        and state["batch_identities"] == panel["batch_identities"]
        and state["final_cache_audit"]["passed"]
        and state["parameter_state_unchanged"]
        and state["gradients_all_none"]
    )
    durable_json(resume_path, state)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not state["passed"]:
        raise SystemExit(f"{condition} evaluation audit failed")
    return state, model_identity, resume_path


def run_preflight(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel, seal = require_sealed_panel(args.fresh_panel)
    disjointness = read_json(args.disjointness_audit)
    reused_panel = read_json(args.reused_panel)
    reused_losses = read_json(args.reused_losses)
    if not disjointness.get("all_required_disjointness_passed"):
        raise SystemExit("fresh disjointness audit is not passed")
    if reused_losses.get("panel_sha256") != reused_panel.get("panel_sha256"):
        raise SystemExit("reused panel/loss identity mismatch")
    val_path = base.validation_path(args.data_root)
    if file_sha256(val_path) != panel["dataset_sha256"]:
        raise SystemExit("validation dataset identity mismatch")
    device = base.require_a100()
    before = {
        "fixed": {"sha256": file_sha256(args.fixed_checkpoint), "bytes": Path(args.fixed_checkpoint).stat().st_size},
        "new": {"sha256": file_sha256(args.new_checkpoint), "bytes": Path(args.new_checkpoint).stat().st_size},
    }
    if before["fixed"]["sha256"] != FIXED_SHA256 or before["new"]["sha256"] != NEW_SHA256:
        raise SystemExit("sealed checkpoint SHA preflight failed")
    pre = {}
    identities = {}
    pre["fixed"], identities["fixed"] = sentinel(
        "fixed", args.fixed_checkpoint, reused_panel, reused_losses, val_path, device
    )
    pre["new"], identities["new"] = sentinel(
        "new", args.new_checkpoint, reused_panel, reused_losses, val_path, device
    )
    checkpoint_identity = {
        "experiment": EXPERIMENT,
        "panel_seal": seal,
        "fixed": {**identities["fixed"], "sha256_before_evaluation": before["fixed"]["sha256"], "bytes_before_evaluation": before["fixed"]["bytes"]},
        "new": {**identities["new"], "sha256_before_evaluation": before["new"]["sha256"], "bytes_before_evaluation": before["new"]["bytes"]},
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "passed_preflight": identities["fixed"]["passed"] and identities["new"]["passed"],
    }
    sentinel_audit = {
        "experiment": EXPERIMENT,
        "evaluator_git_commit": seal["git_commit"],
        "reused_panel_sha256": reused_panel["panel_sha256"],
        "sentinel_sequences_per_checkpoint": SENTINEL_SEQUENCES,
        "pre_evaluation": pre,
        "pre_evaluation_passed": all(row["passed"] for row in pre.values()),
        "post_evaluation": None,
        "passed": False,
    }
    durable_json(output / "CHECKPOINT_IDENTITY.json", checkpoint_identity)
    durable_json(output / "EVALUATOR_SENTINEL_AUDIT.json", sentinel_audit)
    print("EXPERIMENT_2D6F_PREFLIGHT_PASS", seal["git_commit"], flush=True)


def run_evaluate(args):
    output = Path(args.output_dir).resolve()
    panel, seal = require_sealed_panel(args.fresh_panel)
    checkpoint_identity = read_json(output / "CHECKPOINT_IDENTITY.json")
    sentinel_audit = read_json(output / "EVALUATOR_SENTINEL_AUDIT.json")
    if not checkpoint_identity.get("passed_preflight") or not sentinel_audit.get("pre_evaluation_passed"):
        raise SystemExit("2D6F evaluation lacks a passed preflight")
    reused_panel = read_json(args.reused_panel)
    reused_losses = read_json(args.reused_losses)
    val_path = base.validation_path(args.data_root)
    device = base.require_a100()
    panel_file_sha = file_sha256(args.fresh_panel)
    fixed, fixed_identity, fixed_resume = evaluate_condition(
        "fixed", args.fixed_checkpoint, panel, panel_file_sha, val_path, output, device
    )
    new, new_identity, new_resume = evaluate_condition(
        "new", args.new_checkpoint, panel, panel_file_sha, val_path, output, device
    )
    post = {}
    post["fixed"], _ = sentinel(
        "fixed", args.fixed_checkpoint, reused_panel, reused_losses, val_path, device
    )
    post["new"], _ = sentinel(
        "new", args.new_checkpoint, reused_panel, reused_losses, val_path, device
    )
    checkpoint_identity["fixed"].update({
        "sha256_after_evaluation": file_sha256(args.fixed_checkpoint),
        "bytes_after_evaluation": Path(args.fixed_checkpoint).stat().st_size,
        "parameter_digest_before": fixed["model_parameter_digest_before"],
        "parameter_digest_after": fixed["model_parameter_digest_after"],
        "parameter_state_unchanged": fixed["parameter_state_unchanged"],
    })
    checkpoint_identity["new"].update({
        "sha256_after_evaluation": file_sha256(args.new_checkpoint),
        "bytes_after_evaluation": Path(args.new_checkpoint).stat().st_size,
        "parameter_digest_before": new["model_parameter_digest_before"],
        "parameter_digest_after": new["model_parameter_digest_after"],
        "parameter_state_unchanged": new["parameter_state_unchanged"],
    })
    checkpoint_identity["checkpoint_files_unchanged"] = (
        checkpoint_identity["fixed"]["sha256_after_evaluation"] == FIXED_SHA256
        and checkpoint_identity["new"]["sha256_after_evaluation"] == NEW_SHA256
        and checkpoint_identity["fixed"]["bytes_after_evaluation"] == checkpoint_identity["fixed"]["bytes_before_evaluation"]
        and checkpoint_identity["new"]["bytes_after_evaluation"] == checkpoint_identity["new"]["bytes_before_evaluation"]
    )
    checkpoint_identity["passed"] = (
        checkpoint_identity["passed_preflight"]
        and checkpoint_identity["checkpoint_files_unchanged"]
        and fixed_identity["passed"]
        and new_identity["passed"]
        and fixed["parameter_state_unchanged"]
        and new["parameter_state_unchanged"]
    )
    sentinel_audit["post_evaluation"] = post
    sentinel_audit["post_evaluation_passed"] = all(row["passed"] for row in post.values())
    sentinel_audit["no_mutable_state_leakage"] = all(
        sentinel_audit["pre_evaluation"][family]["max_abs_ce"] <= 1e-12
        and post[family]["max_abs_ce"] <= 1e-12
        for family in ("fixed", "new")
    )
    sentinel_audit["passed"] = (
        sentinel_audit["pre_evaluation_passed"]
        and sentinel_audit["post_evaluation_passed"]
        and sentinel_audit["no_mutable_state_leakage"]
    )
    combined = {
        "schema": "experiment_2d6f_fresh_per_sequence_losses_v1",
        "experiment": EXPERIMENT,
        "panel_name": PANEL_NAME,
        "panel_sha256": panel["panel_sha256"],
        "panel_manifest_sha256": panel_file_sha,
        "panel_sealed_git_commit": seal["git_commit"],
        "evaluator_git_commit": seal["git_commit"],
        "precision": "BF16 model execution; FP32 CE logits; FP64 per-sequence and aggregate accumulation",
        "execution": "deployment-equivalent true incremental, one causal token step at a time",
        "sequence_order": panel["batch_indices_in_evaluation_order"],
        "sequences_per_condition": PANEL_SEQUENCES,
        "targets_per_condition": PANEL_TARGETS,
        "conditions_evaluated": ["FRESH_FIXED_REAL", "FRESH_NEW_REAL"],
        "fixed": fixed,
        "new": new,
        "zero_training_counts": {
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameter_updates": 0,
            "scheduler_steps": 0,
            "training_targets": 0,
            "new_checkpoints": 0,
        },
        "passed": fixed["passed"] and new["passed"] and sentinel_audit["passed"] and checkpoint_identity["passed"],
    }
    durable_json(output / "FRESH_PER_SEQUENCE_LOSSES.json", combined)
    durable_json(output / "CHECKPOINT_IDENTITY.json", checkpoint_identity)
    durable_json(output / "EVALUATOR_SENTINEL_AUDIT.json", sentinel_audit)
    fixed_resume.unlink()
    new_resume.unlink()
    if not combined["passed"]:
        raise SystemExit("2D6F fresh evaluation failed")
    print("EXPERIMENT_2D6F_FRESH_EVALUATION_PASS", flush=True)
    print("FRESH_FIXED_REAL_CE", fixed["aggregate_ce"], flush=True)
    print("FRESH_NEW_REAL_CE", new["aggregate_ce"], flush=True)


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    for name, handler in (("preflight", run_preflight), ("evaluate", run_evaluate)):
        current = commands.add_parser(name)
        current.set_defaults(handler=handler)
        for argument in (
            "fixed_checkpoint",
            "new_checkpoint",
            "reused_panel",
            "reused_losses",
            "fresh_panel",
            "disjointness_audit",
            "data_root",
            "output_dir",
        ):
            current.add_argument(f"--{argument.replace('_', '-')}", required=True)
    return root


def main():
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
