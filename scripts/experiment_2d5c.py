#!/usr/bin/env python3
"""Experiment 2D5C fixed-writer B3/B5 W2 representation-pressure driver.

This driver is deliberately fail closed.  It can train only arm C, only from
the accepted 2D3A-1B checkpoint, and only in the two official segments
0->96 and 96->191.  Evaluation of the existing Fixed-100M checkpoint never
constructs or invokes an optimizer step.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

import experiment_2d3a as base
import experiment_2d3a_1b as parent
import experiment_2d4a as d4a
import experiment_2d5c_analysis as analysis
import experiment_2d5c_artifacts as artifacts
import experiment_2d5c_core as core
import experiment_2d5c_finalizer as finalizer


EXPERIMENT = "2D5C"
PROTOCOL = "fixed_writer_b3_b5_w2_representation_pressure_matched_100m_v1"
PROTOCOL_SHA256 = "eae21daf3859eb70d3342261d2891fbb4b2114235b1a87450eb2b4d201183b70"
BRANCH = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m"
FINAL_TAG = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final"
SCHEMA = "exp2d5c_fixed_writer_b3_b5_w2_checkpoint_v1"

SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
SOURCE_NEXT_BATCH = "61dd83544d83c0cf7b4d61005f5a9cf64e2cafa930af1819cba2aae4538e7e61"
SOURCE_NEXT_STREAM = "39f6599f552803150fad33d32aa9c4df5843b058410ff1ba38b5afa469046e97"
SOURCE_COMMIT = "bf977013f5ca359e64d86eb896d445160c49c6bf"
SOURCE_TAG = "experiment-2d3a-alternating-integration-pyramid-1b-final"
TOOLING_COMMIT = "81bd84ff2f7b2d21f0bdd844d4d3f9326147e761"
D4A_FINAL_TAG = "experiment-2d4a-matched-source-depth-routing-250m-final"
D4A_FINAL_TAG_COMMIT = "7d29783682ed63544c3673a880dcb32725e89431"
SOURCE_UPDATES = 1_908
SOURCE_TARGETS = 1_000_341_504

CONTROL_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
CONTROL_NEXT_BATCH = "62800455f294aaf110fbfc024abaa601c30f45d96175acb795a4d162d53da097"
CONTROL_NEXT_STREAM = "cdfd4afb20c268d69e3e3fbc1c39076af21719ba6d2a6636180f12b1afd5a157"

LOCAL_UPDATES = 191
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2_099
FINAL_CUMULATIVE_TARGETS = 1_100_480_512
RESTART_LOCAL_UPDATE = 96
MILESTONES = (48, 96, 144, 191)
MILESTONE_TARGETS = {
    48: 1_025_507_328,
    96: 1_050_673_152,
    144: 1_075_838_976,
    191: 1_100_480_512,
}
PARAMETERS = 124_475_908
CORE_SHA256 = "8befbf790b3e522747cd39da306ec124464bf8dde1604caf64f299efa7e36216"
LARGE_SELECTION_SEED = 2_026_083_001
SHUFFLE_SEED = 2_026_083_002
BOOTSTRAP_SEED = 2_026_083_003
BOOTSTRAP_RESAMPLES = 50_000
NONINFERIORITY_MARGIN = 0.001
REPRESENTATION_DIAGNOSTIC_SCHEMA = (
    "experiment_2d5c_representation_pressure_diagnostic_v2"
)
POD_ID = "h6of430yxncf6h"
POD_NAME = "opposite_azure_ladybug"
VOLUME_ID = "yhzyb27fb5"

CONTROLS = (
    "all_real",
    "b3_off",
    "b3_shuffled",
    "b5_off",
    "b5_shuffled",
    "b3_b5_off",
    "b3_b5_shuffled",
)
RECURRENT_BINS = (
    ("2-7", 2, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    base.durable_json(Path(path), value)


def durable_text(path, value):
    base.durable_text(Path(path), value)


def append_jsonl(path, value):
    base.append_jsonl(Path(path), value)


def sha256(path):
    return base.file_sha256(Path(path))


def file_identity(path):
    """Return a byte identity for an existing regular artifact."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def tensor_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def git_is_ancestor(ancestor, descendant="HEAD"):
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT, check=False, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def implementation_file_sha256():
    relative = (
        "configs/exp2d5c_fixed_writer_b3_b5_w2_matched_100m.json",
        "scripts/experiment_2d5c.py",
        "scripts/experiment_2d5c_analysis.py",
        "scripts/experiment_2d5c_artifacts.py",
        "scripts/experiment_2d5c_core.py",
        "scripts/experiment_2d5c_complete.py",
        "scripts/experiment_2d5c_finalizer.py",
        "scripts/experiment_2d5c_runpod_guard.py",
        "scripts/experiment_2d5c_workflow.py",
    )
    return {name: sha256(REPO_ROOT / name) for name in relative}


def require_branch(clean=False):
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {branch}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("Git worktree must be clean before scientific execution")


def require_exact_file(path, expected, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"wrong {label} SHA-256: {actual}")
    return path


def architecture_manifest(family="C"):
    if family not in ("C", "Fixed"):
        raise ValueError(family)
    pressure = family == "C"
    b3_window = 2 if pressure else 32
    b5_window = 2 if pressure else 64
    manifest = {
        "experiment": EXPERIMENT,
        "family": family,
        "model_weight_lineage": SOURCE_SHA256 if pressure else CONTROL_SHA256,
        "code_index_to_block": {str(index): f"B{index + 1}" for index in range(12)},
        "parameters": PARAMETERS,
        "trainable_parameters": PARAMETERS,
        "new_parameters": 0,
        "state_dict_key_changes": 0,
        "blocks": {
            "B1": {"code_index": 0, "local_window": 2, "writer": "B12", "recurrent_lags": [2, 1023]},
            "B2": {"code_index": 1, "local_window": 1024, "writer": None, "recurrent_lags": None},
            "B3": {"code_index": 2, "local_window": b3_window, "writer": "B10", "recurrent_lags": [b3_window, 1023]},
            "B4": {"code_index": 3, "local_window": 1024, "writer": None, "recurrent_lags": None},
            "B5": {"code_index": 4, "local_window": b5_window, "writer": "B8", "recurrent_lags": [b5_window, 1023]},
            "B6": {"code_index": 5, "local_window": 512, "writer": "B7", "recurrent_lags": [512, 1023]},
            **{
                f"B{index}": {"code_index": index - 1, "local_window": 1024, "writer": None, "recurrent_lags": None}
                for index in range(7, 13)
            },
        },
        "fixed_writer_identity": {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"},
        "ring_capacity": 1023,
        "recurrent_ring_byte_change_vs_fixed": 0,
        "router": False,
        "auxiliary_objective": False,
        "ce_only": True,
    }
    manifest["fingerprint_sha256"] = canonical_sha(manifest)
    return manifest


ARCHITECTURE_C = architecture_manifest("C")
ARCHITECTURE_FIXED = architecture_manifest("Fixed")
ARCHITECTURE_FINGERPRINT_C = ARCHITECTURE_C["fingerprint_sha256"]
ARCHITECTURE_FINGERPRINT_FIXED = ARCHITECTURE_FIXED["fingerprint_sha256"]


def parameter_manifest(model):
    return d4a.named_model_manifest(model)


def optimizer_manifest(model, optimizer):
    names = {parameter: name for name, parameter in model.named_parameters()}
    trainable = set(names)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    coverage = set(grouped) == trainable and len(grouped) == len(set(grouped))
    state_shapes = []
    malformed = []
    for parameter, state in optimizer.state.items():
        name = names.get(parameter, "<unknown>")
        for state_name, value in sorted(state.items()):
            if not torch.is_tensor(value):
                continue
            row = {
                "parameter": name,
                "state": state_name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
            state_shapes.append(row)
            if state_name in ("exp_avg", "exp_avg_sq") and value.shape != parameter.shape:
                malformed.append(row)
    state_shapes.sort(key=lambda row: (row["parameter"], row["state"]))
    return {
        "parameter_group_coverage_exact": coverage,
        "trainable_parameter_tensors": len(trainable),
        "grouped_parameter_tensors": len(grouped),
        "state_parameter_entries": len(optimizer.state),
        "state_tensor_count": len(state_shapes),
        "state_aggregate_sha256": canonical_sha(state_shapes),
        "malformed_state_tensors": malformed,
        "passed": coverage and not malformed and all(parameter in names for parameter in optimizer.state),
    }


def optimizer_group_manifest(model, optimizer):
    names = {parameter: name for name, parameter in model.named_parameters()}
    rows = []
    for index, group in enumerate(optimizer.param_groups):
        rows.append({
            "index": index,
            "parameter_names": [names.get(parameter, "<unknown>") for parameter in group["params"]],
            "options": {
                key: canonical_rng_value(value)
                for key, value in sorted(group.items()) if key != "params"
            },
        })
    return rows


def rebind_optimizer_by_parameter_name(source_model, evaluator_model, source_optimizer):
    """Clone optimizer groups/state onto an evaluator by exact parameter name."""
    source_named = dict(source_model.named_parameters())
    evaluator_named = dict(evaluator_model.named_parameters())
    source_by_parameter = {parameter: name for name, parameter in source_named.items()}
    source_group_names = []
    unknown_group_parameters = []
    rebound_groups = []
    for group in source_optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = source_by_parameter.get(parameter)
            if name is None:
                unknown_group_parameters.append(id(parameter))
                continue
            names.append(name)
        source_group_names.append(names)
        options = copy.deepcopy({key: value for key, value in group.items() if key != "params"})
        rebound_groups.append({
            **options,
            "params": [evaluator_named[name] for name in names if name in evaluator_named],
        })
    names_exact = source_named.keys() == evaluator_named.keys()
    shapes_exact = names_exact and all(
        source_named[name].shape == evaluator_named[name].shape
        and source_named[name].dtype == evaluator_named[name].dtype
        for name in source_named
    )
    grouped_source_names = [name for group in source_group_names for name in group]
    source_coverage_exact = (
        not unknown_group_parameters
        and len(grouped_source_names) == len(set(grouped_source_names))
        and set(grouped_source_names) == set(source_named)
    )
    if not names_exact or not shapes_exact or not source_coverage_exact:
        raise SystemExit(
            "cannot rebind evaluator optimizer: parameter name/shape/group coverage mismatch"
        )
    constructor_parameters = inspect.signature(
        type(source_optimizer).__init__
    ).parameters
    constructor_defaults = {
        key: copy.deepcopy(value)
        for key, value in source_optimizer.defaults.items()
        if key in constructor_parameters and key != "params"
    }
    rebound = type(source_optimizer)(rebound_groups, **constructor_defaults)
    # Preserve every source group field exactly, including scheduler-mutated LR
    # metadata, rather than relying on optimizer-constructor normalization.
    for target_group, source_group in zip(
        rebound.param_groups, source_optimizer.param_groups
    ):
        for key in tuple(target_group):
            if key != "params" and key not in source_group:
                del target_group[key]
        for key, value in source_group.items():
            if key != "params":
                target_group[key] = copy.deepcopy(value)
    rebound.defaults = copy.deepcopy(source_optimizer.defaults)

    unknown_state_parameters = []
    for source_parameter, state in source_optimizer.state.items():
        name = source_by_parameter.get(source_parameter)
        if name is None:
            unknown_state_parameters.append(id(source_parameter))
            continue
        rebound.state[evaluator_named[name]] = copy.deepcopy(state)
    if unknown_state_parameters:
        raise SystemExit("cannot rebind evaluator optimizer: unnamed state parameter")

    source_manifest = optimizer_manifest(source_model, source_optimizer)
    rebound_manifest = optimizer_manifest(evaluator_model, rebound)
    source_groups = optimizer_group_manifest(source_model, source_optimizer)
    rebound_group_rows = optimizer_group_manifest(evaluator_model, rebound)
    source_defaults = canonical_rng_value(source_optimizer.defaults)
    rebound_defaults = canonical_rng_value(rebound.defaults)
    source_state_names = {
        source_by_parameter[parameter] for parameter in source_optimizer.state
    }
    evaluator_by_parameter = {
        parameter: name for name, parameter in evaluator_named.items()
    }
    rebound_state_names = {
        evaluator_by_parameter[parameter] for parameter in rebound.state
    }
    state_values_exact = True
    state_storage_independent = True
    for name in source_state_names:
        source_state = source_optimizer.state[source_named[name]]
        target_state = rebound.state[evaluator_named[name]]
        if source_state.keys() != target_state.keys():
            state_values_exact = False
            continue
        for key in source_state:
            left, right = source_state[key], target_state[key]
            if torch.is_tensor(left) or torch.is_tensor(right):
                exact = (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and left.shape == right.shape
                    and left.dtype == right.dtype
                    and torch.equal(left, right)
                )
                state_values_exact = state_values_exact and exact
                if exact and left.numel():
                    state_storage_independent = (
                        state_storage_independent
                        and left.untyped_storage().data_ptr()
                        != right.untyped_storage().data_ptr()
                    )
            else:
                state_values_exact = state_values_exact and (
                    canonical_rng_value(left) == canonical_rng_value(right)
                )
    checks = {
        "parameter_names_exact": names_exact,
        "parameter_shapes_and_dtypes_exact": shapes_exact,
        "source_group_coverage_exact": source_coverage_exact,
        "rebound_group_coverage_exact": rebound_manifest[
            "parameter_group_coverage_exact"
        ],
        "group_parameter_names_and_metadata_exact": source_groups
        == rebound_group_rows,
        "optimizer_defaults_exact": source_defaults == rebound_defaults,
        "state_parameter_names_exact": source_state_names == rebound_state_names,
        "state_values_exact": state_values_exact,
        "state_tensor_storage_independent": state_storage_independent,
        "state_aggregate_sha256_exact": source_manifest["state_aggregate_sha256"]
        == rebound_manifest["state_aggregate_sha256"],
        "source_optimizer_manifest_passed": source_manifest["passed"],
        "rebound_optimizer_manifest_passed": rebound_manifest["passed"],
        "optimizer_type_exact": type(rebound) is type(source_optimizer),
    }
    evidence = {
        "method": "optimizer groups and all state cloned onto evaluator parameters by exact named-parameter mapping",
        "source_group_manifest": source_groups,
        "rebound_group_manifest": rebound_group_rows,
        "source_defaults": source_defaults,
        "rebound_defaults": rebound_defaults,
        "source_state_manifest": source_manifest,
        "rebound_state_manifest": rebound_manifest,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not evidence["passed"]:
        raise SystemExit(f"evaluator optimizer rebinding failed: {checks}")
    return rebound, evidence


def optimizer_rebinding_preflight_test(model, optimizer):
    model_before = parameter_manifest(model)["aggregate_sha256"]
    optimizer_before = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    rebound, evidence = rebind_optimizer_by_parameter_name(model, model, optimizer)
    model.zero_grad(set_to_none=True)
    model.g_rec_b3.grad = torch.ones_like(model.g_rec_b3)
    rebound.zero_grad(set_to_none=True)
    rebound_zero_grad_reaches_model = model.g_rec_b3.grad is None
    model.zero_grad(set_to_none=True)
    checks = {
        "exact_name_shape_group_and_state_rebinding": evidence["passed"],
        "rebound_zero_grad_reaches_executed_model": rebound_zero_grad_reaches_model,
        "model_unchanged": parameter_manifest(model)["aggregate_sha256"]
        == model_before,
        "source_optimizer_unchanged": optimizer_manifest(model, optimizer)[
            "state_aggregate_sha256"
        ] == optimizer_before,
    }
    result = {
        "rebinding_evidence": evidence,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del rebound
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def canonical_rng_value(value):
    """Convert RNG state into a storage-identity-independent JSON value."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "torch_tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy_array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(key): canonical_rng_value(current)
            for key, current in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_rng_value(current) for current in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported RNG-state value {type(value).__name__}")


def rng_digests(value):
    return {
        name: canonical_sha(canonical_rng_value(state))
        for name, state in sorted(value.items())
    }


def make_c_model(source_checkpoint, device, restore=False):
    fixed, optimizer, loader, payload, source_checks = d4a.load_fixed_source(
        source_checkpoint, device, restore=restore
    )
    model = core.FixedWriterW2PressureGPT(fixed.base).to(device)
    model.g_rec = fixed.g_rec
    model.g_rec_b3 = fixed.g_rec_b3
    model.g_rec_b5 = fixed.g_rec_b5
    model.g_rec_b6 = fixed.g_rec_b6
    fixed_named = dict(fixed.named_parameters())
    c_named = dict(model.named_parameters())
    checks = {
        **source_checks,
        "parameter_names_exact": fixed_named.keys() == c_named.keys(),
        "parameter_shapes_exact": all(
            fixed_named[name].shape == c_named[name].shape for name in fixed_named
        ),
        "parameter_identity_exact": all(
            fixed_named[name] is c_named[name] for name in fixed_named
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS,
        "optimizer_parameter_identity": {
            parameter for group in optimizer.param_groups for parameter in group["params"]
        } == set(model.parameters()),
        "architecture_fingerprint": model.architecture_fingerprint() == ARCHITECTURE_FINGERPRINT_C,
    }
    if not all(checks.values()):
        raise SystemExit(f"C construction failed: {checks}")
    return model, optimizer, loader, payload, checks


def make_fixed_evaluator_from_model(fixed):
    model = core.FixedControlEvaluationGPT(fixed.base).to(base.model_device(fixed))
    model.g_rec = fixed.g_rec
    model.g_rec_b3 = fixed.g_rec_b3
    model.g_rec_b5 = fixed.g_rec_b5
    model.g_rec_b6 = fixed.g_rec_b6
    model.load_state_dict(fixed.state_dict(), strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
        raise SystemExit("Fixed evaluator parameter count mismatch")
    if model.architecture_fingerprint() != ARCHITECTURE_FINGERPRINT_FIXED:
        raise SystemExit("Fixed evaluator architecture fingerprint mismatch")
    return model


def load_fixed_control(control_checkpoint, source_checkpoint, device):
    control_path = require_exact_file(control_checkpoint, CONTROL_SHA256, "Fixed-100M control")
    fixed, optimizer, loader, payload = d4a.load_arm_checkpoint(
        control_path, source_checkpoint, device, restore=False
    )
    if payload.get("arm") != "fixed":
        raise SystemExit("control checkpoint is not the 2D4A Fixed arm")
    checks = {
        "sha256": sha256(control_path) == CONTROL_SHA256,
        "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
        "local_updates": payload.get("d4a_local_updates") == LOCAL_UPDATES,
        "local_targets": payload.get("d4a_local_targets") == LOCAL_TARGETS,
        "global_update": payload.get("inherited_global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("inherited_total_targets") == FINAL_CUMULATIVE_TARGETS,
        "parameters": sum(parameter.numel() for parameter in fixed.parameters()) == PARAMETERS,
        "next_batch": payload.get("next_global_batch_sha256") == CONTROL_NEXT_BATCH,
        "next_stream": payload.get("next_global_batch_stream_sha256") == CONTROL_NEXT_STREAM,
        "strict_reopen": d4a.strict_reopen(control_path, source_checkpoint, device)["passed"],
        "no_router_parameters": not any(name.startswith("source_routers.") for name, _ in fixed.named_parameters()),
    }
    if not all(checks.values()):
        raise SystemExit(f"Fixed control provenance failed: {checks}")
    evaluator = make_fixed_evaluator_from_model(fixed)
    rebound_optimizer, optimizer_rebinding = rebind_optimizer_by_parameter_name(
        fixed, evaluator, optimizer
    )
    checks["optimizer_rebound_by_exact_parameter_name"] = optimizer_rebinding[
        "passed"
    ]
    del optimizer, fixed
    return (
        evaluator,
        rebound_optimizer,
        loader,
        payload,
        checks,
        optimizer_rebinding,
    )


def checkpoint_name(local_update):
    return f"scientific_cumulative_{MILESTONE_TARGETS[int(local_update)]:012d}.pt"


def verify_c_checkpoint_binding(path, local_update, milestone_manifest,
                                final_seal=None):
    """Bind a C checkpoint to its sealed sidecar and training milestone row."""
    path = Path(path).resolve()
    local_update = int(local_update)
    if local_update not in MILESTONES:
        raise SystemExit(f"C checkpoint binding requires a sealed milestone, got {local_update}")
    actual_sha = sha256(path)
    sidecar_path = path.with_suffix(path.suffix + ".verification.json")
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise SystemExit(f"missing C checkpoint verification sidecar: {sidecar_path}")
    if not checksum_path.is_file():
        raise SystemExit(f"missing C checkpoint checksum sidecar: {checksum_path}")
    sidecar = read_json(sidecar_path)
    milestones = read_json(milestone_manifest)
    milestone = milestones.get(str(local_update))
    if not isinstance(milestone, dict):
        raise SystemExit(f"milestone manifest has no sealed update {local_update}")
    checks = {
        "canonical_filename": path.name == checkpoint_name(local_update),
        "checkpoint_bytes": sidecar.get("bytes") == path.stat().st_size,
        "checksum_sidecar_exact": checksum_path.read_text()
        == f"{actual_sha}  {path.name}\n",
        "sidecar_sha256": sidecar.get("sha256") == actual_sha,
        "sidecar_local_update": sidecar.get("local_update") == local_update,
        "sidecar_global_update": sidecar.get("global_update") == SOURCE_UPDATES + local_update,
        "sidecar_cumulative_targets": sidecar.get("cumulative_targets") == MILESTONE_TARGETS[local_update],
        "sidecar_architecture": sidecar.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C,
        "sidecar_strict_reopen": sidecar.get("strict_reopen", {}).get("passed") is True,
        "milestone_sha256": milestone.get("sha256") == actual_sha,
        "milestone_exact_sidecar": milestone == sidecar,
        "milestone_path_name": Path(milestone.get("checkpoint", "")).name == path.name,
        "milestone_local_update": milestone.get("local_update") == local_update,
        "milestone_strict_reopen": milestone.get("strict_reopen", {}).get("passed") is True,
    }
    if final_seal is not None:
        seal = read_json(final_seal)
        checks.update({
            "final_seal": seal.get("sealed") is True,
            "final_seal_local_update": seal.get("local_update") == LOCAL_UPDATES == local_update,
            "final_seal_sha256": seal.get("checkpoint_sha256") == actual_sha,
            "final_seal_milestone_manifest_sha256":
            seal.get("artifact_identity", {}).get("milestone_manifest", {}).get("sha256")
            == sha256(milestone_manifest),
        })
    result = {
        "checkpoint": str(path),
        "local_update": local_update,
        "sha256": actual_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256(sidecar_path),
        "checksum_sidecar": str(checksum_path),
        "checksum_sidecar_sha256": sha256(checksum_path),
        "milestone_manifest": str(Path(milestone_manifest).resolve()),
        "milestone_manifest_sha256": sha256(milestone_manifest),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise SystemExit(f"C checkpoint artifact binding failed: {checks}")
    return result


def pass_count_for_local(local_update):
    return base.pass_count(SOURCE_UPDATES + int(local_update))


def optimizer_steps(optimizer):
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            values.append(int(step.item() if torch.is_tensor(step) else step))
    return values


def loader_from_state(state):
    return base.d1.ExplicitShardLoader(
        state["shards"], state["batch_size"], base.T, state=state
    )


def restart_sentinel(model):
    device = base.model_device(model)
    tokens = (torch.arange(128, device=device, dtype=torch.long) * 7919 + 17)
    tokens = tokens.remainder(50_257).view(1, 128)
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(tokens, num_passes=2)
    selected = result["logits"][0, (0, 31, 63, 127), :64].detach().float().cpu()
    if was_training:
        model.train()
    return {
        "input_sha256": tensor_sha256(tokens),
        "selected_logits_sha256": tensor_sha256(selected),
        "selected_shape": list(selected.shape),
        "selected_logits": selected.tolist(),
        "max_abs": selected.abs().max().item(),
        "finite": bool(torch.isfinite(selected).all()),
    }


def checkpoint_payload(model, optimizer, loader, source_payload, local_update,
                       accumulation, metadata, replay_ledger_sha):
    local_update = int(local_update)
    global_update = SOURCE_UPDATES + local_update
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA,
        "experiment_name": EXPERIMENT,
        "arm": "C",
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "control_checkpoint_sha256": CONTROL_SHA256,
        "source_global_update": SOURCE_UPDATES,
        "source_cumulative_targets": SOURCE_TARGETS,
        "local_updates": local_update,
        "new_targets": local_update * base.GLOBAL_TARGETS,
        "global_update": global_update,
        "cumulative_targets": SOURCE_TARGETS + local_update * base.GLOBAL_TARGETS,
        "targets_per_update": base.GLOBAL_TARGETS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(source_payload.get("scheduler")),
        "loader_state": loader.state_dict(),
        "loader_states": [loader.state_dict()],
        "rng_state": base.capture_rng(),
        "rng_digests": rng_digests(base.capture_rng()),
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": base.next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": base.next_stream_hash(loader, accumulation),
        "next_pass_count": pass_count_for_local(local_update + 1)
        if local_update < LOCAL_UPDATES else None,
        "architecture_manifest": ARCHITECTURE_C,
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
        "fixed_writers": {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"},
        "local_windows": {"B1": 2, "B2": 1024, "B3": 2, "B4": 1024,
                          "B5": 2, "B6": 512, **{f"B{i}": 1024 for i in range(7, 13)}},
        "recurrent_lag_ranges": {"B1": [2, 1023], "B3": [2, 1023],
                                 "B5": [2, 1023], "B6": [512, 1023]},
        "parameter_count": PARAMETERS,
        "replay_ledger_sha256": replay_ledger_sha,
        "optimizer_group_definitions": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "existing_recurrent_gates": base.gate_values(model),
        "restart_sentinel": restart_sentinel(model),
        "git_implementation_commit": metadata["git_implementation_commit"],
        "metadata": metadata,
        "saved_process_id": os.getpid(),
        "saved_at_unix": time.time(),
    }


def load_c_checkpoint(path, source_checkpoint, device, restore=False):
    payload = base.d0.torch_load(Path(path), mmap=False)
    if payload.get("schema") != SCHEMA or payload.get("arm") != "C":
        raise SystemExit("not a 2D5C C checkpoint")
    model, optimizer, _, source, _ = make_c_model(
        source_checkpoint, device, restore=False
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = loader_from_state(payload["loader_state"])
    if restore:
        base.restore_rng(payload["rng_state"])
    return model, optimizer, loader, payload, source


def strict_reopen(path, source_checkpoint, device):
    # Strict reopening is an observational audit.  Model construction consumes
    # RNG, so preserve the live continuation state around the entire reopen.
    entry_rng = base.capture_rng()
    model = optimizer = loader = payload = None
    try:
        model, optimizer, loader, payload, source = load_c_checkpoint(
            path, source_checkpoint, device, restore=False
        )
        local_update = int(payload["local_updates"])
        accumulation = int(payload["gradient_accumulation"])
        sentinel = restart_sentinel(model)
        expected_sentinel = payload.get("restart_sentinel", {})
        left = np.asarray(expected_sentinel.get("selected_logits", []), dtype=np.float64)
        right = np.asarray(sentinel.get("selected_logits", []), dtype=np.float64)
        max_abs = float(np.max(np.abs(left - right))) if left.shape == right.shape and left.size else math.inf
        checks = {
            "schema": payload.get("schema") == SCHEMA,
            "arm": payload.get("arm") == "C",
            "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
            "control_provenance_only": payload.get("control_checkpoint_sha256") == CONTROL_SHA256,
            "local_update_range": 0 <= local_update <= LOCAL_UPDATES,
            "new_targets": payload.get("new_targets") == local_update * base.GLOBAL_TARGETS,
            "global_update": payload.get("global_update") == SOURCE_UPDATES + local_update,
            "cumulative_targets": payload.get("cumulative_targets") == SOURCE_TARGETS + local_update * base.GLOBAL_TARGETS,
            "parameters": sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS,
            "architecture": payload.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C,
            "model_finite": base.model_finite(model),
            "optimizer_finite": base.optimizer_finite(optimizer),
            "optimizer_coverage": optimizer_manifest(model, optimizer)["passed"],
            "next_batch": base.next_batch_hash(loader, accumulation) == payload.get("next_global_batch_sha256"),
            "next_stream": base.next_stream_hash(loader, accumulation) == payload.get("next_global_batch_stream_sha256"),
            "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
            "rng_digests": payload.get("rng_digests") == rng_digests(payload.get("rng_state", {})),
            "scheduler_key_preserved": "scheduler" in payload,
            "scheduler_exact_source": payload.get("scheduler") == source.get("scheduler"),
            "sentinel_finite": sentinel["finite"],
            "sentinel_tolerance": max_abs <= 1e-6,
            "sentinel_exact_digest": sentinel["selected_logits_sha256"] == expected_sentinel.get("selected_logits_sha256"),
            "local_limit": local_update <= LOCAL_UPDATES,
        }
        return {
            "checkpoint": str(Path(path).resolve()),
            "local_update": local_update,
            "checks": checks,
            "sentinel_max_abs": max_abs,
            "rng_observational_audit": True,
            "passed": all(checks.values()),
        }
    finally:
        del model, optimizer, loader, payload
        gc.collect()
        torch.cuda.empty_cache()
        base.restore_rng(entry_rng)


def fsync_path(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(path, model, optimizer, loader, source_payload, local_update,
                    accumulation, metadata, replay_ledger_sha, source_checkpoint,
                    device, sidecars=True):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model, optimizer, loader, source_payload, local_update, accumulation,
        metadata, replay_ledger_sha,
    )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    fsync_path(temporary)
    os.replace(temporary, path)
    fsync_path(path.parent)
    digest = sha256(path)
    audit = strict_reopen(path, source_checkpoint, device)
    if not audit["passed"]:
        raise SystemExit(f"strict 2D5C checkpoint reopen failed: {audit}")
    verification = {
        "checkpoint": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "local_update": int(local_update),
        "global_update": SOURCE_UPDATES + int(local_update),
        "cumulative_targets": SOURCE_TARGETS + int(local_update) * base.GLOBAL_TARGETS,
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
        "optimizer_step_summary": sorted(set(optimizer_steps(optimizer))),
        "optimizer_state_sha256": optimizer_manifest(model, optimizer)["state_aggregate_sha256"],
        "model_state_sha256": parameter_manifest(model)["aggregate_sha256"],
        "scheduler": copy.deepcopy(payload.get("scheduler")),
        "scheduler_sha256": canonical_sha(payload.get("scheduler")),
        "rng_digests": payload["rng_digests"],
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
        "restart_sentinel": payload["restart_sentinel"],
        "saved_process_id": payload["saved_process_id"],
        "strict_reopen": audit,
    }
    if sidecars:
        durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
        durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def continuation_metadata(source_payload, panel_manifest, replay_audit):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "branch": BRANCH,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "code_tooling_lineage": "2D4A finalized tooling at 81bd84ff, without router model semantics",
        "model_weight_lineage": SOURCE_SHA256,
        "control_weight_lineage": CONTROL_SHA256,
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
        "targets_per_update": base.GLOBAL_TARGETS,
        "micro_batch": int(source_payload["loader_state"]["batch_size"]),
        "gradient_accumulation": int(source_payload["gradient_accumulation"]),
        "sequence_length": base.T,
        "pass3_every_global_updates": 32,
        "mandatory_restart_local_update": RESTART_LOCAL_UPDATE,
        "replay_ledger_sha256": replay_audit["ledger_sha256"],
        "large_panel_sha256": panel_manifest["panel_sha256"],
        "scheduler": copy.deepcopy(source_payload.get("scheduler")),
        "scheduler_reconciliation": "accepted checkpoint preserves a scheduler key whose constant-LR state is null; no scheduler was invented, reset, or stepped",
        "optimizer_resets": 0,
        "loader_resets": 0,
        "rng_resets": 0,
        "fixed_control_optimizer_steps": 0,
    }


def unique_same_size_sha_matches(root, reference):
    reference = Path(reference).resolve()
    size = reference.stat().st_size
    candidates = []
    for path in Path(root).rglob("*.pt"):
        try:
            if path.stat().st_size == size:
                candidates.append(path.resolve())
        except FileNotFoundError:
            continue
    rows = [
        {"path": str(path), "bytes": size, "sha256": sha256(path)}
        for path in sorted(candidates)
    ]
    matches = [row for row in rows if row["sha256"] == CONTROL_SHA256]
    return {"same_size_candidates": rows, "sha_matches": matches, "unique": len(matches) == 1}


def write_replay_ledger(loader, accumulation, expected_manifest, control_loader,
                        output):
    replay = loader.clone()
    expected = read_json(expected_manifest)
    expected_batches = expected["logical_global_batches"]
    expected_streams = expected["logical_global_streams"]
    if len(expected_batches) != LOCAL_UPDATES or len(expected_streams) != LOCAL_UPDATES:
        raise SystemExit("existing Fixed replay manifest is not exactly 191 rows")
    rows = []
    previous_chain = "00" * 32
    for local_update in range(1, LOCAL_UPDATES + 1):
        start_cursor = replay.state_dict()
        logical_batch = base.next_batch_hash(replay, accumulation)
        logical_stream = base.next_stream_hash(replay, accumulation)
        microbatches = []
        for micro_index in range(int(accumulation)):
            call_cursor = replay.state_dict()
            x, y = replay.next_batch()
            end_cursor = replay.state_dict()
            actual_start = end_cursor["current_position"] - x.numel()
            microbatches.append({
                "microbatch_index": micro_index,
                "source_shard_index": end_cursor["current_shard"],
                "source_shard": end_cursor["shards"][end_cursor["current_shard"]],
                "start_position": actual_start,
                "end_position": end_cursor["current_position"],
                "cursor_before_call": call_cursor,
                "cursor_after": end_cursor,
                "input_sha256": tensor_sha256(x),
                "target_sha256": tensor_sha256(y),
                "combined_sha256": tensor_sha256(x, y),
                "payload_sha256": base.d0.batch_payload_hash(x, y),
                "sequences": int(x.size(0)),
                "targets": int(y.numel()),
            })
        row = {
            "experiment": EXPERIMENT,
            "arm": "C",
            "local_update": local_update,
            "global_update": SOURCE_UPDATES + local_update,
            "start_cursor": start_cursor,
            "end_cursor": replay.state_dict(),
            "logical_global_batch_sha256": logical_batch,
            "logical_global_stream_sha256": logical_stream,
            "microbatch_membership_order": microbatches,
            "microbatch_count": int(accumulation),
            "target_count": base.GLOBAL_TARGETS,
            "pass_count": pass_count_for_local(local_update),
            "new_cumulative_targets": local_update * base.GLOBAL_TARGETS,
            "lineage_cumulative_targets": SOURCE_TARGETS + local_update * base.GLOBAL_TARGETS,
        }
        row["previous_chain_sha256"] = previous_chain
        row["chain_sha256"] = artifacts.rolling_replay_chain_sha256(
            previous_chain, row
        )
        previous_chain = row["chain_sha256"]
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        append_jsonl(output, row)
    ledger_sha = sha256(output)
    generated_batches = [row["logical_global_batch_sha256"] for row in rows]
    generated_streams = [row["logical_global_stream_sha256"] for row in rows]
    terminal_batch = base.next_batch_hash(replay, accumulation)
    terminal_stream = base.next_stream_hash(replay, accumulation)
    control_state = control_loader.state_dict()
    checks = {
        "rows_exact": len(rows) == LOCAL_UPDATES,
        "local_updates_exact": [row["local_update"] for row in rows] == list(range(1, 192)),
        "global_updates_exact": [row["global_update"] for row in rows] == list(range(1909, 2100)),
        "first_batch": rows[0]["logical_global_batch_sha256"] == SOURCE_NEXT_BATCH,
        "first_stream": rows[0]["logical_global_stream_sha256"] == SOURCE_NEXT_STREAM,
        "all_fixed_batch_hashes": generated_batches == expected_batches,
        "all_fixed_stream_hashes": generated_streams == expected_streams,
        "terminal_cursor": replay.state_dict() == control_state,
        "terminal_next_batch": terminal_batch == CONTROL_NEXT_BATCH,
        "terminal_next_stream": terminal_stream == CONTROL_NEXT_STREAM,
        "targets_per_row": all(row["target_count"] == base.GLOBAL_TARGETS for row in rows),
        "total_targets": sum(row["target_count"] for row in rows) == LOCAL_TARGETS,
        "microbatch_order": all(
            [micro["microbatch_index"] for micro in row["microbatch_membership_order"]]
            == list(range(int(accumulation))) for row in rows
        ),
        "pass_cadence": [row["pass_count"] for row in rows]
        == [pass_count_for_local(update) for update in range(1, 192)],
        "chain_complete": rows[-1]["chain_sha256"] == previous_chain,
    }
    audit = {
        "canonical_chain_definition": "experiment_2d5c_artifacts.rolling_replay_chain_sha256 using domain experiment-2d5c/replay-ledger/v1, previous raw digest, explicit payload length, and canonical JSON; genesis is 32 zero bytes",
        "ledger_path": str(output.resolve()),
        "ledger_sha256": ledger_sha,
        "terminal_chain_sha256": previous_chain,
        "rows": len(rows),
        "initial_cursor": rows[0]["start_cursor"],
        "terminal_cursor": replay.state_dict(),
        "terminal_next_batch_sha256": terminal_batch,
        "terminal_next_stream_sha256": terminal_stream,
        "three_pass_local_updates": [row["local_update"] for row in rows if row["pass_count"] == 3],
        "two_pass_updates": sum(row["pass_count"] == 2 for row in rows),
        "three_pass_updates": sum(row["pass_count"] == 3 for row in rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not audit["passed"]:
        raise SystemExit(f"exact replay reconstruction failed: {checks}")
    return rows, audit


def batch_at_index(val_path, batch_index, batch_size=base.VALIDATION_B):
    count = int(batch_size) * base.T
    state = {
        "shards": [str(Path(val_path).resolve())],
        "batch_size": int(batch_size),
        "sequence_length": base.T,
        "current_shard": 0,
        "current_position": int(batch_index) * count,
    }
    loader = base.d1.ExplicitShardLoader([val_path], int(batch_size), base.T, state=state)
    return loader.next_batch()


def sequence_rows(x, y, batch_index):
    rows = []
    count = x.size(0) * base.T
    for sequence_index in range(x.size(0)):
        start = int(batch_index) * count + sequence_index * base.T
        rows.append({
            "batch_index": int(batch_index),
            "sequence_index": sequence_index,
            "dataset_input_span": [start, start + base.T - 1],
            "dataset_target_span": [start + 1, start + base.T],
            "input_sha256": tensor_sha256(x[sequence_index]),
            "target_sha256": tensor_sha256(y[sequence_index]),
            "combined_sha256": tensor_sha256(x[sequence_index], y[sequence_index]),
            "targets": base.T,
        })
    return rows


def canonical_batch_target_span(batch_index, batch_size, sequence_length):
    """Return the exact half-open scored-target span for a canonical batch."""
    targets_per_batch = int(batch_size) * int(sequence_length)
    start = int(batch_index) * targets_per_batch + 1
    return [start, start + targets_per_batch]


def half_open_span_intersections(left, right):
    intersections = []
    for left_span in left:
        for right_span in right:
            start = max(int(left_span[0]), int(right_span[0]))
            end = min(int(left_span[1]), int(right_span[1]))
            if start < end:
                intersections.append({
                    "left": [int(left_span[0]), int(left_span[1])],
                    "right": [int(right_span[0]), int(right_span[1])],
                    "intersection": [start, end],
                })
    return intersections


def historical_panel_rows(paths, val_path):
    """Bind historical batch manifests to this exact canonical validation data.

    Historical 2D3A/2D4A manifests expose batch identities, not per-sequence
    identities.  Replaying every claimed batch at its claimed canonical index
    establishes an exact data/index binding without pretending that unavailable
    historical per-sequence hashes were recorded.
    """
    val_path = Path(val_path).resolve()
    current_dataset_sha256 = sha256(val_path)
    rows = []
    for label, path in paths:
        value = read_json(path)
        start = value.get("start_batch")
        batches = value.get("batches")
        batch_size = value.get("batch_size")
        sequence_length = value.get("sequence_length")
        identities = value.get("batch_identities")
        indices = (
            []
            if start is None or batches is None
            else list(range(int(start), int(start) + int(batches)))
        )
        schema_checks = {
            "start_batch_nonnegative_integer": isinstance(start, int) and start >= 0,
            "positive_integer_batch_count": isinstance(batches, int) and batches > 0,
            "canonical_batch_size": int(batch_size or -1) == int(base.VALIDATION_B),
            "canonical_sequence_length": int(sequence_length or -1) == int(base.T),
            "batch_identity_count_exact": isinstance(identities, list)
            and len(identities) == len(indices),
            "batch_identities_have_complete_hashes": isinstance(identities, list)
            and all(
                isinstance(row, dict)
                and all(
                    isinstance(row.get(name), str) and len(row[name]) == 64
                    for name in ("input_sha256", "target_sha256", "combined_sha256")
                )
                for row in identities
            ),
        }
        replayed_identities = []
        if all(schema_checks.values()):
            replayed_identities = [
                base.batch_identity(*batch_at_index(val_path, index))
                for index in indices
            ]
        replay_exact_by_index = bool(replayed_identities) and all(
            all(observed[name] == expected[name] for name in (
                "input_sha256", "target_sha256", "combined_sha256"
            ))
            for observed, expected in zip(replayed_identities, identities)
        )
        manifest_subset_recomputed = (
            None
            if not schema_checks["batch_identity_count_exact"]
            else base.aggregate_hashes(row["combined_sha256"] for row in identities)
        )
        claimed_dataset_sha256 = value.get("dataset_sha256")
        claimed_sha_compatible = (
            claimed_dataset_sha256 is None
            or claimed_dataset_sha256 == current_dataset_sha256
        )
        sequence_identities = value.get("sequence_identities")
        sequence_history_available = isinstance(sequence_identities, list)
        canonical_spans = [
            canonical_batch_target_span(index, batch_size, sequence_length)
            for index in indices
        ] if all(schema_checks.values()) else []
        dataset_identity_verified = (
            all(schema_checks.values())
            and replay_exact_by_index
            and manifest_subset_recomputed == value.get("subset_sha256")
            and claimed_sha_compatible
        )
        rows.append({
            "label": label,
            "manifest": str(Path(path).resolve()),
            "start_batch": start,
            "batches": batches,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "batch_indices": indices,
            "subset_sha256": value["subset_sha256"],
            "batch_identities": identities,
            "canonical_target_spans_half_open": canonical_spans,
            "dataset_identity": {
                "historical_validation_shard": value.get("validation_shard")
                or value.get("dataset"),
                "current_validation_shard": str(val_path),
                "historical_dataset_sha256": claimed_dataset_sha256,
                "current_dataset_sha256": current_dataset_sha256,
                "method": "replay every claimed canonical batch index on the current validation shard and require exact input/target/combined SHA-256 identity",
                "same_index_batch_identity_replay_exact": replay_exact_by_index,
                "manifest_subset_recomputed_sha256": manifest_subset_recomputed,
                "claimed_dataset_sha_compatible": claimed_sha_compatible,
                "verified": dataset_identity_verified,
            },
            "schema_checks": schema_checks,
            "per_sequence_history": {
                "available_in_historical_manifest": sequence_history_available,
                "scope": (
                    "exact supplied sequence_identities"
                    if sequence_history_available
                    else "unavailable; no historical per-sequence disjointness claim is made"
                ),
                "sequence_identities": sequence_identities,
            },
        })
    return rows


def prepare_panels(val_path, training_shards, historical_paths, output):
    val_path = Path(val_path).resolve()
    historical = historical_panel_rows(historical_paths, val_path)
    historical_identity_failures = {
        panel["label"]: {
            "schema_checks": panel["schema_checks"],
            "dataset_identity": panel["dataset_identity"],
        }
        for panel in historical
        if not panel["dataset_identity"]["verified"]
        or not all(panel["schema_checks"].values())
    }
    if historical_identity_failures:
        raise SystemExit(
            "historical panels are not exactly bound to the canonical validation "
            f"dataset and indices: {historical_identity_failures}"
        )
    core_xys = [batch_at_index(val_path, index) for index in range(4)]
    core_identities = [base.batch_identity(x, y) for x, y in core_xys]
    core_sha = base.aggregate_hashes(row["combined_sha256"] for row in core_identities)
    if core_sha != CORE_SHA256:
        raise SystemExit(f"immutable mechanism core SHA mismatch: {core_sha}")
    all_core_sequences = [
        row for batch_index, (x, y) in enumerate(core_xys)
        for row in sequence_rows(x, y, batch_index)
    ]
    diagnostic_seed = int.from_bytes(
        hashlib.sha256(f"{LARGE_SELECTION_SEED}:diagnostic-core".encode()).digest()[:8],
        "big",
    )
    diagnostic_rng = np.random.default_rng(diagnostic_seed)
    selected_sequence_indices = diagnostic_rng.choice(256, size=32, replace=False).tolist()
    diagnostic_rows = [all_core_sequences[index] for index in selected_sequence_indices]

    token_count = int(np.load(val_path, mmap_mode="r").shape[0])
    tokens_per_batch = base.VALIDATION_B * base.T
    available_batches = (token_count - 1) // tokens_per_batch
    forbidden_indices = set(range(4))
    forbidden_hashes = {row["combined_sha256"] for row in core_identities}
    for panel in historical:
        forbidden_indices.update(panel["batch_indices"])
        forbidden_hashes.update(row["combined_sha256"] for row in panel["batch_identities"])
    candidates = np.arange(available_batches, dtype=np.int64)
    rng = np.random.default_rng(LARGE_SELECTION_SEED)
    ordered = rng.permutation(candidates).tolist()
    chosen = []
    batch_identities = []
    sequences = []
    for batch_index in ordered:
        if int(batch_index) in forbidden_indices:
            continue
        x, y = batch_at_index(val_path, int(batch_index))
        identity = base.batch_identity(x, y)
        if identity["combined_sha256"] in forbidden_hashes:
            continue
        chosen.append(int(batch_index))
        batch_identities.append(identity)
        sequences.extend(sequence_rows(x, y, int(batch_index)))
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise SystemExit("could not select a 32-batch fresh large panel")
    panel_sha = base.aggregate_hashes(row["combined_sha256"] for row in batch_identities)
    sequence_hashes = {row["combined_sha256"] for row in sequences}
    chosen_target_spans = [
        canonical_batch_target_span(index, base.VALIDATION_B, base.T)
        for index in chosen
    ]
    disjointness = {}
    for panel in historical:
        overlap_batches = sorted(
            {row["combined_sha256"] for row in panel["batch_identities"]}
            & {row["combined_sha256"] for row in batch_identities}
        )
        overlap_indices = sorted(set(chosen) & set(panel["batch_indices"]))
        span_intersections = half_open_span_intersections(
            chosen_target_spans, panel["canonical_target_spans_half_open"]
        )
        historical_sequence_rows = panel["per_sequence_history"]["sequence_identities"]
        historical_sequence_hashes = (
            None
            if historical_sequence_rows is None
            else {
                row["combined_sha256"] for row in historical_sequence_rows
                if isinstance(row, dict) and "combined_sha256" in row
            }
        )
        sequence_intersection = (
            None
            if historical_sequence_hashes is None
            else sorted(sequence_hashes & historical_sequence_hashes)
        )
        verified = (
            panel["dataset_identity"]["verified"]
            and all(panel["schema_checks"].values())
            and not overlap_batches
            and not overlap_indices
            and not span_intersections
            and (sequence_intersection is None or not sequence_intersection)
        )
        disjointness[panel["label"]] = {
            "historical_subset_sha256": panel["subset_sha256"],
            "historical_manifest": panel["manifest"],
            "dataset_identity": panel["dataset_identity"],
            "historical_schema_checks": panel["schema_checks"],
            "batch_hash_intersection": overlap_batches,
            "batch_index_intersection": overlap_indices,
            "canonical_target_span_definition": "half-open [batch_index * (batch_size * sequence_length) + 1, next_batch_index * (batch_size * sequence_length) + 1)",
            "canonical_target_span_intersections": span_intersections,
            "canonical_span_nonoverlap_verified": not span_intersections,
            "per_sequence_history_available": historical_sequence_hashes is not None,
            "sequence_hash_intersection": sequence_intersection,
            "per_sequence_claim_scope": panel["per_sequence_history"]["scope"],
            "verification_scope": "exact same-dataset canonical batch spans and batch identities; sequence hashes only when supplied by the historical manifest",
            "verified": verified,
        }
    core_sequence_hashes = {row["combined_sha256"] for row in all_core_sequences}
    training_resolved = {str(Path(path).resolve()) for path in training_shards}
    split_disjoint = str(val_path) not in training_resolved
    large_manifest = {
        "experiment": EXPERIMENT,
        "frozen_before_training": True,
        "selection_seed": LARGE_SELECTION_SEED,
        "selection_algorithm": "numpy.default_rng(seed).permutation(all complete canonical validation batches), reject frozen historical/core batch indices and hashes, take first 32; historical indices are accepted only after exact same-index batch-identity replay on this dataset",
        "dataset": str(val_path),
        "dataset_sha256": sha256(val_path),
        "dataset_split": "validation",
        "token_count": token_count,
        "batch_indices_in_evaluation_order": chosen,
        "canonical_target_spans_half_open": chosen_target_spans,
        "batch_identities": batch_identities,
        "sequence_identities": sequences,
        "sequence_count": len(sequences),
        "targets_per_sequence": base.T,
        "targets_per_condition": len(sequences) * base.T,
        "panel_sha256": panel_sha,
        "training_shards": sorted(training_resolved),
        "disjointness": {
            "training_split": {"verified": split_disjoint, "method": "canonical validation shard path is absent from accepted training loader shards"},
            "immutable_core": {
                "verified": not (sequence_hashes & core_sequence_hashes),
                "sequence_hash_intersection": sorted(sequence_hashes & core_sequence_hashes),
            },
            **disjointness,
        },
    }
    large_manifest["all_required_disjointness_passed"] = all(
        row["verified"] for row in large_manifest["disjointness"].values()
    )
    core_manifest = {
        "experiment": EXPERIMENT,
        "dataset": str(val_path),
        "dataset_sha256": sha256(val_path),
        "dataset_split": "validation",
        "batch_indices_in_evaluation_order": list(range(4)),
        "batch_identities": core_identities,
        "subset_sha256": core_sha,
        "sequences": 256,
        "targets_per_condition": 262_144,
        "diagnostic_selection_seed": diagnostic_seed,
        "diagnostic_selection_indices": selected_sequence_indices,
        "diagnostic_sequence_identities": diagnostic_rows,
        "diagnostic_subset_sha256": artifacts.canonical_json_sha256(diagnostic_rows),
    }
    shuffle = {
        "seed": SHUFFLE_SEED,
        "definition": "one shared path-consistent seeded nonzero cyclic donor roll per batch size",
        "permutations": {},
    }
    for batch_size in (32, 64):
        permutation_record = artifacts.seeded_cyclic_derangement(
            batch_size, SHUFFLE_SEED, domain=f"2d5c-shuffle-batch-{batch_size}"
        )
        permutation = permutation_record["donor_permutation"]
        shuffle["permutations"][str(batch_size)] = {
            "values": permutation,
            "sha256": artifacts.canonical_json_sha256(permutation),
            "algorithm_manifest": permutation_record,
            "fixed_points": [index for index, donor in enumerate(permutation) if index == donor],
        }
    shuffle["passed"] = all(
        not row["fixed_points"] and sorted(row["values"]) == list(range(int(size)))
        for size, row in shuffle["permutations"].items()
    )
    if not large_manifest["all_required_disjointness_passed"] or not shuffle["passed"]:
        raise SystemExit("panel disjointness or shuffle freeze failed")
    durable_json(Path(output) / "PANEL_MANIFEST_CORE.json", core_manifest)
    durable_json(Path(output) / "PANEL_MANIFEST_LARGE.json", large_manifest)
    durable_json(Path(output) / "SHUFFLE_MANIFEST.json", shuffle)
    return core_manifest, large_manifest, shuffle


def run_prepare(args):
    require_branch(clean=True)
    anchor_checks = {
        "source_tag_commit": git("rev-parse", f"{SOURCE_TAG}^{{commit}}")
        == SOURCE_COMMIT,
        "tooling_origin_commit": git(
            "rev-parse", "origin/experiment-2d4a-matched-source-depth-routing-250m"
        ) == TOOLING_COMMIT,
        "d4a_final_tag_commit": git("rev-parse", f"{D4A_FINAL_TAG}^{{commit}}")
        == D4A_FINAL_TAG_COMMIT,
        "tooling_is_ancestor": git_is_ancestor(TOOLING_COMMIT),
    }
    if not all(anchor_checks.values()):
        raise SystemExit(f"repository anchors are not reconciled: {anchor_checks}")
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            "prepare output is not empty; preserve/quarantine it and use a fresh directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    source_path = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "2D3A-1B source")
    source_stat = source_path.stat()
    model, optimizer, loader, source, source_checks = make_c_model(
        source_path, device, restore=False
    )
    source_reopen = parent.strict_reopen(source_path, SOURCE_UPDATES, source["metadata"], device)
    source_model_manifest = parameter_manifest(model)
    source_optimizer_manifest = optimizer_manifest(model, optimizer)
    (
        control,
        control_optimizer,
        control_loader,
        control_payload,
        control_checks,
        control_optimizer_rebinding,
    ) = load_fixed_control(args.control_checkpoint, source_path, device)
    unique_control = unique_same_size_sha_matches(args.control_search_root, args.control_checkpoint)
    control_checks["unique_sha_match"] = unique_control["unique"]
    if not all(control_checks.values()):
        raise SystemExit(f"control checks failed: {control_checks}")

    ledger_path = output / "DATA_REPLAY_LEDGER.jsonl"
    _, replay_audit = write_replay_ledger(
        loader, int(source["gradient_accumulation"]), args.fixed_replay_manifest,
        control_loader, ledger_path,
    )
    durable_json(output / "DATA_REPLAY_AUDIT.json", replay_audit)
    val_path = base.validation_path(Path(args.data_root))
    historical_paths = (
        ("final_2d3a_1b_large", args.d3a_large_manifest),
        ("2d4a_100m_large", args.d4a_large_manifest),
        ("2d4a_250m_large", args.d4a250_large_manifest),
    )
    core_manifest, large_manifest, shuffle_manifest = prepare_panels(
        val_path, source["loader_state"]["shards"], historical_paths, output
    )
    scope_lock = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "exactly_one_newly_trained_arm": ["C"],
        "training_source_sha256": SOURCE_SHA256,
        "control_checkpoint_sha256": CONTROL_SHA256,
        "local_update_limit": LOCAL_UPDATES,
        "target_limit": LOCAL_TARGETS,
        "A_prohibited": True,
        "B_prohibited": True,
        "continuation_beyond_100m_prohibited": True,
        "fixed_control_optimizer_steps": 0,
        "routed_source_prohibited": True,
        "entrypoint_enforces_limit": True,
    }
    source_provenance = {
        "checkpoint": str(source_path),
        "sha256": sha256(source_path),
        "bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "canonical_path": str(source_path),
        "schema": source.get("schema"),
        "global_update": source.get("d3a_completed_updates"),
        "cumulative_targets": source.get("d3a_processed_targets"),
        "next_batch_sha256": source.get("next_global_batch_sha256"),
        "next_stream_sha256": source.get("next_global_batch_stream_sha256"),
        "source_checks": source_checks,
        "strict_reopen": source_reopen,
        "model_state_manifest": source_model_manifest,
        "optimizer_state_manifest": source_optimizer_manifest,
        "rng_digests": rng_digests(source["rng_state"]),
        "scheduler_key_present": "scheduler" in source,
        "scheduler_state": source.get("scheduler"),
        "loader_state": source["loader_state"],
        "passed": all(source_checks.values()) and source_reopen["passed"]
        and source_optimizer_manifest["passed"],
    }
    control_provenance = {
        "checkpoint": str(Path(args.control_checkpoint).resolve()),
        "sha256": sha256(args.control_checkpoint),
        "bytes": Path(args.control_checkpoint).stat().st_size,
        "checks": control_checks,
        "unique_search": unique_control,
        "metadata": {
            "parent_sha256": control_payload.get("parent_checkpoint_sha256"),
            "local_updates": control_payload.get("d4a_local_updates"),
            "new_targets": control_payload.get("d4a_local_targets"),
            "global_update": control_payload.get("inherited_global_update"),
            "cumulative_targets": control_payload.get("inherited_total_targets"),
            "architecture": control_payload.get("architecture_manifest"),
        },
        "optimizer_steps_in_2d5c": 0,
        "optimizer_rebinding": control_optimizer_rebinding,
        "passed": all(control_checks.values()),
    }
    environment = {
        "experiment": EXPERIMENT,
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "source_tag_object": git("rev-parse", SOURCE_TAG),
        "source_tag_commit": git("rev-parse", f"{SOURCE_TAG}^{{commit}}"),
        "origin_branch": git("rev-parse", "origin/experiment-2d4a-matched-source-depth-routing-250m"),
        "d4a_final_tag": D4A_FINAL_TAG,
        "d4a_final_tag_commit": git("rev-parse", f"{D4A_FINAL_TAG}^{{commit}}"),
        "anchor_checks": anchor_checks,
        "implementation_file_sha256": implementation_file_sha256(),
        "repository_status_before_prepare": "clean",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "bf16": True,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "volume_id": args.volume_id,
    }
    durable_json(output / "SCOPE_LOCK.json", scope_lock)
    durable_json(output / "SOURCE_PROVENANCE.json", source_provenance)
    durable_json(output / "FIXED_CONTROL_PROVENANCE.json", control_provenance)
    durable_json(output / "ENVIRONMENT_MANIFEST.json", environment)
    durable_json(output / "ARCHITECTURE_MANIFEST_C.json", ARCHITECTURE_C)
    durable_json(output / "ARCHITECTURE_MANIFEST_FIXED.json", ARCHITECTURE_FIXED)
    prepare_audit = {
        "scope_lock": all(value for key, value in scope_lock.items() if key.endswith("prohibited")),
        "source": source_provenance["passed"],
        "control": control_provenance["passed"],
        "replay": replay_audit["passed"],
        "core": core_manifest["subset_sha256"] == CORE_SHA256,
        "large_panel": large_manifest["all_required_disjointness_passed"],
        "shuffle": shuffle_manifest["passed"],
        "one_a100_80gb": torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(device),
    }
    prepare_audit["passed"] = all(prepare_audit.values())
    prepare_audit["artifact_sha256"] = {
        "data_replay_ledger": sha256(output / "DATA_REPLAY_LEDGER.jsonl"),
        "data_replay_audit": sha256(output / "DATA_REPLAY_AUDIT.json"),
        "core_panel_manifest": sha256(output / "PANEL_MANIFEST_CORE.json"),
        "large_panel_manifest": sha256(output / "PANEL_MANIFEST_LARGE.json"),
        "shuffle_manifest": sha256(output / "SHUFFLE_MANIFEST.json"),
        "scope_lock": sha256(output / "SCOPE_LOCK.json"),
        "source_provenance": sha256(output / "SOURCE_PROVENANCE.json"),
        "fixed_control_provenance": sha256(output / "FIXED_CONTROL_PROVENANCE.json"),
    }
    durable_json(output / "PRETRAIN_FREEZE_AUDIT.json", prepare_audit)
    if not prepare_audit["passed"]:
        raise SystemExit(f"2D5C prepare failed: {prepare_audit}")
    del model, optimizer, loader, control, control_optimizer, control_loader
    gc.collect()
    torch.cuda.empty_cache()
    print("EXPERIMENT_2D5C_PRETRAIN_FREEZE_READY", flush=True)


def panel_batch_indices(manifest):
    if "batch_indices_in_evaluation_order" in manifest:
        return [int(value) for value in manifest["batch_indices_in_evaluation_order"]]
    start = int(manifest.get("start_batch", 0))
    batches = int(manifest.get("batches", 4))
    return list(range(start, start + batches))


def permutation_for_batch(shuffle_manifest, batch_size, device):
    row = shuffle_manifest["permutations"][str(int(batch_size))]
    values = row["values"]
    if sorted(values) != list(range(int(batch_size))) or any(
        index == donor for index, donor in enumerate(values)
    ):
        raise SystemExit("frozen shuffle is not a fixed-point-free permutation")
    return torch.tensor(values, dtype=torch.long, device=device)


def incremental_condition(model, x, y, control, permutation, audit_positions=()):
    state = model.init_incremental_state(
        x.size(0), device=x.device, dtype=torch.bfloat16
    )
    per_sequence_nll = torch.zeros(x.size(0), dtype=torch.float64)
    per_position_nll = torch.zeros(base.T, dtype=torch.float64)
    cache_rows = []
    shuffled = control.endswith("shuffled")
    for position in range(base.T):
        wants_audit = position + 1 in audit_positions
        result = model.incremental_step(
            x[:, position], state, control=control,
            recurrent_permutation=permutation if shuffled else None,
            return_diagnostics=wants_audit,
            diagnostic_attention_weights=False,
        )
        if wants_audit:
            logits, state, diagnostic = result
            cache_rows.append({
                "position": position + 1,
                "cache_audit": diagnostic["cache_audit"],
                "links": {
                    key: {
                        "recurrent_positions": None
                        if row is None or row.get("recurrent_positions") is None
                        else row["recurrent_positions"].detach().cpu().tolist(),
                        "gate_coefficient": None
                        if row is None else float(row["gate_coefficient"].detach().float()),
                    }
                    for key, row in diagnostic["links"].items()
                },
            })
        else:
            logits, state = result
        losses = F.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        ).double().cpu()
        per_sequence_nll += losses
        per_position_nll[position] += losses.sum()
    targets = int(x.numel())
    return {
        "nll_sum": per_sequence_nll.sum().item(),
        "targets": targets,
        "per_sequence_nll": per_sequence_nll.tolist(),
        "per_sequence_ce": (per_sequence_nll / base.T).tolist(),
        "per_position_nll_sum": per_position_nll.tolist(),
        "cache_rows": cache_rows,
        "final_cache_audit": model.incremental_cache_audit(state),
    }


def make_evaluation_identity(model, family, checkpoint_sha256, local_update,
                             panel_manifest_path, shuffle_manifest_path):
    identity = {
        "family": family,
        "checkpoint_sha256": checkpoint_sha256,
        "local_update": int(local_update),
        "model_state_sha256": parameter_manifest(model)["aggregate_sha256"],
        "architecture_fingerprint": model.architecture_fingerprint(),
        "panel_manifest_path": str(Path(panel_manifest_path).resolve()),
        "panel_manifest_sha256": sha256(panel_manifest_path),
        "shuffle_manifest_path": str(Path(shuffle_manifest_path).resolve()),
        "shuffle_manifest_sha256": sha256(shuffle_manifest_path),
    }
    identity["identity_sha256"] = canonical_sha(identity)
    return identity


def empty_eval_state(family, controls, panel, architecture_fingerprint,
                     evaluation_identity):
    return {
        "experiment": EXPERIMENT,
        "family": family,
        "controls_requested": list(controls),
        "architecture_fingerprint": architecture_fingerprint,
        "evaluation_identity": evaluation_identity,
        "panel_sha256": panel.get("panel_sha256", panel.get("subset_sha256")),
        "batch_indices_in_evaluation_order": panel_batch_indices(panel),
        "completed_batch_indices": [],
        "batch_identities": [],
        "conditions": {
            control: {
                "nll_sum": 0.0,
                "targets": 0,
                "per_sequence_nll": [],
                "per_sequence_ce": [],
                "per_position_nll_sum": [0.0] * base.T,
                "cache_rows": [],
                "final_cache_audit": None,
            }
            for control in controls
        },
        "same_sequence_order_all_conditions": True,
        "cache_reset_between_conditions": True,
        "status": "running",
    }


def finalize_eval_state(state, started, device, model, val_path, shuffle_manifest):
    for row in state["conditions"].values():
        row["validation_loss"] = row["nll_sum"] / row["targets"]
        row["validation_targets"] = row["targets"]
        row["paired_sequences"] = len(row["per_sequence_nll"])
        row["per_position_loss"] = (
            np.asarray(row.pop("per_position_nll_sum"), dtype=np.float64)
            / row["paired_sequences"]
        ).tolist()
    first_batch = state["batch_indices_in_evaluation_order"][0]
    cpu_x, cpu_y = batch_at_index(val_path, first_batch)
    x, y = cpu_x[:4].to(device), cpu_y[:4].to(device)
    permutation = permutation_for_batch(shuffle_manifest, 64, device)[:4]
    # ALL_REAL does not consume the permutation; using a four-row sentinel is
    # therefore path-identical and independent across rows.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sentinel = incremental_condition(model, x, y, "all_real", permutation)
    expected = state["conditions"]["all_real"]["per_sequence_ce"][:4]
    delta = np.max(np.abs(np.asarray(expected) - np.asarray(sentinel["per_sequence_ce"])))
    state["all_real_terminal_sentinel"] = {
        "repeated_sequences": 4,
        "max_abs_ce": float(delta),
        "passed": bool(delta <= 1e-12),
    }
    state["performance"] = {
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    state["status"] = "complete"
    state["passed"] = (
        state["all_real_terminal_sentinel"]["passed"]
        and all(
            row["validation_targets"] == state["conditions"]["all_real"]["validation_targets"]
            for row in state["conditions"].values()
        )
        and all(row["final_cache_audit"]["passed"] for row in state["conditions"].values())
    )
    return state


def evaluate_incremental_panel(model, val_path, panel, controls, shuffle_manifest,
                               family, output_path, evaluation_identity,
                               resume=True):
    model.eval()
    device = base.model_device(model)
    architecture_fingerprint = model.architecture_fingerprint()
    output_path = Path(output_path)
    if resume and output_path.exists():
        state = read_json(output_path)
        if state.get("evaluation_identity") != evaluation_identity:
            raise SystemExit("evaluation checkpoint/panel/shuffle identity mismatch")
        if state.get("status") == "complete":
            return state
        expected = (family, list(controls), architecture_fingerprint,
                    panel.get("panel_sha256", panel.get("subset_sha256")))
        actual = (state["family"], state["controls_requested"],
                  state["architecture_fingerprint"], state["panel_sha256"])
        if actual != expected:
            raise SystemExit("evaluation resume metadata mismatch")
    else:
        state = empty_eval_state(
            family, controls, panel, architecture_fingerprint,
            evaluation_identity,
        )
    completed = set(state["completed_batch_indices"])
    permutation = permutation_for_batch(shuffle_manifest, base.VALIDATION_B, device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    audit_positions = (1, 2, 3, 8, 16, 32, 64, 128, 512, 1024)
    with torch.no_grad():
        for ordinal, batch_index in enumerate(state["batch_indices_in_evaluation_order"]):
            if batch_index in completed:
                continue
            cpu_x, cpu_y = batch_at_index(val_path, batch_index)
            identity = base.batch_identity(cpu_x, cpu_y)
            expected_identity = panel.get("batch_identities", [])[ordinal]
            if expected_identity and identity != expected_identity:
                raise SystemExit(f"panel batch identity mismatch at ordinal {ordinal}")
            x, y = cpu_x.to(device), cpu_y.to(device)
            for control in controls:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    current = incremental_condition(
                        model, x, y, control, permutation,
                        audit_positions if ordinal == 0 else (),
                    )
                row = state["conditions"][control]
                row["nll_sum"] += current["nll_sum"]
                row["targets"] += current["targets"]
                row["per_sequence_nll"].extend(current["per_sequence_nll"])
                row["per_sequence_ce"].extend(current["per_sequence_ce"])
                row["per_position_nll_sum"] = (
                    np.asarray(row["per_position_nll_sum"], dtype=np.float64)
                    + np.asarray(current["per_position_nll_sum"], dtype=np.float64)
                ).tolist()
                row["cache_rows"].extend(current["cache_rows"])
                row["final_cache_audit"] = current["final_cache_audit"]
            state["completed_batch_indices"].append(batch_index)
            state["batch_identities"].append(identity)
            durable_json(output_path, state)
            print(
                f"2D5C {family} incremental batch {ordinal + 1}/{len(state['batch_indices_in_evaluation_order'])}",
                flush=True,
            )
            del x, y, cpu_x, cpu_y
            torch.cuda.empty_cache()
    state = finalize_eval_state(
        state, started, device, model, val_path, shuffle_manifest
    )
    durable_json(output_path, state)
    return state


def diagnostic_subset_batch(core_manifest, val_path, device):
    rows = []
    for selected in core_manifest["diagnostic_sequence_identities"]:
        x, y = batch_at_index(val_path, int(selected["batch_index"]))
        index = int(selected["sequence_index"])
        if tensor_sha256(x[index], y[index]) != selected["combined_sha256"]:
            raise SystemExit("diagnostic sequence identity mismatch")
        rows.append((x[index], y[index]))
    x = torch.stack([row[0] for row in rows]).to(device)
    y = torch.stack([row[1] for row in rows]).to(device)
    return x, y


def ring_index_mapping_test(model, device):
    """Probe the real recurrent-ring selector across two physical rollovers."""
    capacity = int(core.RECURRENT_RING_CAPACITY)
    channels = int(model.config.n_embd)
    initial_positions = tuple(range(capacity))
    ring = torch.arange(
        capacity, device=device, dtype=torch.float32
    ).view(1, capacity, 1).expand(1, capacity, channels).clone()
    with torch.no_grad():
        after_first, positions_first = model._append_ring(
            ring,
            initial_positions,
            torch.full((1, 1, channels), float(capacity), device=device),
            capacity,
        )
        after_second, positions_second = model._append_ring(
            after_first,
            positions_first,
            torch.full((1, 1, channels), float(capacity + 1), device=device),
            capacity + 1,
        )
        query_position = capacity + 2
        bank = model._incremental_bank_from_ring(
            after_second,
            positions_second,
            query_position,
            minimum_lag=2,
            mode="full",
        )
        excluded_newest_changed = after_second.clone()
        excluded_newest_changed[:, -1].fill_(-1_000_000.0)
        bank_after_excluded_change = model._incremental_bank_from_ring(
            excluded_newest_changed,
            positions_second,
            query_position,
            minimum_lag=2,
            mode="full",
        )

    expected_ring_positions = tuple(range(2, capacity + 2))
    expected_bank_positions = list(range(2, capacity + 1))
    observed_positions = bank.positions[0].detach().cpu().tolist()
    observed_values = bank.values[0, :, 0].detach().cpu().tolist()
    physical_mapping = {
        "lag_1023": {
            "logical_source_position": 2,
            "physical_ring_index": positions_second.index(2),
            "selected_bank_index": observed_positions.index(2),
            "encoded_value": observed_values[observed_positions.index(2)],
        },
        "lag_2": {
            "logical_source_position": capacity,
            "physical_ring_index": positions_second.index(capacity),
            "selected_bank_index": observed_positions.index(capacity),
            "encoded_value": observed_values[observed_positions.index(capacity)],
        },
    }
    current_alias_rejected = False
    future_alias_rejected = False
    try:
        model._incremental_bank_from_ring(
            after_second,
            tuple(range(3, capacity + 3)),
            query_position,
            minimum_lag=2,
            mode="full",
        )
    except ValueError:
        current_alias_rejected = True
    try:
        model._incremental_bank_from_ring(
            after_second,
            tuple(range(4, capacity + 4)),
            query_position,
            minimum_lag=2,
            mode="full",
        )
    except ValueError:
        future_alias_rejected = True

    checks = {
        "first_rollover_positions_exact": positions_first
        == tuple(range(1, capacity + 1)),
        "second_rollover_positions_exact": positions_second
        == expected_ring_positions,
        "physical_capacity_exact": after_second.size(1) == capacity,
        "eligible_positions_exact": observed_positions == expected_bank_positions,
        "eligible_values_follow_physical_mapping": observed_values
        == [float(value) for value in expected_bank_positions],
        "lag_1023_maps_to_physical_slot_0": physical_mapping["lag_1023"]
        == {
            "logical_source_position": 2,
            "physical_ring_index": 0,
            "selected_bank_index": 0,
            "encoded_value": 2.0,
        },
        "lag_2_maps_to_physical_slot_1021": physical_mapping["lag_2"]
        == {
            "logical_source_position": capacity,
            "physical_ring_index": capacity - 2,
            "selected_bank_index": capacity - 2,
            "encoded_value": float(capacity),
        },
        "lag_1_newest_slot_excluded": (capacity + 1) not in observed_positions,
        "excluded_newest_cannot_change_selected_bank": torch.equal(
            bank.values, bank_after_excluded_change.values
        ),
        "no_current_or_future_selected": all(
            int(position) < query_position for position in observed_positions
        ),
        "current_position_metadata_rejected": current_alias_rejected,
        "future_position_metadata_rejected": future_alias_rejected,
    }
    return {
        "ring_capacity": capacity,
        "query_position_after_rollover": query_position,
        "ring_positions": list(positions_second),
        "selected_position_range": [observed_positions[0], observed_positions[-1]],
        "physical_mapping": physical_mapping,
        "checks": checks,
        "passed": all(checks.values()),
    }


def causality_test(model, device, post_rollover_probe=None):
    generator = torch.Generator(device=device).manual_seed(2_026_083_004)
    rows = []
    for length, cutoff in ((80, 47), (1024, 700)):
        tokens = torch.randint(0, 50_257, (1, length), generator=generator, device=device)
        changed = tokens.clone()
        changed[:, cutoff + 1:] = torch.randint(
            0, 50_257, changed[:, cutoff + 1:].shape,
            generator=generator, device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parallel_left = model.forward_multi_pass(tokens, num_passes=2)["logits"]
            parallel_right = model.forward_multi_pass(changed, num_passes=2)["logits"]
            incremental_left = model.incremental_logits(tokens)["logits"]
            incremental_right = model.incremental_logits(changed)["logits"]
        parallel_delta = (
            parallel_left[:, :cutoff + 1].float()
            - parallel_right[:, :cutoff + 1].float()
        ).abs().max().item()
        incremental_delta = (
            incremental_left[:, :cutoff + 1].float()
            - incremental_right[:, :cutoff + 1].float()
        ).abs().max().item()
        rows.append({
            "length": length,
            "cutoff": cutoff,
            "ring_reaches_capacity": length == 1024,
            "parallel_prefix_max_abs": parallel_delta,
            "incremental_prefix_max_abs": incremental_delta,
            "passed": parallel_delta == 0.0 and incremental_delta == 0.0,
        })
    post_rollover_probe = (
        ring_index_mapping_test(model, device)
        if post_rollover_probe is None
        else post_rollover_probe
    )
    return {
        "token_suffix_perturbation_cases": rows,
        "post_rollover_ring_causality": post_rollover_probe,
        "passed": all(row["passed"] for row in rows)
        and post_rollover_probe["passed"],
    }


def incremental_state_comparison(left, right, model):
    cache_rows = []
    for block_index, (left_cache, right_cache) in enumerate(
        zip(left.caches, right.caches), start=1
    ):
        none_parity = (left_cache is None) == (right_cache is None)
        key_exact = none_parity and (
            left_cache is None or torch.equal(left_cache.key, right_cache.key)
        )
        value_exact = none_parity and (
            left_cache is None or torch.equal(left_cache.value, right_cache.value)
        )
        cache_rows.append({
            "block": block_index,
            "none_parity": bool(none_parity),
            "key_exact": bool(key_exact),
            "value_exact": bool(value_exact),
            "passed": bool(none_parity and key_exact and value_exact),
        })
    ring_rows = {}
    for name in ("h7", "h8", "h10", "h12"):
        positions_exact = getattr(left, f"{name}_positions") == getattr(
            right, f"{name}_positions"
        )
        values_exact = torch.equal(
            getattr(left, f"{name}_ring"), getattr(right, f"{name}_ring")
        )
        ring_rows[name] = {
            "positions_exact": bool(positions_exact),
            "values_exact": bool(values_exact),
            "passed": bool(positions_exact and values_exact),
        }
    scalar_checks = {
        "position_exact": left.position == right.position,
        "batch_size_exact": left.batch_size == right.batch_size,
        "b6_full_native_exact": left.b6_full_native == right.b6_full_native,
        "cache_lengths_exact": model.incremental_cache_lengths(left)
        == model.incremental_cache_lengths(right),
        "cache_count_exact": len(left.caches) == len(right.caches) == 12,
    }
    return {
        "scalar_checks": scalar_checks,
        "all_kv_caches": cache_rows,
        "writer_rings": ring_rows,
        "passed": all(scalar_checks.values())
        and all(row["passed"] for row in cache_rows)
        and all(row["passed"] for row in ring_rows.values()),
    }


def incremental_reload_equivalence(model, device):
    generator = torch.Generator(device=device).manual_seed(2_026_083_005)
    tokens = torch.randint(0, 50_257, (2, 160), generator=generator, device=device)
    full = model.init_incremental_state(2, device=device, dtype=torch.bfloat16)
    split = model.init_incremental_state(2, device=device, dtype=torch.bfloat16)
    full_logits, split_logits = [], []
    temporary = Path(f"/tmp/exp2d5c_incremental_state_{os.getpid()}.pt")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            logits, full = model.incremental_step(tokens[:, position], full)
            full_logits.append(logits)
        for start, end in ((0, 37), (37, 91), (91, 160)):
            for position in range(start, end):
                logits, split = model.incremental_step(tokens[:, position], split)
                split_logits.append(logits)
            if end == 91:
                torch.save(split, temporary)
                split = torch.load(temporary, map_location=device, weights_only=False)
    temporary.unlink(missing_ok=True)
    left, right = torch.cat(full_logits, 1), torch.cat(split_logits, 1)
    delta = (left.float() - right.float()).abs().max().item()
    state_comparison = incremental_state_comparison(full, split, model)
    return {
        "logits_exact": bool(torch.equal(left, right)),
        "logits_max_abs": delta,
        "state_comparison": state_comparison,
        "all_kv_caches_exact": all(
            row["passed"] for row in state_comparison["all_kv_caches"]
        ),
        "all_h7_h8_h10_h12_rings_exact": all(
            row["passed"] for row in state_comparison["writer_rings"].values()
        ),
        "cache_state_exact": state_comparison["passed"],
        "temporary_removed": not temporary.exists(),
        "passed": bool(
            torch.equal(left, right)
            and state_comparison["passed"]
            and not temporary.exists()
        ),
    }


def attached_gradient_test(model, optimizer, val_path, device):
    model_before = parameter_manifest(model)["aggregate_sha256"]
    optimizer_before = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    loader = base.d1.ExplicitShardLoader([val_path], 2, 96)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model.forward_pass(x, targets=y)
        first["h10"].retain_grad()
        first["h8"].retain_grad()
        second = model.forward_pass(
            x, targets=y, b1_recurrent_source=first["h12"],
            b3_recurrent_source=first["h10"], b5_recurrent_source=first["h8"],
            b6_recurrent_source=first["h7"], return_diagnostics=True,
        )
        loss = second["loss"]
    loss.backward()
    rows = {}
    for name, source, gate in (
        ("B10_to_B3", first["h10"], model.g_rec_b3),
        ("B8_to_B5", first["h8"], model.g_rec_b5),
    ):
        gradient = source.grad.detach().float()
        gate_gradient = gate.grad.detach().float()
        rows[name] = {
            "writer_gradient_norm": gradient.norm().item(),
            "writer_gradient_finite": bool(torch.isfinite(gradient).all()),
            "writer_gradient_nonzero": bool(torch.count_nonzero(gradient)),
            "gate_gradient_norm": gate_gradient.norm().item(),
            "gate_gradient_finite": bool(torch.isfinite(gate_gradient).all()),
            "gate_gradient_nonzero": bool(torch.count_nonzero(gate_gradient)),
        }
        rows[name]["passed"] = all(value for key, value in rows[name].items() if key.endswith("finite") or key.endswith("nonzero"))
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    model_after = parameter_manifest(model)["aggregate_sha256"]
    optimizer_after = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    return {
        "ce_only_loss": float(loss.detach()),
        "links": rows,
        "model_unchanged": model_before == model_after,
        "optimizer_unchanged": optimizer_before == optimizer_after,
        "no_auxiliary_loss": True,
        "passed": all(row["passed"] for row in rows.values())
        and model_before == model_after and optimizer_before == optimizer_after,
    }


def control_specificity_test(model, device, shuffle_manifest):
    generator = torch.Generator(device=device).manual_seed(2_026_083_006)
    batch_size = 32
    tokens = torch.randint(
        0, 50_257, (batch_size, 16), generator=generator, device=device
    )
    frozen_permutation = shuffle_manifest["permutations"][str(batch_size)]
    permutation_values = frozen_permutation["values"]
    permutation = torch.tensor(permutation_values, device=device, dtype=torch.long)
    manifest_checks = {
        "seed_exact": int(shuffle_manifest["seed"]) == SHUFFLE_SEED,
        "permutation_sha_exact": artifacts.canonical_json_sha256(permutation_values)
        == frozen_permutation["sha256"],
        "permutation_domain_exact": sorted(permutation_values)
        == list(range(batch_size)),
        "derangement_exact": all(
            index != donor for index, donor in enumerate(permutation_values)
        ),
    }
    was_training = model.training
    model.eval()
    outputs = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        baseline = model.incremental_logits(tokens, return_diagnostics=True)
        first = baseline["logits"]
        for control in CONTROLS[1:]:
            current = model.incremental_logits(
                tokens, control=control,
                recurrent_permutation=permutation if control.endswith("shuffled") else None,
                return_diagnostics=True,
            )
            logits = current["logits"]
            outputs[control] = {
                "logits_sha256": tensor_sha256(logits),
                "max_abs_delta_vs_all_real": float(
                    (logits.float() - first.float()).abs().max()
                ),
                "logits_changed_vs_all_real": not torch.equal(logits, first),
                "probe": current["diagnostics"][2],
            }
        final = model.incremental_logits(tokens)["logits"]
    if was_training:
        model.train()
    sets = {control: model.control_sets(control) for control in CONTROLS}
    expected = {
        "all_real": (set(), set()),
        "b3_off": ({2}, set()),
        "b3_shuffled": (set(), {2}),
        "b5_off": ({4}, set()),
        "b5_shuffled": (set(), {4}),
        "b3_b5_off": ({2, 4}, set()),
        "b3_b5_shuffled": (set(), {2, 4}),
    }
    baseline_probe = baseline["diagnostics"][2]

    def optional_tensor_exact(left, right):
        return (left is None) == (right is None) and (
            left is None or torch.equal(left, right)
        )

    intervention_evidence = {}
    for control in CONTROLS[1:]:
        off, shuffled = sets[control]
        probe = outputs[control].pop("probe")
        link_checks = {}
        for block_index in (0, 2, 4, 5):
            name = f"b{block_index + 1}"
            observed = probe["links"][name]
            reference = baseline_probe["links"][name]
            if block_index in off:
                checks = {
                    "off_bank_removed": observed["recurrent_positions"] is None
                    and observed["recurrent_source_reads"] is None,
                    "off_gate_zero": float(observed["gate_coefficient"]) == 0.0,
                }
            elif block_index in shuffled:
                checks = {
                    "positions_preserved": optional_tensor_exact(
                        observed["recurrent_positions"],
                        reference["recurrent_positions"],
                    ),
                    "gate_preserved": torch.equal(
                        observed["gate_coefficient"],
                        reference["gate_coefficient"],
                    ),
                    "frozen_donor_permutation_applied_to_actual_source_reads": (
                        observed["recurrent_source_reads"] is not None
                        and reference["recurrent_source_reads"] is not None
                        and torch.equal(
                            observed["recurrent_source_reads"],
                            reference["recurrent_source_reads"].index_select(
                                0, permutation
                            ),
                        )
                    ),
                }
            else:
                checks = {
                    "positions_unchanged_at_first_eligible_query": optional_tensor_exact(
                        observed["recurrent_positions"],
                        reference["recurrent_positions"],
                    ),
                    "gate_unchanged_at_first_eligible_query": torch.equal(
                        observed["gate_coefficient"],
                        reference["gate_coefficient"],
                    ),
                    "source_reads_unchanged_at_first_eligible_query": optional_tensor_exact(
                        observed["recurrent_source_reads"],
                        reference["recurrent_source_reads"],
                    ),
                }
            link_checks[name] = {
                "targeted": block_index in off | shuffled,
                "checks": checks,
                "passed": all(checks.values()),
            }
        intervention_evidence[control] = {
            **outputs[control],
            "reported_control_exact": probe["control"] == control,
            "links": link_checks,
            "passed": outputs[control]["logits_changed_vs_all_real"]
            and probe["control"] == control
            and all(row["passed"] for row in link_checks.values()),
        }
    all_real_repeat_exact = bool(torch.equal(first, final))
    controls_exact = all(sets[name] == expected[name] for name in CONTROLS)
    return {
        "batch_size": batch_size,
        "sequence_length": tokens.size(1),
        "first_eligible_query_probe": 2,
        "frozen_shuffle_manifest_checks": manifest_checks,
        "all_real_logits_sha256": tensor_sha256(first),
        "control_sets": {
            name: {"off": sorted(value[0]), "shuffled": sorted(value[1])}
            for name, value in sets.items()
        },
        "intervention_outputs": intervention_evidence,
        "all_interventions_change_actual_logits": all(
            row["logits_changed_vs_all_real"]
            for row in intervention_evidence.values()
        ),
        "only_b3_b5_targeted": controls_exact,
        "b1_b6_always_real_aligned": all(
            0 not in off | shuffled and 5 not in off | shuffled
            for off, shuffled in sets.values()
        ),
        "all_real_repeat_exact": all_real_repeat_exact,
        "flag_leakage_absent": all_real_repeat_exact,
        "passed": controls_exact
        and all(manifest_checks.values())
        and all_real_repeat_exact
        and all(row["passed"] for row in intervention_evidence.values()),
    }


def forbidden_component_audit(model):
    config_path = REPO_ROOT / "configs" / "exp2d5c_fixed_writer_b3_b5_w2_matched_100m.json"
    config_architecture = read_json(config_path)["architecture_c"]
    parameter_names = [name.lower() for name, _ in model.named_parameters()]
    buffer_names = [name.lower() for name, _ in model.named_buffers()]
    module_rows = [
        {"name": name.lower(), "type": type(module).__name__.lower()}
        for name, module in model.named_modules()
    ]
    forbidden_terms = (
        "router", "routing", "route_logits", "route_beta", "attnres",
        "teacher", "distill", "auxiliary", "aux_head",
    )
    registered_name_hits = sorted({
        value
        for value in parameter_names + buffer_names
        + [f'{row["name"]}:{row["type"]}' for row in module_rows]
        if any(term in value for term in forbidden_terms)
    })
    allowed_projection_suffixes = (
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", "lm_head"
    )
    projection_modules = sorted(
        row["name"] for row in module_rows
        if row["name"] and (
            "proj" in row["name"]
            or row["type"] in {"linear", "conv1d"}
        )
    )
    unexpected_projection_modules = [
        name for name in projection_modules
        if not name.endswith(allowed_projection_suffixes)
    ]
    runtime_config = {
        str(name).lower(): value for name, value in vars(model.config).items()
    }
    runtime_config_hits = sorted(
        name for name, value in runtime_config.items()
        if any(term in name for term in forbidden_terms) and bool(value)
    )
    config_architecture_text = json.dumps(
        config_architecture, sort_keys=True, separators=(",", ":")
    ).lower()
    config_checks = {
        "parameter_count_exact": int(config_architecture["parameters"]) == PARAMETERS,
        "new_parameters_zero": int(config_architecture["new_parameters"]) == 0,
        "no_forbidden_architecture_keys_or_values": not any(
            term in config_architecture_text for term in forbidden_terms
        ),
        "runtime_manifest_new_parameters_zero": ARCHITECTURE_C["new_parameters"] == 0,
        "runtime_manifest_state_dict_changes_zero": ARCHITECTURE_C[
            "state_dict_key_changes"
        ] == 0,
        "runtime_manifest_router_false": ARCHITECTURE_C["router"] is False,
        "runtime_manifest_auxiliary_false": ARCHITECTURE_C[
            "auxiliary_objective"
        ] is False,
        "runtime_manifest_ce_only": ARCHITECTURE_C["ce_only"] is True,
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    module_checks = {
        "parameter_count_exact": parameter_count == PARAMETERS,
        "forbidden_registered_names_absent": not registered_name_hits,
        "forbidden_runtime_config_absent": not runtime_config_hits,
        "unexpected_projection_modules_absent": not unexpected_projection_modules,
    }
    return {
        "configuration_path": str(config_path),
        "parameter_count": parameter_count,
        "configuration_checks": config_checks,
        "module_checks": module_checks,
        "forbidden_registered_name_hits": registered_name_hits,
        "forbidden_runtime_config_hits": runtime_config_hits,
        "registered_projection_modules": projection_modules,
        "unexpected_projection_modules": unexpected_projection_modules,
        "registered_buffers": buffer_names,
        "fixed_writer_sources": dict(core.FIXED_WRITERS),
        "passed": all(config_checks.values()) and all(module_checks.values()),
    }


def representation_diagnostic_feasibility_smoke(model, optimizer, val_path, device):
    """Execute one real batch-1 x 1024 attached-gradient diagnostic."""
    loader = base.d1.ExplicitShardLoader([val_path], 1, base.T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x[0].to(device), cpu_y[0].to(device)
    model_before = parameter_manifest(model)["aggregate_sha256"]
    optimizer_before = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = one_sequence_representation_diagnostic(
        model, x, y, "C0", 1.0
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    link_rows = {}
    for link in ("b3", "b5"):
        current = result["links"][link]
        source_nonzero_pairs = sum(
            int(row["nonzero_pairs"])
            for row in current["gradient"]["source"].values()
        )
        writer = current["actual_writer_gradient"]
        link_rows[link] = {
            "source_gradient_nonzero_pairs": source_nonzero_pairs,
            "writer_positions": int(writer["positions"]),
            "writer_positions_with_gradient": int(writer["positions_with_gradient"]),
            "writer_positions_with_nonzero_gradient": int(
                writer["positions_with_nonzero_gradient"]
            ),
            "actual_writer_gradient_l2": float(
                writer["l2_norm_of_all_elements"]
            ),
            "passed": source_nonzero_pairs > 0
            and writer["nonzero_back_to_actual_writer"]
            and math.isfinite(float(writer["l2_norm_of_all_elements"])),
        }
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    model_after = parameter_manifest(model)["aggregate_sha256"]
    optimizer_after = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    checks = {
        "production_batch_shape_exact": list(cpu_x.shape) == [1, base.T]
        and list(cpu_y.shape) == [1, base.T],
        "all_targets_executed": all(
            row["writer_positions"] == base.T for row in link_rows.values()
        ),
        "mean_ce_finite": result["finite"]
        and math.isfinite(float(result["mean_ce"])),
        "b3_attached_gradient_feasible": link_rows["b3"]["passed"],
        "b5_attached_gradient_feasible": link_rows["b5"]["passed"],
        "model_unchanged": model_before == model_after,
        "optimizer_unchanged": optimizer_before == optimizer_after,
        "elapsed_time_measured": math.isfinite(elapsed) and elapsed > 0.0,
        "peak_memory_measured": peak_allocated >= baseline_allocated
        and peak_reserved >= baseline_reserved,
    }
    summary = {
        "shape": {
            "batch_sequences": 1,
            "sequence_length": base.T,
            "targets": base.T,
            "dtype": "torch.bfloat16",
        },
        "elapsed_seconds": elapsed,
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
            "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
        },
        "mean_ce": float(result["mean_ce"]),
        "links": link_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del result, x, y, cpu_x, cpu_y
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def lag_coverage_test(model, device):
    length = base.T
    rows = {}
    query_positions = (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64,
                       127, 128, 511, 512, 1023)
    for block_index, expected_min in ((0, 2), (2, 2), (4, 2), (5, 512)):
        local = model.local_mask(block_index, length, device).cpu()
        recurrent = model.recurrent_mask(
            block_index, length, length, device
        ).cpu()
        current = []
        passed = True
        local_window = core.LOCAL_WINDOWS_C[block_index]
        for query in query_positions:
            available = torch.arange(query + 1)
            lags = query - available
            local_lags = set(lags[local[query, :query + 1]].tolist())
            recurrent_lags = set(lags[recurrent[query, :query + 1]].tolist())
            expected_local = set(range(0, min(query, local_window - 1) + 1))
            expected_recurrent = set(range(expected_min, min(query, 1023) + 1))
            checks = {
                "local_exact": local_lags == expected_local,
                "recurrent_exact": recurrent_lags == expected_recurrent,
                "no_overlap": not (local_lags & recurrent_lags),
                "no_missing": local_lags | recurrent_lags == set(range(query + 1)),
                "no_future": not bool(local[query, query + 1:].any() or recurrent[query, query + 1:].any()),
            }
            passed = passed and all(checks.values())
            current.append({
                "query_position": query,
                "local_lag_minmax": None if not local_lags else [min(local_lags), max(local_lags)],
                "recurrent_lag_minmax": None if not recurrent_lags else [min(recurrent_lags), max(recurrent_lags)],
                "checks": checks,
            })
        rows[f"B{block_index + 1}"] = {
            "local_window": local_window,
            "recurrent_min_lag": expected_min,
            "positions": current,
            "passed": passed,
        }
    return {"blocks": rows, "passed": all(row["passed"] for row in rows.values())}


def cache_capacity_test(model, device):
    tokens = (torch.arange(base.T, device=device) * 3571 + 11).remainder(50_257).view(1, -1)
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    rows = []
    checkpoints = {1, 2, 3, 8, 16, 32, 64, 128, 512, 1024}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(base.T):
            wants = position + 1 in checkpoints
            result = model.incremental_step(
                tokens[:, position], state, return_diagnostics=wants,
                diagnostic_attention_weights=False,
            )
            if wants:
                _, state, diagnostic = result
                rows.append({
                    "position": position + 1,
                    "cache_lengths": diagnostic["cache_audit"]["cache_lengths"],
                    "ring_lengths": diagnostic["cache_audit"]["ring_lengths"],
                    "recurrent_positions": {
                        key: [] if row is None or row.get("recurrent_positions") is None
                        else row["recurrent_positions"].detach().cpu().reshape(-1).tolist()
                        for key, row in diagnostic["links"].items()
                    },
                })
            else:
                _, state = result
    audit = model.incremental_cache_audit(state)
    reset = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    checks = {
        "B3_one_historical_native": audit["cache_lengths"][2] == 1,
        "B5_one_historical_native": audit["cache_lengths"][4] == 1,
        "B1_unchanged": audit["cache_lengths"][0] == 1,
        "B6_unchanged": audit["cache_lengths"][5] == 511,
        "rings_unchanged_1023": all(value == 1023 for value in audit["ring_lengths"].values()),
        "old_entries_evicted": all(
            positions == tuple(range(1, 1024)) for positions in (
                state.h7_positions, state.h8_positions, state.h10_positions,
                state.h12_positions,
            )
        ),
        "reset_complete": all(value == 0 for value in model.incremental_cache_lengths(reset))
        and all(not getattr(reset, f"{name}_positions") for name in ("h7", "h8", "h10", "h12")),
        "physical_audit": audit["passed"],
    }
    return {"positions": rows, "final_audit": audit, "checks": checks, "passed": all(checks.values())}


def smoke_update(model, optimizer, loader, accumulation, local_update, device):
    global_update = SOURCE_UPDATES + int(local_update)
    before_steps = optimizer_steps(optimizer)
    batch_hash = base.next_batch_hash(loader, accumulation)
    stream_hash = base.next_stream_hash(loader, accumulation)
    row = base.train_update(
        model, optimizer, loader, accumulation, global_update, device
    )
    after_steps = optimizer_steps(optimizer)
    increments = len(before_steps) == len(after_steps) and all(
        after == before + 1 for before, after in zip(sorted(before_steps), sorted(after_steps))
    )
    row.update({
        "local_update": int(local_update),
        "global_update": global_update,
        "new_targets": int(local_update) * base.GLOBAL_TARGETS,
        "cumulative_targets": SOURCE_TARGETS + int(local_update) * base.GLOBAL_TARGETS,
        "consumed_batch_sha256": batch_hash,
        "consumed_stream_sha256": stream_hash,
        "optimizer_step_increment_exact": increments,
        "optimizer_steps_before_summary": sorted(set(before_steps)),
        "optimizer_steps_after_summary": sorted(set(after_steps)),
        "forward_passes": row["pass_count"] * int(accumulation),
        "backward_invocations": int(accumulation),
        "optimizer_steps": 1,
        "scheduler_steps": 0,
    })
    if not increments:
        raise SystemExit("optimizer step did not advance exactly once")
    return row


def disposable_smoke(source_checkpoint, val_path, pretrain_dir, device, output):
    source_before = sha256(source_checkpoint)
    model, optimizer, loader, source, _ = make_c_model(
        source_checkpoint, device, restore=True
    )
    accumulation = int(source["gradient_accumulation"])
    replay = [
        json.loads(line) for line in (Path(pretrain_dir) / "DATA_REPLAY_LEDGER.jsonl").read_text().splitlines()
        if line.strip()
    ]
    panel = read_json(Path(pretrain_dir) / "PANEL_MANIFEST_LARGE.json")
    replay_audit = read_json(Path(pretrain_dir) / "DATA_REPLAY_AUDIT.json")
    metadata = continuation_metadata(source, panel, replay_audit)
    rows = []
    for local_update in (1, 2):
        expected = replay[local_update - 1]
        if base.next_batch_hash(loader, accumulation) != expected["logical_global_batch_sha256"]:
            raise SystemExit("disposable smoke replay mismatch")
        rows.append(smoke_update(model, optimizer, loader, accumulation, local_update, device))
    temporary = Path(f"/tmp/DISPOSABLE_exp2d5c_update0002_{os.getpid()}.pt")
    verification = save_checkpoint(
        temporary, model, optimizer, loader, source, 2, accumulation, metadata,
        replay_audit["ledger_sha256"], source_checkpoint, device, sidecars=False,
    )
    in_memory_row = smoke_update(model, optimizer, loader, accumulation, 3, device)
    in_memory_model = parameter_manifest(model)["aggregate_sha256"]
    in_memory_optimizer = optimizer_manifest(model, optimizer)["state_aggregate_sha256"]
    in_memory_loader = loader.state_dict()
    reloaded, reloaded_optimizer, reloaded_loader, _, _ = load_c_checkpoint(
        temporary, source_checkpoint, device, restore=True
    )
    reloaded_row = smoke_update(
        reloaded, reloaded_optimizer, reloaded_loader, accumulation, 3, device
    )
    continuation_checks = {
        "model_exact": parameter_manifest(reloaded)["aggregate_sha256"] == in_memory_model,
        "optimizer_exact": optimizer_manifest(reloaded, reloaded_optimizer)["state_aggregate_sha256"] == in_memory_optimizer,
        "loader_exact": reloaded_loader.state_dict() == in_memory_loader,
        "batch_exact": reloaded_row["consumed_batch_sha256"] == in_memory_row["consumed_batch_sha256"],
        "stream_exact": reloaded_row["consumed_stream_sha256"] == in_memory_row["consumed_stream_sha256"],
        "pass_count_exact": reloaded_row["pass_count"] == in_memory_row["pass_count"],
    }
    rows.append(reloaded_row)
    # Continue only the disposable reloaded branch to the first scheduled
    # three-pass global update (global 1920 / local 12).
    for local_update in range(4, 13):
        expected = replay[local_update - 1]
        if base.next_batch_hash(reloaded_loader, accumulation) != expected["logical_global_batch_sha256"]:
            raise SystemExit("disposable smoke replay mismatch before three-pass coverage")
        rows.append(smoke_update(
            reloaded, reloaded_optimizer, reloaded_loader,
            accumulation, local_update, device,
        ))
    temporary.unlink(missing_ok=True)
    report = {
        "label": "DISPOSABLE",
        "official_source_unchanged": sha256(source_checkpoint) == source_before == SOURCE_SHA256,
        "updates_executed_on_disposable_copies": 12,
        "rows": rows,
        "three_pass_exercised": rows[-1]["pass_count"] == 3,
        "maximum_expected_configuration_exercised": rows[-1]["pass_count"] == 3,
        "checkpoint": verification,
        "continuation_comparison": continuation_checks,
        "temporary_checkpoint_removed": not temporary.exists(),
        "peak_allocated_vram_mb": max(row["peak_allocated_vram_mb"] for row in rows),
        "peak_reserved_vram_mb": max(row["peak_reserved_vram_mb"] for row in rows),
        "passed": all(continuation_checks.values())
        and rows[-1]["pass_count"] == 3
        and sha256(source_checkpoint) == SOURCE_SHA256
        and not temporary.exists(),
    }
    durable_json(Path(output) / "DISPOSABLE_SMOKE_REPORT.json", report)
    del model, optimizer, loader, reloaded, reloaded_optimizer, reloaded_loader
    gc.collect()
    torch.cuda.empty_cache()
    return report


def initial_geometry_shock(source_checkpoint, val_path, core_manifest,
                           shuffle_manifest, core_manifest_path,
                           shuffle_manifest_path, device, output):
    fixed, _, _, _, _ = d4a.load_fixed_source(
        source_checkpoint, device, restore=False
    )
    parent_model = make_fixed_evaluator_from_model(fixed)
    parent_identity = make_evaluation_identity(
        parent_model, "PARENT_FIXED_REAL", SOURCE_SHA256, 0,
        core_manifest_path, shuffle_manifest_path,
    )
    parent_eval = evaluate_incremental_panel(
        parent_model, val_path, core_manifest, ("all_real",), shuffle_manifest,
        "PARENT_FIXED_REAL", Path(output) / "INITIAL_PARENT_CORE_RAW.json",
        parent_identity,
    )
    del parent_model, fixed
    gc.collect()
    torch.cuda.empty_cache()
    c_model, _, _, _, _ = make_c_model(source_checkpoint, device, restore=False)
    c_identity = make_evaluation_identity(
        c_model, "C0", SOURCE_SHA256, 0,
        core_manifest_path, shuffle_manifest_path,
    )
    c_eval = evaluate_incremental_panel(
        c_model, val_path, core_manifest, CONTROLS, shuffle_manifest,
        "C0", Path(output) / "INITIAL_C0_CORE_RAW.json", c_identity,
    )
    parent_real = parent_eval["conditions"]["all_real"]["validation_loss"]
    real = c_eval["conditions"]["all_real"]["validation_loss"]
    conditions = c_eval["conditions"]
    result = {
        "parent_fixed_real_ce": parent_real,
        "c0_all_real_ce": real,
        "immediate_geometry_shock": real - parent_real,
        "c0_b3_recurrent_gain": conditions["b3_off"]["validation_loss"] - real,
        "c0_b3_sequence_gap": conditions["b3_shuffled"]["validation_loss"] - real,
        "c0_b5_recurrent_gain": conditions["b5_off"]["validation_loss"] - real,
        "c0_b5_sequence_gap": conditions["b5_shuffled"]["validation_loss"] - real,
        "c0_combined_recurrent_gain": conditions["b3_b5_off"]["validation_loss"] - real,
        "c0_combined_sequence_gap": conditions["b3_b5_shuffled"]["validation_loss"] - real,
        "core_sha256": c_eval["panel_sha256"],
        "targets_per_condition": conditions["all_real"]["validation_targets"],
        "paired_sequences": conditions["all_real"]["paired_sequences"],
        "parent_raw_artifact": "INITIAL_PARENT_CORE_RAW.json",
        "c0_raw_artifact": "INITIAL_C0_CORE_RAW.json",
        "trained_outcome": False,
        "passed": parent_eval["passed"] and c_eval["passed"]
        and c_eval["panel_sha256"] == CORE_SHA256,
    }
    durable_json(Path(output) / "INITIAL_GEOMETRY_SHOCK.json", result)
    del c_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    pretrain = Path(args.pretrain_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = read_json(pretrain / "PRETRAIN_FREEZE_AUDIT.json")
    if not frozen.get("passed"):
        raise SystemExit("pretraining freeze audit is not passed")
    prepared_environment = read_json(pretrain / "ENVIRONMENT_MANIFEST.json")
    current_head = git("rev-parse", "HEAD")
    source_path = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    model, optimizer, _, _, construction = make_c_model(source_path, device, restore=False)
    val_path = base.validation_path(Path(args.data_root))
    core_manifest = read_json(pretrain / "PANEL_MANIFEST_CORE.json")
    large_manifest = read_json(pretrain / "PANEL_MANIFEST_LARGE.json")
    shuffle_manifest = read_json(pretrain / "SHUFFLE_MANIFEST.json")
    ring_mapping = ring_index_mapping_test(model, device)
    tests = {
        "construction": {"checks": construction, "passed": all(construction.values())},
        "lag_coverage_nonoverlap": lag_coverage_test(model, device),
        "ring_index_mapping": ring_mapping,
        "causality": causality_test(model, device, ring_mapping),
        "incremental_cache_reload_equivalence": incremental_reload_equivalence(model, device),
        "cache_capacity_eviction": cache_capacity_test(model, device),
        "optimizer_name_rebinding": optimizer_rebinding_preflight_test(
            model, optimizer
        ),
        "attached_gradients": attached_gradient_test(model, optimizer, val_path, device),
        "control_specificity": control_specificity_test(model, device, shuffle_manifest),
        "forbidden_components": forbidden_component_audit(model),
        "representation_diagnostic_production_shape_feasibility": (
            representation_diagnostic_feasibility_smoke(
                model, optimizer, val_path, device
            )
        ),
    }
    tests["all_real_repeat_stable"] = {
        "passed": tests["control_specificity"]["all_real_repeat_exact"]
    }
    tests["passed"] = all(row.get("passed", False) for key, row in tests.items() if key != "passed")
    durable_json(output / "PREFLIGHT_TESTS.json", tests)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if not tests["passed"]:
        raise SystemExit("mandatory architecture/causality/cache preflight failed")
    smoke = disposable_smoke(source_path, val_path, pretrain, device, output)
    if not smoke["passed"]:
        raise SystemExit("disposable smoke failed")
    initial = initial_geometry_shock(
        source_path, val_path, core_manifest, shuffle_manifest,
        pretrain / "PANEL_MANIFEST_CORE.json",
        pretrain / "SHUFFLE_MANIFEST.json", device, output
    )
    checks = {
        "pretrain_freeze": frozen["passed"],
        "mandatory_tests": tests["passed"],
        "disposable_smoke": smoke["passed"],
        "initial_geometry_shock_complete": initial["passed"],
        "large_panel_frozen": large_manifest["all_required_disjointness_passed"],
        "one_a100_80gb": torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(device),
        "stop_capability_verified": bool(args.stop_capability_verified),
        "storage_inventory_verified": bool(args.storage_inventory_verified),
        "network_volume_free_ge_12gib": int(args.network_volume_free_bytes) >= 12 * 1024**3,
        "source_unchanged": sha256(source_path) == SOURCE_SHA256,
        "scope_exact": read_json(pretrain / "SCOPE_LOCK.json")["exactly_one_newly_trained_arm"] == ["C"],
        "git_clean_at_entry": True,
        "core_manifest_frozen_sha": sha256(pretrain / "PANEL_MANIFEST_CORE.json")
        == frozen["artifact_sha256"]["core_panel_manifest"],
        "large_manifest_frozen_sha": sha256(pretrain / "PANEL_MANIFEST_LARGE.json")
        == frozen["artifact_sha256"]["large_panel_manifest"],
        "shuffle_manifest_frozen_sha": sha256(pretrain / "SHUFFLE_MANIFEST.json")
        == frozen["artifact_sha256"]["shuffle_manifest"],
        "replay_ledger_frozen_sha": sha256(pretrain / "DATA_REPLAY_LEDGER.jsonl")
        == frozen["artifact_sha256"]["data_replay_ledger"],
        "replay_audit_frozen_sha": sha256(pretrain / "DATA_REPLAY_AUDIT.json")
        == frozen["artifact_sha256"]["data_replay_audit"],
        "implementation_files_unchanged_since_prepare": implementation_file_sha256()
        == prepared_environment["implementation_file_sha256"],
        "implementation_commit_pushed": git("rev-parse", f"origin/{BRANCH}")
        == current_head,
        "tooling_ancestor_preserved": git_is_ancestor(TOOLING_COMMIT),
    }
    audit = {
        "experiment": EXPERIMENT,
        "git_commit": current_head,
        "checks": checks,
        "network_volume_free_bytes": int(args.network_volume_free_bytes),
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "volume_id": args.volume_id,
        "frozen_artifact_sha256": frozen["artifact_sha256"],
        "implementation_file_sha256": implementation_file_sha256(),
        "authorized": all(checks.values()),
    }
    durable_json(output / "PREFLIGHT_AUDIT.json", audit)
    if not audit["authorized"]:
        raise SystemExit(f"2D5C preflight did not authorize training: {checks}")
    print("EXPERIMENT_2D5C_PREFLIGHT_PASS", flush=True)


def merge_json(path, key, value):
    path = Path(path)
    payload = read_json(path) if path.exists() else {}
    payload[str(key)] = value
    durable_json(path, payload)


def load_replay_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != LOCAL_UPDATES or [row["local_update"] for row in rows] != list(range(1, 192)):
        raise SystemExit("frozen replay ledger is not exactly local updates 1-191")
    if not artifacts.verify_replay_chain(rows, rows[-1]["chain_sha256"]):
        raise SystemExit("frozen replay ledger chain verification failed")
    return rows


def heartbeat(output, local_update, status, row=None, checkpoint=None):
    durable_json(Path(output) / "HEARTBEAT_C.json", {
        "experiment": EXPERIMENT,
        "arm": "C",
        "status": status,
        "local_update": int(local_update),
        "global_update": SOURCE_UPDATES + int(local_update),
        "new_targets": int(local_update) * base.GLOBAL_TARGETS,
        "cumulative_targets": SOURCE_TARGETS + int(local_update) * base.GLOBAL_TARGETS,
        "latest_metrics": row,
        "checkpoint": checkpoint,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
    })


def preexit_restart_record(model, optimizer, loader, payload_verification,
                           source_payload):
    return {
        "experiment": EXPERIMENT,
        "local_update": RESTART_LOCAL_UPDATE,
        "saved_process_id": os.getpid(),
        "checkpoint": payload_verification,
        "checkpoint_file_sha256": payload_verification["sha256"],
        "checkpoint_file_bytes": payload_verification["bytes"],
        "model_aggregate_sha256": parameter_manifest(model)["aggregate_sha256"],
        "optimizer_aggregate_sha256": optimizer_manifest(model, optimizer)["state_aggregate_sha256"],
        "optimizer_step_summary": sorted(set(optimizer_steps(optimizer))),
        "scheduler_state": copy.deepcopy(source_payload.get("scheduler")),
        "scheduler_sha256": canonical_sha(source_payload.get("scheduler")),
        "loader_state": loader.state_dict(),
        "rng_digests": rng_digests(base.capture_rng()),
        "next_batch_sha256": payload_verification["next_global_batch_sha256"],
        "next_stream_sha256": payload_verification["next_global_batch_stream_sha256"],
        "next_pass_count": pass_count_for_local(97),
        "restart_sentinel": payload_verification["restart_sentinel"],
        "status": "fresh_process_required",
    }


def midpoint_restart_audit(preexit_path, model, optimizer, loader, loaded,
                           checkpoint_path, source_checkpoint, source_payload,
                           device):
    before = read_json(preexit_path)
    checkpoint_path = Path(checkpoint_path).resolve()
    after_rng = rng_digests(base.capture_rng())
    sentinel = restart_sentinel(model)
    expected_logits = np.asarray(before["restart_sentinel"]["selected_logits"], dtype=np.float64)
    actual_logits = np.asarray(sentinel["selected_logits"], dtype=np.float64)
    sentinel_delta = float(np.max(np.abs(expected_logits - actual_logits)))
    checks = {
        "local_update": loaded.get("local_updates") == RESTART_LOCAL_UPDATE,
        "fresh_process": before["saved_process_id"] != os.getpid(),
        "saved_process_exact": loaded.get("saved_process_id")
        == before["saved_process_id"],
        "checkpoint_byte_sha256": sha256(checkpoint_path)
        == before["checkpoint_file_sha256"]
        == before["checkpoint"]["sha256"],
        "checkpoint_byte_count": checkpoint_path.stat().st_size
        == before["checkpoint_file_bytes"]
        == before["checkpoint"]["bytes"],
        "strict_reopen": strict_reopen(checkpoint_path, source_checkpoint, device)["passed"],
        "architecture": loaded.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C,
        "global_update": loaded.get("global_update")
        == SOURCE_UPDATES + RESTART_LOCAL_UPDATE,
        "cumulative_targets": loaded.get("cumulative_targets")
        == MILESTONE_TARGETS[RESTART_LOCAL_UPDATE],
        "model_digest": parameter_manifest(model)["aggregate_sha256"] == before["model_aggregate_sha256"],
        "optimizer_digest": optimizer_manifest(model, optimizer)["state_aggregate_sha256"] == before["optimizer_aggregate_sha256"],
        "optimizer_steps": sorted(set(optimizer_steps(optimizer))) == before["optimizer_step_summary"],
        "scheduler_exact_preexit": loaded.get("scheduler")
        == before["scheduler_state"],
        "scheduler_exact_source": loaded.get("scheduler")
        == source_payload.get("scheduler"),
        "scheduler_digest": canonical_sha(loaded.get("scheduler"))
        == before["scheduler_sha256"],
        "loader_state": loader.state_dict() == before["loader_state"],
        "rng_digests": after_rng == before["rng_digests"],
        "next_batch": base.next_batch_hash(loader, int(loaded["gradient_accumulation"])) == before["next_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, int(loaded["gradient_accumulation"])) == before["next_stream_sha256"],
        "next_pass_count": pass_count_for_local(97) == before["next_pass_count"],
        "sentinel_tolerance": sentinel_delta <= 1e-6,
        "sentinel_exact": sentinel["selected_logits_sha256"] == before["restart_sentinel"]["selected_logits_sha256"],
    }
    return {
        "pre_exit": before,
        "resumed_process_id": os.getpid(),
        "checkpoint_file_sha256": sha256(checkpoint_path),
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "scheduler_sha256": canonical_sha(loaded.get("scheduler")),
        "sentinel_max_abs": sentinel_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_train(args):
    require_branch(clean=True)
    if args.arm != "C":
        raise SystemExit("2D5C can train only arm C")
    end = int(args.end_local_update)
    if end not in (96, 191):
        raise SystemExit("2D5C training may end only at local update 96 or 191")
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = read_json(args.preflight_audit)
    if not preflight.get("authorized"):
        raise SystemExit("2D5C preflight has not authorized scientific training")
    current_head = git("rev-parse", "HEAD")
    code_binding = {
        "preflight_commit_is_head": preflight.get("git_commit") == current_head,
        "head_is_pushed_origin_branch": git("rev-parse", f"origin/{BRANCH}")
        == current_head,
        "implementation_file_sha256": preflight.get("implementation_file_sha256")
        == implementation_file_sha256(),
    }
    if not all(code_binding.values()):
        raise SystemExit(f"official training code/preflight binding failed: {code_binding}")
    frozen_hashes = preflight.get("frozen_artifact_sha256", {})
    binding_checks = {
        "replay_ledger": sha256(args.replay_ledger)
        == frozen_hashes.get("data_replay_ledger"),
        "replay_audit": sha256(args.replay_audit)
        == frozen_hashes.get("data_replay_audit"),
        "large_panel": sha256(args.large_panel)
        == frozen_hashes.get("large_panel_manifest"),
    }
    if not all(binding_checks.values()):
        raise SystemExit(f"official training artifact binding failed: {binding_checks}")
    replay_rows = load_replay_rows(args.replay_ledger)
    replay_audit = read_json(args.replay_audit)
    if replay_audit.get("ledger_sha256") != sha256(args.replay_ledger) or not replay_audit.get("passed"):
        raise SystemExit("frozen replay ledger identity mismatch")
    source_path = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    if args.resume_checkpoint:
        model, optimizer, loader, loaded, source = load_c_checkpoint(
            args.resume_checkpoint, source_path, device, restore=True
        )
        start = int(loaded["local_updates"])
        if (start, end) != (96, 191):
            raise SystemExit(f"unauthorized resume segment {start}->{end}")
        restart = midpoint_restart_audit(
            args.midpoint_preexit, model, optimizer, loader, loaded,
            args.resume_checkpoint, source_path, source, device,
        )
        durable_json(output / "MIDPOINT_RESTART_AUDIT.json", restart)
        if not restart["passed"]:
            raise SystemExit(f"mandatory midpoint restart failed: {restart['checks']}")
        metadata = loaded["metadata"]
    else:
        if end != 96:
            raise SystemExit("official C training must begin with segment 0->96")
        model, optimizer, loader, source, source_checks = make_c_model(
            source_path, device, restore=True
        )
        if not all(source_checks.values()):
            raise SystemExit("source failed validation at official train start")
        start = 0
        panel = read_json(args.large_panel)
        metadata = continuation_metadata(source, panel, replay_audit)
    accumulation = int(source["gradient_accumulation"])
    if accumulation != 16 or int(source["loader_state"]["batch_size"]) != 32:
        raise SystemExit("recovered single-GPU Fixed recipe is not microbatch 32 / accumulation 16")
    metric_path = output / "TRAINING_LOG.jsonl"
    actual_ledger_path = output / "TRAINING_REPLAY_ACTUAL.jsonl"
    existing_metrics = [
        json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()
    ] if metric_path.exists() else []
    if len(existing_metrics) != start:
        raise SystemExit(f"training log rows {len(existing_metrics)} do not match start {start}")
    for local_update in range(start + 1, end + 1):
        if local_update > LOCAL_UPDATES:
            raise SystemExit("hard local-update limit 191 reached")
        expected = replay_rows[local_update - 1]
        current_batch = base.next_batch_hash(loader, accumulation)
        current_stream = base.next_stream_hash(loader, accumulation)
        expected_pass = pass_count_for_local(local_update)
        invariants = {
            "local_update": expected["local_update"] == local_update,
            "global_update": expected["global_update"] == SOURCE_UPDATES + local_update,
            "start_cursor": loader.state_dict() == expected["start_cursor"],
            "batch_hash": current_batch == expected["logical_global_batch_sha256"],
            "stream_hash": current_stream == expected["logical_global_stream_sha256"],
            "pass_count": expected["pass_count"] == expected_pass,
            "target_count": expected["target_count"] == base.GLOBAL_TARGETS,
        }
        if not all(invariants.values()):
            raise SystemExit(f"pre-forward replay invariant failed at update {local_update}: {invariants}")
        row = smoke_update(
            model, optimizer, loader, accumulation, local_update, device
        )
        row.update({
            "train_ce_nll": row["pass_losses"][-1],
            "process_id": os.getpid(),
            "optimizer_lrs": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
            "recurrent_gate_summary": base.gate_values(model),
            "optimizer_step_success": True,
            "pre_forward_invariants": invariants,
            "expected_replay_chain_sha256": expected["chain_sha256"],
            "end_cursor_exact": loader.state_dict() == expected["end_cursor"],
            "ce_only": True,
        })
        if row["pass_count"] != expected_pass or not row["end_cursor_exact"]:
            raise SystemExit(f"post-update cadence/cursor failure at {local_update}")
        append_jsonl(metric_path, row)
        append_jsonl(actual_ledger_path, {
            "local_update": local_update,
            "global_update": SOURCE_UPDATES + local_update,
            "batch_sha256": current_batch,
            "stream_sha256": current_stream,
            "pass_count": expected_pass,
            "target_count": base.GLOBAL_TARGETS,
            "start_cursor": expected["start_cursor"],
            "end_cursor": loader.state_dict(),
            "frozen_chain_sha256": expected["chain_sha256"],
            "process_id": os.getpid(),
        })
        heartbeat(output, local_update, "training", row=row)
        if local_update in MILESTONES:
            checkpoint_path = Path(args.scientific_checkpoint_dir) / checkpoint_name(local_update)
            verification = save_checkpoint(
                checkpoint_path, model, optimizer, loader, source, local_update,
                accumulation, metadata, replay_audit["ledger_sha256"], source_path,
                device,
            )
            if local_update == LOCAL_UPDATES:
                if verification["next_global_batch_sha256"] != CONTROL_NEXT_BATCH or verification["next_global_batch_stream_sha256"] != CONTROL_NEXT_STREAM:
                    raise SystemExit("final C loader state does not match Fixed-100M terminal state")
            merge_json(output / "MILESTONE_CHECKPOINTS.json", local_update, verification)
            heartbeat(output, local_update, "checkpoint_verified", row=row, checkpoint=verification)
            if local_update == RESTART_LOCAL_UPDATE:
                before = preexit_restart_record(
                    model, optimizer, loader, verification, source
                )
                durable_json(output / "MIDPOINT_RESTART_PREEXIT.json", before)
    if end == RESTART_LOCAL_UPDATE:
        heartbeat(output, end, "fresh_process_restart_required")
        print("EXPERIMENT_2D5C_SEGMENT_COMPLETE 0->96 FRESH_PROCESS_REQUIRED", flush=True)
        return
    metrics = [json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
    actual = [json.loads(line) for line in actual_ledger_path.read_text().splitlines() if line.strip()]
    milestones = read_json(output / "MILESTONE_CHECKPOINTS.json")
    restart_preexit = read_json(output / "MIDPOINT_RESTART_PREEXIT.json")
    restart_audit = read_json(output / "MIDPOINT_RESTART_AUDIT.json")
    final_milestone = milestones.get(str(LOCAL_UPDATES), {})
    first_processes = {row.get("process_id") for row in metrics[:RESTART_LOCAL_UPDATE]}
    second_processes = {row.get("process_id") for row in metrics[RESTART_LOCAL_UPDATE:]}
    final_optimizer_manifest = optimizer_manifest(model, optimizer)
    final_model_manifest = parameter_manifest(model)
    checks = {
        "updates_exact": len(metrics) == LOCAL_UPDATES and [row["local_update"] for row in metrics] == list(range(1, 192)),
        "optimizer_updates_exact": sum(row["optimizer_steps"] for row in metrics) == LOCAL_UPDATES,
        "no_skipped_updates": all(row["optimizer_step_success"] for row in metrics),
        "targets_exact": len(metrics) * base.GLOBAL_TARGETS == LOCAL_TARGETS,
        "final_global_update": metrics[-1]["global_update"] == FINAL_GLOBAL_UPDATE,
        "final_cumulative_targets": metrics[-1]["cumulative_targets"] == FINAL_CUMULATIVE_TARGETS,
        "batch_replay_exact": [row["batch_sha256"] for row in actual]
        == [row["logical_global_batch_sha256"] for row in replay_rows],
        "stream_replay_exact": [row["stream_sha256"] for row in actual]
        == [row["logical_global_stream_sha256"] for row in replay_rows],
        "pass_cadence_exact": [row["pass_count"] for row in actual]
        == [row["pass_count"] for row in replay_rows],
        "actual_replay_rows_exact": len(actual) == LOCAL_UPDATES
        and [row["local_update"] for row in actual] == list(range(1, LOCAL_UPDATES + 1)),
        "replay_chain_exact": [row["frozen_chain_sha256"] for row in actual]
        == [row["chain_sha256"] for row in replay_rows],
        "midpoint_restart": restart_audit["passed"],
        "two_exact_training_processes": len(first_processes) == 1
        and len(second_processes) == 1
        and first_processes != second_processes
        and None not in first_processes | second_processes,
        "preexit_process_exact": first_processes
        == {restart_preexit["saved_process_id"]},
        "resume_process_exact": second_processes
        == {restart_audit["resumed_process_id"]}
        == {final_milestone.get("saved_process_id")}
        == {os.getpid()},
        "optimizer_terminal_step_exact": sorted(set(optimizer_steps(optimizer)))
        == [FINAL_GLOBAL_UPDATE]
        == metrics[-1]["optimizer_steps_after_summary"],
        "optimizer_terminal_digest_exact": final_milestone.get("optimizer_state_sha256")
        == final_optimizer_manifest["state_aggregate_sha256"],
        "model_terminal_digest_exact": final_milestone.get("model_state_sha256")
        == final_model_manifest["aggregate_sha256"],
        "scheduler_exact_source": final_milestone.get("scheduler")
        == source.get("scheduler"),
        "all_milestones_exact": set(milestones)
        == {str(update) for update in MILESTONES},
        "final_milestone_checkpoint_sha_exact": final_milestone.get("sha256")
        == sha256(final_milestone["checkpoint"]),
        "fixed_control_optimizer_steps": 0 == 0,
        "no_training_beyond": max(row["local_update"] for row in metrics) == LOCAL_UPDATES,
    }
    artifact_identity = {
        "training_log": file_identity(metric_path),
        "training_replay_actual": file_identity(actual_ledger_path),
        "replay_ledger": file_identity(args.replay_ledger),
        "replay_audit": file_identity(args.replay_audit),
        "milestone_manifest": file_identity(output / "MILESTONE_CHECKPOINTS.json"),
        "midpoint_restart_preexit": file_identity(output / "MIDPOINT_RESTART_PREEXIT.json"),
        "midpoint_restart_audit": file_identity(output / "MIDPOINT_RESTART_AUDIT.json"),
        "final_checkpoint": file_identity(final_milestone["checkpoint"]),
    }
    durable_json(output / "TRAINING_COMPLETE.json", {
        "schema": "experiment_2d5c_training_complete_v1",
        "experiment": EXPERIMENT,
        "checks": checks,
        "artifact_identity": artifact_identity,
        "optimizer_evidence": {
            "step_summary": sorted(set(optimizer_steps(optimizer))),
            "state_sha256": final_optimizer_manifest["state_aggregate_sha256"],
            "model_state_sha256": final_model_manifest["aggregate_sha256"],
        },
        "process_evidence": {
            "pre_restart_process_id": restart_preexit["saved_process_id"],
            "post_restart_process_id": restart_audit["resumed_process_id"],
            "final_checkpoint_saved_process_id": final_milestone["saved_process_id"],
            "fresh_process_boundary": first_processes != second_processes,
        },
        "scheduler_evidence": {
            "state": copy.deepcopy(source.get("scheduler")),
            "sha256": canonical_sha(source.get("scheduler")),
        },
        "replay_evidence": {
            "rows": len(actual),
            "frozen_ledger_sha256": replay_audit["ledger_sha256"],
            "terminal_chain_sha256": replay_rows[-1]["chain_sha256"],
        },
        "training_wall_seconds": sum(row["wall_seconds"] for row in metrics),
        "mean_targets_per_second": statistics.fmean(row["targets_per_second"] for row in metrics),
        "passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise SystemExit(f"terminal training audit failed: {checks}")
    heartbeat(output, LOCAL_UPDATES, "training_complete")
    print("EXPERIMENT_2D5C_TRAINING_COMPLETE 191 UPDATES", flush=True)


def load_scientific_evaluator(family, checkpoint, source_checkpoint, device,
                              expected_local_update=None, milestone_manifest=None,
                              final_seal=None):
    """Load one evaluation family without restoring or advancing saved RNG."""
    source_checkpoint = require_exact_file(
        source_checkpoint, SOURCE_SHA256, "2D3A-1B source checkpoint"
    )
    optimizer_rebinding = None
    if family == "C0":
        if checkpoint:
            raise SystemExit("C0 is loaded only from the accepted 2D3A source")
        model, optimizer, loader, payload, checks = make_c_model(
            source_checkpoint, device, restore=False
        )
        local_update = 0
        checkpoint_sha = SOURCE_SHA256
        del optimizer, loader
    elif family == "C":
        if not checkpoint:
            raise SystemExit("a sealed C checkpoint is required")
        if not milestone_manifest:
            raise SystemExit("C evaluation requires the training milestone manifest")
        checkpoint = Path(checkpoint).resolve()
        binding = verify_c_checkpoint_binding(
            checkpoint, expected_local_update, milestone_manifest, final_seal
        )
        model, optimizer, loader, payload, _ = load_c_checkpoint(
            checkpoint, source_checkpoint, device, restore=False
        )
        local_update = int(payload["local_updates"])
        checks = strict_reopen(checkpoint, source_checkpoint, device)["checks"]
        checkpoint_sha = sha256(checkpoint)
        del optimizer, loader
    elif family == "Fixed":
        if not checkpoint:
            raise SystemExit("the exact Fixed-100M checkpoint is required")
        (
            model,
            optimizer,
            loader,
            payload,
            checks,
            optimizer_rebinding,
        ) = load_fixed_control(checkpoint, source_checkpoint, device)
        local_update = LOCAL_UPDATES
        checkpoint_sha = CONTROL_SHA256
        del optimizer, loader
    elif family == "Parent":
        if checkpoint:
            raise SystemExit("Parent is loaded only from the accepted source")
        fixed, optimizer, loader, payload, parent_checks = d4a.load_fixed_source(
            source_checkpoint, device, restore=False
        )
        model = make_fixed_evaluator_from_model(fixed)
        checks = parent_checks
        local_update = 0
        checkpoint_sha = SOURCE_SHA256
        del fixed, optimizer, loader
    else:
        raise SystemExit(f"unknown evaluation family {family}")
    if expected_local_update is not None and local_update != int(expected_local_update):
        raise SystemExit(
            f"{family} checkpoint local update {local_update} != expected "
            f"{expected_local_update}"
        )
    expected_fingerprint = (
        ARCHITECTURE_FINGERPRINT_C if family in ("C", "C0")
        else ARCHITECTURE_FINGERPRINT_FIXED
    )
    load_checks = {
        "construction": all(checks.values()),
        "checkpoint_sha256": checkpoint_sha,
        "architecture_fingerprint": model.architecture_fingerprint(),
        "architecture_fingerprint_exact": (
            model.architecture_fingerprint() == expected_fingerprint
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_count_exact": sum(p.numel() for p in model.parameters()) == PARAMETERS,
        "local_update": local_update,
        "rng_not_restored": True,
        "optimizer_steps_in_evaluator": 0,
        "optimizer_rebinding": optimizer_rebinding,
    }
    if family == "C":
        load_checks["checkpoint_binding"] = binding
    if not (
        load_checks["construction"]
        and load_checks["architecture_fingerprint_exact"]
        and load_checks["parameter_count_exact"]
    ):
        raise SystemExit(f"scientific evaluator load failed: {load_checks}")
    model.eval()
    return model, payload, load_checks


def parallel_control_kwargs(control, permutation):
    off, shuffled = core.FixedWriterW2PressureGPT.control_sets(control)
    kwargs = {}
    if 2 in off:
        kwargs["b3_gate_override"] = 0.0
    if 4 in off:
        kwargs["b5_gate_override"] = 0.0
    if 2 in shuffled:
        kwargs["b3_recurrent_permutation"] = permutation
    if 4 in shuffled:
        kwargs["b5_recurrent_permutation"] = permutation
    return kwargs


def evaluate_parallel_panel(model, val_path, panel, controls, shuffle_manifest,
                            family, output_path, evaluation_identity,
                            num_passes=2):
    """Secondary, explicitly non-primary path-matched multi-pass evaluation."""
    output_path = Path(output_path)
    if output_path.exists():
        previous = read_json(output_path)
        if previous.get("evaluation_identity") != evaluation_identity:
            raise SystemExit("parallel evaluation checkpoint/panel/shuffle identity mismatch")
        if previous.get("status") == "complete":
            return previous
        raise SystemExit("partial parallel evaluation is not resumable; use a fresh path")
    device = base.model_device(model)
    permutation = permutation_for_batch(shuffle_manifest, base.VALIDATION_B, device)
    state = {
        "experiment": EXPERIMENT,
        "family": family,
        "evaluation_identity": evaluation_identity,
        "evaluation_mode": f"path_matched_parallel_{int(num_passes)}pass",
        "primary": False,
        "warning": "Never compare these values to true-incremental values.",
        "panel_sha256": panel.get("panel_sha256", panel.get("subset_sha256")),
        "controls_requested": list(controls),
        "conditions": {
            name: {"nll_sum": 0.0, "targets": 0, "per_sequence_nll": [],
                   "per_sequence_ce": []}
            for name in controls
        },
        "batch_identities": [],
        "status": "running",
    }
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for ordinal, batch_index in enumerate(panel_batch_indices(panel)):
            cpu_x, cpu_y = batch_at_index(val_path, batch_index)
            identity = base.batch_identity(cpu_x, cpu_y)
            expected = panel.get("batch_identities", [])[ordinal]
            if expected and identity != expected:
                raise SystemExit("parallel panel identity mismatch")
            x, y = cpu_x.to(device), cpu_y.to(device)
            for control in controls:
                kwargs = parallel_control_kwargs(control, permutation)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.forward_multi_pass(
                        x, targets=None, num_passes=int(num_passes), **kwargs
                    )
                logits = result["logits"].float()
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    reduction="none",
                ).view(y.size(0), -1).double().sum(dim=1).cpu()
                row = state["conditions"][control]
                row["nll_sum"] += losses.sum().item()
                row["targets"] += int(y.numel())
                row["per_sequence_nll"].extend(losses.tolist())
                row["per_sequence_ce"].extend((losses / base.T).tolist())
            state["batch_identities"].append(identity)
            durable_json(output_path, state)
            del x, y, cpu_x, cpu_y
            torch.cuda.empty_cache()
    for row in state["conditions"].values():
        row["validation_loss"] = row["nll_sum"] / row["targets"]
        row["validation_targets"] = row["targets"]
        row["paired_sequences"] = len(row["per_sequence_nll"])
    state["wall_seconds"] = time.monotonic() - started
    state["same_sequence_order_all_conditions"] = True
    state["status"] = "complete"
    state["passed"] = all(
        row["targets"] == state["conditions"]["all_real"]["targets"]
        for row in state["conditions"].values()
    )
    durable_json(output_path, state)
    return state


def summarize_mechanism_conditions(evaluation):
    rows = evaluation["conditions"]
    real = rows["all_real"]["validation_loss"]
    summary = {
        "all_real_ce": real,
        "b3_recurrent_gain": rows["b3_off"]["validation_loss"] - real,
        "b3_sequence_gap": rows["b3_shuffled"]["validation_loss"] - real,
        "b5_recurrent_gain": rows["b5_off"]["validation_loss"] - real,
        "b5_sequence_gap": rows["b5_shuffled"]["validation_loss"] - real,
        "combined_recurrent_gain": rows["b3_b5_off"]["validation_loss"] - real,
        "combined_sequence_gap": rows["b3_b5_shuffled"]["validation_loss"] - real,
    }
    summary["off_interaction_redundancy_synergy"] = (
        summary["combined_recurrent_gain"]
        - summary["b3_recurrent_gain"] - summary["b5_recurrent_gain"]
    )
    summary["shuffled_interaction_redundancy_synergy"] = (
        summary["combined_sequence_gap"]
        - summary["b3_sequence_gap"] - summary["b5_sequence_gap"]
    )
    summary.update({
        "paired_sequences": rows["all_real"]["paired_sequences"],
        "targets_per_condition": rows["all_real"]["validation_targets"],
        "panel_sha256": evaluation["panel_sha256"],
        "primary_true_incremental": True,
    })
    return summary


def evaluation_protocol_checks(family, panel_kind, expected_local_update,
                               all_real_only, parallel_output,
                               final_checkpoint_seal, milestone_manifest):
    """Fail-closed protocol matrix for every scientific evaluation family."""
    update = int(expected_local_update)
    parallel_requested = bool(parallel_output)
    required_parallel = (
        panel_kind == "core"
        and ((family == "C0" and update == 0)
             or (family == "C" and update in (96, LOCAL_UPDATES)))
    )
    valid_location = {
        "Parent": panel_kind == "core" and update == 0,
        "C0": panel_kind == "core" and update == 0,
        "C": update in MILESTONES
        and (panel_kind == "core" or (panel_kind == "large" and update == LOCAL_UPDATES)),
        "Fixed": update == LOCAL_UPDATES and panel_kind in ("core", "large"),
    }.get(family, False)
    checks = {
        "family_panel_update_allowed": valid_location,
        "all_real_only_parent_only": not all_real_only or family == "Parent",
        "parent_all_real_only": family != "Parent" or all_real_only,
        "full_controls_non_parent": family == "Parent" or not all_real_only,
        "parallel_core_only": not parallel_requested or panel_kind == "core",
        "required_parallel_present": not required_parallel or parallel_requested,
        "c_milestone_manifest_present": family != "C" or bool(milestone_manifest),
        "non_c_has_no_milestone_manifest": family == "C" or not milestone_manifest,
        "c191_final_seal_present": family != "C" or update != LOCAL_UPDATES
        or bool(final_checkpoint_seal),
        "final_seal_applies_only_c191": not final_checkpoint_seal
        or (family == "C" and update == LOCAL_UPDATES),
    }
    return {
        "family": family,
        "panel_kind": panel_kind,
        "expected_local_update": update,
        "required_parallel": required_parallel,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_evaluate(args):
    require_branch(clean=True)
    protocol = evaluation_protocol_checks(
        args.family, args.panel_kind, args.expected_local_update,
        args.all_real_only, args.parallel_output, args.final_checkpoint_seal,
        args.milestone_manifest,
    )
    if not protocol["passed"]:
        raise SystemExit(f"evaluation protocol matrix failed: {protocol['checks']}")
    device = base.require_a100()
    freeze = read_json(args.pretrain_freeze_audit)
    if not freeze.get("passed"):
        raise SystemExit("pretraining freeze audit is not passed")
    model, payload, load_audit = load_scientific_evaluator(
        args.family, args.checkpoint, args.source_checkpoint, device,
        args.expected_local_update, args.milestone_manifest,
        args.final_checkpoint_seal,
    )
    panel = read_json(args.panel_manifest)
    shuffle = read_json(args.shuffle_manifest)
    frozen_hashes = freeze["artifact_sha256"]
    panel_hash_key = (
        "large_panel_manifest" if args.panel_kind == "large"
        else "core_panel_manifest"
    )
    frozen_checks = {
        "panel_file_sha256": sha256(args.panel_manifest)
        == frozen_hashes[panel_hash_key],
        "shuffle_file_sha256": sha256(args.shuffle_manifest)
        == frozen_hashes["shuffle_manifest"],
        "shuffle_seed": shuffle.get("seed") == SHUFFLE_SEED,
        "shuffle_passed": shuffle.get("passed") is True,
        "core_intrinsic_sha": args.panel_kind != "core"
        or panel.get("subset_sha256") == CORE_SHA256,
        "large_intrinsic_sha": args.panel_kind != "large"
        or isinstance(panel.get("panel_sha256"), str),
        "c_checkpoint_large_panel": args.family != "C"
        or args.panel_kind != "large"
        or payload.get("metadata", {}).get("large_panel_sha256")
        == panel.get("panel_sha256"),
    }
    if not all(frozen_checks.values()):
        raise SystemExit(f"evaluation frozen-artifact binding failed: {frozen_checks}")
    controls = ("all_real",) if args.all_real_only else CONTROLS
    output_path = Path(args.output_path).resolve()
    if args.panel_kind == "large" and args.family == "C":
        seal = read_json(args.final_checkpoint_seal)
        actual = sha256(Path(args.checkpoint))
        if not (
            seal.get("sealed")
            and seal.get("checkpoint_sha256") == actual
            and seal.get("local_update") == LOCAL_UPDATES
        ):
            raise SystemExit("large-panel C evaluation requires the matching final seal")
    evaluation_identity = make_evaluation_identity(
        model, args.family, load_audit["checkpoint_sha256"],
        load_audit["local_update"], args.panel_manifest,
        args.shuffle_manifest,
    )
    result = evaluate_incremental_panel(
        model, base.validation_path(Path(args.data_root)), panel, controls, shuffle,
        args.family, output_path, evaluation_identity, resume=True,
    )
    audit = {
        "load": load_audit,
        "evaluation_passed": result["passed"],
        "panel_kind": args.panel_kind,
        "controls_exact": result["controls_requested"] == list(controls),
        "targets_per_condition": result["conditions"]["all_real"]["validation_targets"],
        "paired_sequences": result["conditions"]["all_real"]["paired_sequences"],
        "frozen_artifact_checks": frozen_checks,
        "evaluation_identity": evaluation_identity,
        "protocol_requirements": protocol,
    }
    audit["passed"] = (
        audit["evaluation_passed"] and audit["controls_exact"]
        and all(frozen_checks.values()) and protocol["passed"]
        and result.get("evaluation_identity") == evaluation_identity
        and audit["targets_per_condition"]
        == (2_097_152 if args.panel_kind == "large" else 262_144)
        and audit["paired_sequences"]
        == (2_048 if args.panel_kind == "large" else 256)
    )
    durable_json(output_path.with_suffix(".audit.json"), audit)
    if not audit["passed"]:
        raise SystemExit(f"scientific evaluation audit failed: {audit}")
    if len(controls) == len(CONTROLS):
        durable_json(output_path.with_suffix(".summary.json"),
                     summarize_mechanism_conditions(result))
    if args.parallel_output:
        parallel = evaluate_parallel_panel(
            model, base.validation_path(Path(args.data_root)), panel, controls,
            shuffle, args.family, args.parallel_output, evaluation_identity,
            args.parallel_passes,
        )
        parallel_audit = {
            "experiment": EXPERIMENT,
            "family": args.family,
            "expected_local_update": int(args.expected_local_update),
            "evaluation_identity": evaluation_identity,
            "required_by_protocol": protocol["required_parallel"],
            "checks": {
                "passed": parallel.get("passed") is True,
                "complete": parallel.get("status") == "complete",
                "core_panel": args.panel_kind == "core",
                "path_matched_mode": parallel.get("evaluation_mode")
                == f"path_matched_parallel_{int(args.parallel_passes)}pass",
                "controls_exact": parallel.get("controls_requested") == list(controls),
                "targets_exact": all(
                    row.get("validation_targets") == 262_144
                    for row in parallel.get("conditions", {}).values()
                ),
                "paired_sequences_exact": all(
                    row.get("paired_sequences") == 256
                    for row in parallel.get("conditions", {}).values()
                ),
                "identity_exact": parallel.get("evaluation_identity")
                == evaluation_identity,
            },
        }
        parallel_audit["passed"] = all(parallel_audit["checks"].values())
        durable_json(Path(args.parallel_output).with_suffix(".audit.json"), parallel_audit)
        if not parallel_audit["passed"]:
            raise SystemExit(f"parallel evaluation audit failed: {parallel_audit}")
    del model, payload
    gc.collect()
    torch.cuda.empty_cache()
    print(f"EXPERIMENT_2D5C_EVALUATION_COMPLETE {args.family} {args.panel_kind}", flush=True)


def run_seal_final(args):
    require_branch(clean=True)
    device = base.require_a100()
    checkpoint = Path(args.checkpoint).resolve()
    binding = verify_c_checkpoint_binding(
        checkpoint, LOCAL_UPDATES, args.milestone_manifest
    )
    model, optimizer, loader, payload, source = load_c_checkpoint(
        checkpoint,
        require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source"),
        device,
        restore=False,
    )
    reopen = strict_reopen(checkpoint, args.source_checkpoint, device)
    training = read_json(args.training_complete)
    training_rows = [
        json.loads(line) for line in Path(args.training_log).read_text().splitlines()
        if line.strip()
    ]
    actual_rows = [
        json.loads(line)
        for line in Path(args.training_replay_actual).read_text().splitlines()
        if line.strip()
    ]
    replay_rows = load_replay_rows(args.replay_ledger)
    replay_audit = read_json(args.replay_audit)
    restart_preexit = read_json(args.midpoint_restart_preexit)
    restart = read_json(args.midpoint_restart_audit)
    milestones = read_json(args.milestone_manifest)
    final_milestone = milestones.get(str(LOCAL_UPDATES), {})
    optimizer_evidence = {
        "step_summary": sorted(set(optimizer_steps(optimizer))),
        "state_sha256": optimizer_manifest(model, optimizer)["state_aggregate_sha256"],
        "model_state_sha256": parameter_manifest(model)["aggregate_sha256"],
    }
    artifact_identity = {
        "checkpoint": file_identity(checkpoint),
        "checkpoint_verification_sidecar": file_identity(
            checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
        ),
        "checkpoint_checksum_sidecar": file_identity(
            checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        ),
        "milestone_manifest": file_identity(args.milestone_manifest),
        "training_log": file_identity(args.training_log),
        "training_replay_actual": file_identity(args.training_replay_actual),
        "training_complete": file_identity(args.training_complete),
        "replay_ledger": file_identity(args.replay_ledger),
        "replay_audit": file_identity(args.replay_audit),
        "midpoint_restart_preexit": file_identity(args.midpoint_restart_preexit),
        "midpoint_restart_audit": file_identity(args.midpoint_restart_audit),
    }
    training_artifacts = training.get("artifact_identity", {})
    expected_training_artifact_keys = {
        "training_log", "training_replay_actual", "replay_ledger", "replay_audit",
        "milestone_manifest", "midpoint_restart_preexit",
        "midpoint_restart_audit", "final_checkpoint",
    }
    training_identity_checks = {
        name: training_artifacts.get(name) == (
            artifact_identity["checkpoint"] if name == "final_checkpoint"
            else artifact_identity[name]
        )
        for name in expected_training_artifact_keys
    }
    first_processes = {
        row.get("process_id") for row in training_rows[:RESTART_LOCAL_UPDATE]
    }
    second_processes = {
        row.get("process_id") for row in training_rows[RESTART_LOCAL_UPDATE:]
    }
    checks = {
        "milestone_binding": binding["passed"],
        "milestone_manifest_exact": set(milestones)
        == {str(update) for update in MILESTONES},
        "final_milestone_exact_checkpoint": final_milestone.get("sha256")
        == artifact_identity["checkpoint"]["sha256"]
        and final_milestone.get("bytes") == artifact_identity["checkpoint"]["bytes"],
        "strict_reopen": reopen["passed"],
        "strict_reopen_new_process": payload.get("saved_process_id") != os.getpid(),
        "local_update": payload.get("local_updates") == LOCAL_UPDATES,
        "new_targets": payload.get("new_targets") == LOCAL_TARGETS,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_CUMULATIVE_TARGETS,
        "training_complete_schema": training.get("schema")
        == "experiment_2d5c_training_complete_v1",
        "training_complete": training.get("passed") is True
        and all(training.get("checks", {}).values()),
        "training_artifact_keys_exact": set(training_artifacts)
        == expected_training_artifact_keys,
        "training_artifact_identity_exact": all(training_identity_checks.values()),
        "training_rows_exact": len(training_rows) == LOCAL_UPDATES
        and [row.get("local_update") for row in training_rows]
        == list(range(1, LOCAL_UPDATES + 1)),
        "actual_replay_rows_exact": len(actual_rows) == LOCAL_UPDATES
        and [row.get("local_update") for row in actual_rows]
        == list(range(1, LOCAL_UPDATES + 1)),
        "batch_replay_exact": [row.get("batch_sha256") for row in actual_rows]
        == [row["logical_global_batch_sha256"] for row in replay_rows],
        "stream_replay_exact": [row.get("stream_sha256") for row in actual_rows]
        == [row["logical_global_stream_sha256"] for row in replay_rows],
        "pass_replay_exact": [row.get("pass_count") for row in actual_rows]
        == [row["pass_count"] for row in replay_rows],
        "chain_replay_exact": [row.get("frozen_chain_sha256") for row in actual_rows]
        == [row["chain_sha256"] for row in replay_rows],
        "replay_audit_exact": replay_audit.get("passed") is True
        and replay_audit.get("rows") == LOCAL_UPDATES
        and replay_audit.get("ledger_sha256")
        == artifact_identity["replay_ledger"]["sha256"]
        == payload.get("replay_ledger_sha256"),
        "next_batch": payload.get("next_global_batch_sha256") == CONTROL_NEXT_BATCH,
        "next_stream": payload.get("next_global_batch_stream_sha256") == CONTROL_NEXT_STREAM,
        "parameter_count": sum(p.numel() for p in model.parameters()) == PARAMETERS,
        "architecture": payload.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C,
        "optimizer_step_exact": optimizer_evidence["step_summary"]
        == [FINAL_GLOBAL_UPDATE]
        == training_rows[-1].get("optimizer_steps_after_summary"),
        "optimizer_state_exact": optimizer_evidence
        == training.get("optimizer_evidence")
        and optimizer_evidence["state_sha256"]
        == final_milestone.get("optimizer_state_sha256"),
        "scheduler_exact": payload.get("scheduler") == source.get("scheduler")
        == training.get("scheduler_evidence", {}).get("state")
        == restart_preexit.get("scheduler_state"),
        "scheduler_digest_exact": canonical_sha(payload.get("scheduler"))
        == training.get("scheduler_evidence", {}).get("sha256")
        == restart_preexit.get("scheduler_sha256")
        == restart.get("scheduler_sha256"),
        "midpoint_restart_exact": restart.get("passed") is True
        and restart.get("pre_exit") == restart_preexit
        and restart.get("checkpoint_file_sha256")
        == restart_preexit.get("checkpoint_file_sha256"),
        "training_process_partition_exact": len(first_processes) == 1
        and len(second_processes) == 1
        and first_processes != second_processes
        and None not in first_processes | second_processes,
        "process_evidence_exact": first_processes
        == {restart_preexit.get("saved_process_id")}
        and second_processes
        == {restart.get("resumed_process_id")}
        == {payload.get("saved_process_id")}
        and training.get("process_evidence", {}).get("pre_restart_process_id")
        == restart_preexit.get("saved_process_id")
        and training.get("process_evidence", {}).get("post_restart_process_id")
        == restart.get("resumed_process_id"),
    }
    result = {
        "schema": "experiment_2d5c_final_checkpoint_provenance_v1",
        "experiment": EXPERIMENT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": artifact_identity["checkpoint"]["sha256"],
        "checkpoint_bytes": artifact_identity["checkpoint"]["bytes"],
        "local_update": LOCAL_UPDATES,
        "global_update": FINAL_GLOBAL_UPDATE,
        "cumulative_targets": FINAL_CUMULATIVE_TARGETS,
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "control_checkpoint_sha256": payload.get("control_checkpoint_sha256"),
        "architecture_fingerprint": payload.get("architecture_fingerprint"),
        "parameter_count": payload.get("parameter_count"),
        "replay_ledger_sha256": payload.get("replay_ledger_sha256"),
        "next_global_batch_sha256": payload.get("next_global_batch_sha256"),
        "next_global_stream_sha256": payload.get("next_global_batch_stream_sha256"),
        "optimizer_step_summary": optimizer_evidence["step_summary"],
        "optimizer_state_sha256": optimizer_evidence["state_sha256"],
        "model_state_sha256": optimizer_evidence["model_state_sha256"],
        "saved_training_process_id": payload.get("saved_process_id"),
        "sealing_process_id": os.getpid(),
        "rng_digests": payload.get("rng_digests"),
        "scheduler": payload.get("scheduler"),
        "loader_state": payload.get("loader_state"),
        "metadata": payload.get("metadata"),
        "checks": checks,
        "training_artifact_identity_checks": training_identity_checks,
        "artifact_identity": artifact_identity,
        "optimizer_evidence": optimizer_evidence,
        "process_evidence": training.get("process_evidence"),
        "milestone_binding": binding,
        "sealed": all(checks.values()),
        "sealed_at_unix": time.time(),
    }
    durable_json(args.output_path, result)
    durable_text(checkpoint.with_suffix(checkpoint.suffix + ".sha256"),
                 f"{result['checkpoint_sha256']}  {checkpoint.name}\n")
    if not result["sealed"]:
        raise SystemExit(f"final checkpoint sealing failed: {checks}")
    del model, optimizer, loader, payload
    gc.collect()
    torch.cuda.empty_cache()
    print("EXPERIMENT_2D5C_FINAL_CHECKPOINT_SEALED", flush=True)


def fill_incremental_state(model, device, batch_size=1):
    state = model.init_incremental_state(
        int(batch_size), device=device, dtype=torch.bfloat16
    )
    model.eval()
    tokens = (torch.arange(base.T, device=device, dtype=torch.long) * 7919 + 31)
    tokens = tokens.remainder(50_257).unsqueeze(0).expand(int(batch_size), -1)
    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(base.T):
            _, state = model.incremental_step(tokens[:, position], state)
    audit = model.incremental_cache_audit(state)
    audit["batch_size"] = int(batch_size)
    audit["dtype_required"] = "torch.bfloat16"
    audit["all_persistent_tensors_bf16"] = all(
        row[part] is None or row[part]["dtype"] == "torch.bfloat16"
        for row in audit["cache_storage"] for part in ("key", "value")
    ) and all(
        row["dtype"] == "torch.bfloat16" for row in audit["ring_storage"].values()
    )
    audit["allocator"] = {
        "allocated_growth_bytes_including_execution_residue": (
            torch.cuda.memory_allocated(device) - before_allocated
        ),
        "reserved_growth_bytes_including_allocator_slack": (
            torch.cuda.memory_reserved(device) - before_reserved
        ),
        "peak_allocated_bytes_during_fill": torch.cuda.max_memory_allocated(device),
        "reserved_unused_excluded_from_persistent_state": True,
    }
    return state, audit


def cache_component_bytes(audit):
    rows = {}
    for block in audit["cache_storage"]:
        key = block["key"]
        value = block["value"]
        logical = sum(0 if part is None else part["expected_bytes"] for part in (key, value))
        physical = sum(0 if part is None else part["actual_bytes"] for part in (key, value))
        rows[f"B{block['block']}_same_layer_local_kv"] = {
            "logical_payload_bytes": logical,
            "actual_unique_storage_bytes": physical,
            "historical_positions": audit["historical_local_kv_positions"][f"B{block['block']}"],
        }
    for name, row in audit["ring_storage"].items():
        rows[f"recurrent_ring_{name}"] = {
            "logical_payload_bytes": row["expected_bytes"],
            "actual_unique_storage_bytes": row["actual_bytes"],
            "historical_positions": audit["ring_lengths"][name],
        }
    return rows


def run_memory_audit(args):
    require_branch(clean=True)
    device = base.require_a100()
    fixed, _, fixed_load = load_scientific_evaluator(
        "Fixed", args.fixed_checkpoint, args.source_checkpoint, device, LOCAL_UPDATES
    )
    fixed_state, fixed_audit = fill_incremental_state(fixed, device, 1)
    fixed_components = cache_component_bytes(fixed_audit)
    del fixed_state, fixed
    gc.collect()
    torch.cuda.empty_cache()
    c_model, _, c_load = load_scientific_evaluator(
        "C", args.c_checkpoint, args.source_checkpoint, device, LOCAL_UPDATES,
        args.milestone_manifest, args.final_checkpoint_seal,
    )
    c_state, c_audit = fill_incremental_state(c_model, device, 1)
    c_components = cache_component_bytes(c_audit)
    logical_reduction = (
        fixed_audit["logical_payload_bytes"] - c_audit["logical_payload_bytes"]
    )
    physical_reduction = (
        fixed_audit["actual_unique_storage_bytes"]
        - c_audit["actual_unique_storage_bytes"]
    )
    component_deltas = {
        name: {
            "fixed_logical_bytes": row["logical_payload_bytes"],
            "c_logical_bytes": c_components[name]["logical_payload_bytes"],
            "logical_reduction_bytes": (
                row["logical_payload_bytes"]
                - c_components[name]["logical_payload_bytes"]
            ),
            "fixed_physical_bytes": row["actual_unique_storage_bytes"],
            "c_physical_bytes": c_components[name]["actual_unique_storage_bytes"],
            "physical_reduction_bytes": (
                row["actual_unique_storage_bytes"]
                - c_components[name]["actual_unique_storage_bytes"]
            ),
            "fixed_historical_positions": row["historical_positions"],
            "c_historical_positions": c_components[name]["historical_positions"],
        }
        for name, row in fixed_components.items()
    }
    recurrent_names = [f"recurrent_ring_{name}" for name in ("h7", "h8", "h10", "h12")]
    checks = {
        "fixed_audit": fixed_audit["passed"],
        "c_audit": c_audit["passed"],
        "bf16": fixed_audit["all_persistent_tensors_bf16"] and c_audit["all_persistent_tensors_bf16"],
        "logical_reduction_expected_282624": logical_reduction == 282_624,
        "b3_positions_31_to_1": (
            fixed_audit["b3_historical_local_kv"] == 31
            and c_audit["b3_historical_local_kv"] == 1
        ),
        "b5_positions_63_to_1": (
            fixed_audit["b5_historical_local_kv"] == 63
            and c_audit["b5_historical_local_kv"] == 1
        ),
        "recurrent_rings_unchanged": all(
            component_deltas[name]["logical_reduction_bytes"] == 0
            and component_deltas[name]["physical_reduction_bytes"] == 0
            for name in recurrent_names
        ),
        "only_b3_b5_native_local_changed": all(
            row["logical_reduction_bytes"] == 0
            for name, row in component_deltas.items()
            if name not in ("B3_same_layer_local_kv", "B5_same_layer_local_kv")
        ),
        "physical_measurement_authoritative": True,
        "load_audits": fixed_load["architecture_fingerprint_exact"] and c_load["architecture_fingerprint_exact"],
    }
    result = {
        "experiment": EXPERIMENT,
        "canonical_batch_size": 1,
        "linear_batch_scaling": True,
        "dtype": "BF16",
        "fixed": fixed_audit,
        "c": c_audit,
        "components": component_deltas,
        "logical_persistent_state_reduction_bytes": logical_reduction,
        "logical_persistent_state_reduction_kib": logical_reduction / 1024,
        "actual_physical_persistent_state_reduction_bytes": physical_reduction,
        "actual_physical_persistent_state_reduction_kib": physical_reduction / 1024,
        "checks": checks,
        "passed": all(checks.values()),
    }
    durable_json(args.output_json, result)
    lines = [
        "# 2D5C BF16 persistent inference-state accounting", "",
        "| Component | Fixed logical bytes | C logical bytes | Logical reduction | Fixed physical bytes | C physical bytes | Physical reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(component_deltas.items()):
        lines.append(
            f"| {name} | {row['fixed_logical_bytes']:,} | {row['c_logical_bytes']:,} | "
            f"{row['logical_reduction_bytes']:,} | {row['fixed_physical_bytes']:,} | "
            f"{row['c_physical_bytes']:,} | {row['physical_reduction_bytes']:,} |"
        )
    lines += ["", f"Logical reduction: **{logical_reduction:,} bytes ({logical_reduction / 1024:.0f} KiB)**.",
              f"Measured physical reduction: **{physical_reduction:,} bytes ({physical_reduction / 1024:.0f} KiB)**."]
    durable_text(args.output_table, "\n".join(lines) + "\n")
    if not result["passed"]:
        raise SystemExit(f"BF16 persistent-state audit failed: {checks}")
    del c_state, c_model
    gc.collect()
    torch.cuda.empty_cache()
    print("EXPERIMENT_2D5C_BF16_PERSISTENT_STATE_AUDIT_PASS", flush=True)


def moment_new():
    return {"count": 0, "sum": 0.0, "sum_squares": 0.0,
            "minimum": None, "maximum": None}


def moment_add(row, values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return
    if not np.isfinite(array).all():
        raise SystemExit("non-finite representation diagnostic value")
    row["count"] += int(array.size)
    row["sum"] += float(array.sum(dtype=np.float64))
    row["sum_squares"] += float(np.square(array).sum(dtype=np.float64))
    low, high = float(array.min()), float(array.max())
    row["minimum"] = low if row["minimum"] is None else min(row["minimum"], low)
    row["maximum"] = high if row["maximum"] is None else max(row["maximum"], high)


def moment_merge(target, source):
    target["count"] += int(source["count"])
    target["sum"] += float(source["sum"])
    target["sum_squares"] += float(source["sum_squares"])
    if source["minimum"] is not None:
        target["minimum"] = (
            source["minimum"] if target["minimum"] is None
            else min(target["minimum"], source["minimum"])
        )
        target["maximum"] = (
            source["maximum"] if target["maximum"] is None
            else max(target["maximum"], source["maximum"])
        )


def moment_finalize(row):
    if not row["count"]:
        return {**row, "mean": None, "standard_deviation": None}
    mean = row["sum"] / row["count"]
    variance = max(0.0, row["sum_squares"] / row["count"] - mean * mean)
    return {**row, "mean": mean, "standard_deviation": math.sqrt(variance)}


def diagnostic_link_accumulator(heads):
    local_bins = ("0", "1", *(name for name, _, _ in RECURRENT_BINS))
    return {
        "recurrent_attention": {
            "bins": {
                name: {
                    "mass_per_token_by_head": [moment_new() for _ in range(heads)],
                    "opportunities": 0,
                    "uniform_pair_probability_sum": 0.0,
                }
                for name, _, _ in RECURRENT_BINS
            },
            "entropy_by_head": [moment_new() for _ in range(heads)],
        },
        "local_attention": {
            "bins": {
                name: {
                    "mass_per_token_by_head": [moment_new() for _ in range(heads)],
                    "opportunities": 0,
                    "uniform_pair_probability_sum": 0.0,
                }
                for name in local_bins
            },
            "entropy_by_head": [moment_new() for _ in range(heads)],
        },
        "contribution": {
            name: [moment_new() for _ in range(heads)]
            for name in (
                "local_l2", "local_rms", "gated_recurrent_l2",
                "gated_recurrent_rms", "recurrent_to_local_l2_ratio",
                "local_recurrent_cosine",
            )
        },
        "contribution_aggregate": {
            name: moment_new() for name in (
                "local_l2", "local_rms", "gated_recurrent_l2",
                "gated_recurrent_rms", "recurrent_to_local_l2_ratio",
                "local_recurrent_cosine", "local_post_c_proj_l2",
                "gated_recurrent_post_c_proj_l2",
                "post_c_proj_recurrent_to_local_l2_ratio",
                "post_c_proj_cosine",
            )
        },
        "gradient": {
            kind: {
                name: {
                    "opportunities": 0,
                    "sum_pair_norm": 0.0,
                    "sum_squared_gradient": 0.0,
                    "nonzero_pairs": 0,
                    "per_head": [
                        {"opportunities": 0, "sum_pair_norm": 0.0,
                         "sum_squared_gradient": 0.0, "nonzero_pairs": 0}
                        for _ in range(heads)
                    ] if kind in ("key", "value") else None,
                }
                for name, _, _ in RECURRENT_BINS
            }
            for kind in ("source", "key", "value")
        },
        "actual_writer_gradient": {
            "positions": 0, "positions_with_gradient": 0,
            "positions_with_nonzero_gradient": 0,
            "sum_norm": 0.0, "sum_squared_gradient": 0.0,
        },
    }


def lag_bin_name(lag, include_native=False):
    lag = int(lag)
    if include_native and lag in (0, 1):
        return str(lag)
    for name, low, high in RECURRENT_BINS:
        if low <= lag <= high:
            return name
    return None


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if float(denominator) != 0.0 else None


def cosine_rows(left, right):
    left = left.float().reshape(*left.shape[:-1], -1)
    right = right.float().reshape(*right.shape[:-1], -1)
    return F.cosine_similarity(left, right, dim=-1, eps=1e-12)


def accumulate_attention_and_contribution(accumulator, diagnostic):
    for branch, positions_key, include_native in (
        ("recurrent_attention", "recurrent_positions", False),
        ("local_attention", "local_positions", True),
    ):
        weights = diagnostic[f"{branch.split('_')[0]}_attention_weights"]
        positions = diagnostic[positions_key]
        if weights is None or positions is None or weights.size(-1) == 0:
            continue
        probability = weights.detach().float().cpu()[:, :, 0, :]
        query_position = int(diagnostic["query_position"])
        lags = query_position - positions.detach().cpu().reshape(-1)
        eligible = int(lags.numel())
        entropy = -(probability.clamp_min(1e-30).log() * probability).sum(dim=-1)
        for head in range(probability.size(1)):
            moment_add(accumulator[branch]["entropy_by_head"][head], entropy[:, head])
        bin_rows = accumulator[branch]["bins"]
        for name in bin_rows:
            mask = torch.tensor(
                [lag_bin_name(value, include_native) == name for value in lags.tolist()],
                dtype=torch.bool,
            )
            selected = int(mask.sum())
            if not selected:
                continue
            masses = probability[:, :, mask].sum(dim=-1)
            for head in range(probability.size(1)):
                moment_add(bin_rows[name]["mass_per_token_by_head"][head], masses[:, head])
            batch = int(probability.size(0))
            bin_rows[name]["opportunities"] += selected * batch
            bin_rows[name]["uniform_pair_probability_sum"] += (
                batch * selected / eligible
            )

    local = diagnostic["local_pre_c_proj"].detach().float().cpu()[:, :, 0, :]
    recurrent = diagnostic["gated_recurrent_pre_c_proj"].detach().float().cpu()[:, :, 0, :]
    local_l2 = local.square().sum(dim=-1).sqrt()
    recurrent_l2 = recurrent.square().sum(dim=-1).sqrt()
    local_rms = local.square().mean(dim=-1).sqrt()
    recurrent_rms = recurrent.square().mean(dim=-1).sqrt()
    ratio = recurrent_l2 / local_l2.clamp_min(1e-30)
    cosine = F.cosine_similarity(local, recurrent, dim=-1, eps=1e-12)
    values = {
        "local_l2": local_l2,
        "local_rms": local_rms,
        "gated_recurrent_l2": recurrent_l2,
        "gated_recurrent_rms": recurrent_rms,
        "recurrent_to_local_l2_ratio": ratio,
        "local_recurrent_cosine": cosine,
    }
    for name, tensor in values.items():
        for head in range(tensor.size(1)):
            moment_add(accumulator["contribution"][name][head], tensor[:, head])
    flat_local = local.reshape(local.size(0), -1)
    flat_recurrent = recurrent.reshape(recurrent.size(0), -1)
    post_local = diagnostic["local_post_c_proj"].detach().float().cpu().reshape(local.size(0), -1)
    post_recurrent = diagnostic["gated_recurrent_post_c_proj"].detach().float().cpu().reshape(local.size(0), -1)
    aggregates = {
        "local_l2": flat_local.norm(dim=-1),
        "local_rms": flat_local.square().mean(dim=-1).sqrt(),
        "gated_recurrent_l2": flat_recurrent.norm(dim=-1),
        "gated_recurrent_rms": flat_recurrent.square().mean(dim=-1).sqrt(),
        "recurrent_to_local_l2_ratio": flat_recurrent.norm(dim=-1) / flat_local.norm(dim=-1).clamp_min(1e-30),
        "local_recurrent_cosine": F.cosine_similarity(flat_local, flat_recurrent, dim=-1, eps=1e-12),
        "local_post_c_proj_l2": post_local.norm(dim=-1),
        "gated_recurrent_post_c_proj_l2": post_recurrent.norm(dim=-1),
        "post_c_proj_recurrent_to_local_l2_ratio": post_recurrent.norm(dim=-1) / post_local.norm(dim=-1).clamp_min(1e-30),
        "post_c_proj_cosine": F.cosine_similarity(post_local, post_recurrent, dim=-1, eps=1e-12),
    }
    for name, tensor in aggregates.items():
        moment_add(accumulator["contribution_aggregate"][name], tensor)


def accumulate_gradient_bins(accumulator, diagnostic):
    positions = diagnostic["recurrent_positions"]
    if positions is None or positions.numel() == 0:
        return
    lags = int(diagnostic["query_position"]) - positions.detach().cpu().reshape(-1)
    tensors = {
        "source": diagnostic["recurrent_source_reads"],
        "key": diagnostic["recurrent_key_reads"],
        "value": diagnostic["recurrent_value_reads"],
    }
    for kind, tensor in tensors.items():
        if tensor is None or tensor.grad is None:
            continue
        gradient = tensor.grad.detach().float().cpu()
        for name, _, _ in RECURRENT_BINS:
            mask = torch.tensor(
                [lag_bin_name(value) == name for value in lags.tolist()],
                dtype=torch.bool,
            )
            if not int(mask.sum()):
                continue
            row = accumulator["gradient"][kind][name]
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
                    head_row["sum_squared_gradient"] += float(head_selected.square().sum())
                    head_row["nonzero_pairs"] += int((head_norm > 0).sum())


def finalize_diagnostic_link(accumulator, family, block):
    for branch in ("recurrent_attention", "local_attention"):
        for row in accumulator[branch]["bins"].values():
            row["mass_per_token_by_head"] = [
                moment_finalize(value) for value in row["mass_per_token_by_head"]
            ]
            total_mass = sum(value["sum"] for value in row["mass_per_token_by_head"])
            head_count = len(row["mass_per_token_by_head"])
            paired_opportunities = row["opportunities"] * head_count
            row["opportunity_normalized_mean_attention_probability"] = safe_ratio(
                total_mass, paired_opportunities
            )
            uniform_sum = row["uniform_pair_probability_sum"] * head_count
            row["uniform_reference_mean_probability"] = safe_ratio(
                uniform_sum, paired_opportunities
            )
            observed = row["opportunity_normalized_mean_attention_probability"]
            uniform = row["uniform_reference_mean_probability"]
            row["ratio_to_uniform_over_eligible_lags"] = (
                None if observed is None or uniform in (None, 0.0) else observed / uniform
            )
        accumulator[branch]["entropy_by_head"] = [
            moment_finalize(value) for value in accumulator[branch]["entropy_by_head"]
        ]
    minimum = 2 if family in ("C", "C0") else (32 if block == "b3" else 64)
    for name, low, high in RECURRENT_BINS:
        if high < minimum:
            accumulator["recurrent_attention"]["bins"][name]["availability"] = "N/A"
        else:
            accumulator["recurrent_attention"]["bins"][name]["availability"] = "eligible"
    for group in ("contribution",):
        for name, rows in accumulator[group].items():
            accumulator[group][name] = [moment_finalize(value) for value in rows]
    accumulator["contribution_aggregate"] = {
        name: moment_finalize(value)
        for name, value in accumulator["contribution_aggregate"].items()
    }
    for kind, bins in accumulator["gradient"].items():
        total_norm = sum(row["sum_pair_norm"] for row in bins.values())
        for row in bins.values():
            row["l2_norm_of_all_elements"] = math.sqrt(row["sum_squared_gradient"])
            row["mean_norm_per_eligible_query_lag_pair"] = safe_ratio(
                row["sum_pair_norm"], row["opportunities"]
            )
            row["normalized_fraction_by_pair_norm"] = safe_ratio(
                row["sum_pair_norm"], total_norm
            )
            if row["per_head"] is not None:
                for head_row in row["per_head"]:
                    head_row["l2_norm_of_all_elements"] = math.sqrt(
                        head_row["sum_squared_gradient"]
                    )
                    head_row["mean_norm_per_eligible_query_lag_pair"] = safe_ratio(
                        head_row["sum_pair_norm"], head_row["opportunities"]
                    )
    writer = accumulator["actual_writer_gradient"]
    writer["l2_norm_of_all_elements"] = math.sqrt(writer["sum_squared_gradient"])
    writer["nonzero_back_to_actual_writer"] = writer["positions_with_nonzero_gradient"] > 0
    writer["gradient_scope"] = (
        "temporal_recurrent_ring_write_edge_only; excludes the same-token "
        "ordinary residual path"
    )
    return accumulator


def finite_numeric_tree(value):
    """Require every emitted numeric diagnostic to be finite, allowing N/A."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numeric_tree(item) for item in value)
    return False


def load_diagnostic_bundle(family, checkpoint, source_checkpoint, device,
                           expected_local_update, milestone_manifest=None,
                           final_seal=None):
    source_checkpoint = require_exact_file(
        source_checkpoint, SOURCE_SHA256, "2D3A-1B source"
    )
    optimizer_rebinding = None
    if family == "Parent":
        fixed, source_optimizer, loader, payload, checks = d4a.load_fixed_source(
            source_checkpoint, device, restore=False
        )
        model = make_fixed_evaluator_from_model(fixed)
        optimizer, optimizer_rebinding = rebind_optimizer_by_parameter_name(
            fixed, model, source_optimizer
        )
        checks["optimizer_rebound_by_exact_parameter_name"] = (
            optimizer_rebinding["passed"]
        )
        del fixed, source_optimizer
        local_update = 0
    elif family == "C0":
        model, optimizer, loader, payload, checks = make_c_model(
            source_checkpoint, device, restore=False
        )
        local_update = 0
    elif family == "C":
        if not milestone_manifest:
            raise SystemExit("C diagnostics require the training milestone manifest")
        binding = verify_c_checkpoint_binding(
            checkpoint, expected_local_update, milestone_manifest, final_seal
        )
        model, optimizer, loader, payload, _ = load_c_checkpoint(
            checkpoint, source_checkpoint, device, restore=False
        )
        checks = strict_reopen(checkpoint, source_checkpoint, device)["checks"]
        checks["checkpoint_binding"] = binding["passed"]
        local_update = int(payload["local_updates"])
    elif family == "Fixed":
        (
            model,
            optimizer,
            loader,
            payload,
            checks,
            optimizer_rebinding,
        ) = load_fixed_control(checkpoint, source_checkpoint, device)
        local_update = LOCAL_UPDATES
    else:
        raise SystemExit(f"invalid diagnostic family {family}")
    if local_update != int(expected_local_update):
        raise SystemExit("diagnostic checkpoint local-update mismatch")
    if optimizer_rebinding is None:
        direct_manifest = optimizer_manifest(model, optimizer)
        optimizer_rebinding = {
            "method": "checkpoint/source optimizer already bound to the executed model parameter objects",
            "group_manifest": optimizer_group_manifest(model, optimizer),
            "optimizer_manifest": direct_manifest,
            "checks": {
                "parameter_group_coverage_exact": direct_manifest[
                    "parameter_group_coverage_exact"
                ],
                "optimizer_manifest_passed": direct_manifest["passed"],
            },
            "passed": direct_manifest["passed"],
        }
    if not optimizer_rebinding["passed"]:
        raise SystemExit("diagnostic optimizer is not bound to the executed model")
    del loader
    return (
        model,
        optimizer,
        payload,
        checks,
        local_update,
        optimizer_rebinding,
    )


def one_sequence_representation_diagnostic(model, x, y, family, scale):
    device = x.device
    heads = int(model.config.n_head)
    accumulators = {
        "b3": diagnostic_link_accumulator(heads),
        "b5": diagnostic_link_accumulator(heads),
    }
    records = {"b3": [], "b5": []}
    writer_records = {"b3": [], "b5": []}
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    losses = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(base.T):
            logits, state, diagnostic = model.incremental_step(
                x[position].view(1), state, control="all_real",
                return_diagnostics=True, diagnostic_attention_weights=True,
                return_block_states=False, diagnostic_retain_grad=True,
            )
            losses.append(F.cross_entropy(logits[:, 0].float(), y[position].view(1)))
            for link in ("b3", "b5"):
                link_row = diagnostic["links"][link]
                accumulate_attention_and_contribution(accumulators[link], link_row)
                records[link].append(link_row)
            writer_records["b3"].append(diagnostic["writer_block_states"]["b10"])
            writer_records["b5"].append(diagnostic["writer_block_states"]["b8"])
    mean_ce = torch.stack(losses).mean()
    (mean_ce * float(scale)).backward()
    for link in ("b3", "b5"):
        for row in records[link]:
            accumulate_gradient_bins(accumulators[link], row)
        writer = accumulators[link]["actual_writer_gradient"]
        for value in writer_records[link]:
            writer["positions"] += 1
            if value.grad is None:
                continue
            writer["positions_with_gradient"] += 1
            gradient = value.grad.detach().float()
            norm = float(gradient.norm())
            writer["sum_norm"] += norm
            writer["sum_squared_gradient"] += float(gradient.square().sum())
            writer["positions_with_nonzero_gradient"] += int(norm > 0.0)
        finalize_diagnostic_link(accumulators[link], family, link)
    finite_statistics = finite_numeric_tree(accumulators)
    return {
        "mean_ce": float(mean_ce.detach()),
        "loss_gradient_scale": float(scale),
        "links": accumulators,
        "finite_statistics": finite_statistics,
        "finite": math.isfinite(float(mean_ce.detach())) and finite_statistics,
    }


def merge_representation_rows(rows, family, gates):
    # Preserve all 32 sequence-level summaries, and separately aggregate their
    # raw moment/sum fields so token- and sequence-level dispersion stay auditable.
    combined = {
        "family": family,
        "diagnostic_sequences": len(rows),
        "targets": len(rows) * base.T,
        "gate_transformation": "tanh(raw_gate_parameter)",
        "gates": gates,
        "per_sequence": rows,
        "links": {},
    }
    for link in ("b3", "b5"):
        heads = len(rows[0]["links"][link]["recurrent_attention"]["entropy_by_head"])
        aggregate = diagnostic_link_accumulator(heads)
        # Merge finalized per-sequence records back by their sufficient statistics.
        for sequence in rows:
            current = sequence["links"][link]
            for branch in ("recurrent_attention", "local_attention"):
                for name, source in current[branch]["bins"].items():
                    target = aggregate[branch]["bins"][name]
                    target["opportunities"] += source["opportunities"]
                    target["uniform_pair_probability_sum"] += source["uniform_pair_probability_sum"]
                    for head, value in enumerate(source["mass_per_token_by_head"]):
                        moment_merge(target["mass_per_token_by_head"][head], value)
                for head, value in enumerate(current[branch]["entropy_by_head"]):
                    moment_merge(aggregate[branch]["entropy_by_head"][head], value)
            for name, source_rows in current["contribution"].items():
                for head, value in enumerate(source_rows):
                    moment_merge(aggregate["contribution"][name][head], value)
            for name, value in current["contribution_aggregate"].items():
                moment_merge(aggregate["contribution_aggregate"][name], value)
            for kind in ("source", "key", "value"):
                for name, source in current["gradient"][kind].items():
                    target = aggregate["gradient"][kind][name]
                    for key in ("opportunities", "sum_pair_norm", "sum_squared_gradient", "nonzero_pairs"):
                        target[key] += source[key]
                    if target["per_head"] is not None:
                        for head, source_head in enumerate(source["per_head"]):
                            for key in ("opportunities", "sum_pair_norm", "sum_squared_gradient", "nonzero_pairs"):
                                target["per_head"][head][key] += source_head[key]
            source_writer = current["actual_writer_gradient"]
            target_writer = aggregate["actual_writer_gradient"]
            for key in ("positions", "positions_with_gradient", "positions_with_nonzero_gradient", "sum_norm", "sum_squared_gradient"):
                target_writer[key] += source_writer[key]
        combined["links"][link] = finalize_diagnostic_link(aggregate, family, link)
        combined["links"][link]["per_sequence_dispersion"] = {
            metric: moment_finalize(value)
            for metric, value in {
                "mean_ce": (lambda acc: [moment_add(acc, [row["mean_ce"]]) for row in rows] and acc)(moment_new()),
                "writer_gradient_l2": (lambda acc: [moment_add(acc, [row["links"][link]["actual_writer_gradient"]["l2_norm_of_all_elements"]]) for row in rows] and acc)(moment_new()),
            }.items()
        }
    combined["passed"] = (
        len(rows) == 32
        and all(row["finite"] for row in rows)
        and finite_numeric_tree(combined["links"])
        and all(
            combined["links"][link]["actual_writer_gradient"]["nonzero_back_to_actual_writer"]
            and sum(
                row["nonzero_pairs"]
                for row in combined["links"][link]["gradient"]["source"].values()
            ) > 0
            for link in ("b3", "b5")
        )
    )
    combined["finite_statistics"] = finite_numeric_tree(combined["links"])
    return combined


def run_representation_diagnostics(args):
    require_branch(clean=True)
    device = base.require_a100()
    freeze = read_json(args.pretrain_freeze_audit)
    if not freeze.get("passed"):
        raise SystemExit("representation diagnostics require passed pretraining freeze")
    freeze_checks = {
        "core_manifest_sha256": sha256(args.core_manifest)
        == freeze["artifact_sha256"]["core_panel_manifest"],
    }
    if not all(freeze_checks.values()):
        raise SystemExit(f"diagnostic frozen core binding failed: {freeze_checks}")
    (
        model,
        optimizer,
        payload,
        load_checks,
        local_update,
        optimizer_rebinding,
    ) = load_diagnostic_bundle(
        args.family,
        args.checkpoint,
        args.source_checkpoint,
        device,
        args.expected_local_update,
        args.milestone_manifest,
        args.final_checkpoint_seal,
    )
    model.eval()
    output_dir = Path(args.output_dir).resolve()
    sequence_dir = output_dir / f"{args.label}_sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    core_manifest = read_json(args.core_manifest)
    identities = core_manifest["diagnostic_sequence_identities"]
    if len(identities) != 32:
        raise SystemExit("frozen representation diagnostic subset is not 32 sequences")
    before_model = parameter_manifest(model)
    before_optimizer = optimizer_manifest(model, optimizer)
    checkpoint_sha = (
        sha256(args.checkpoint) if args.checkpoint
        else SOURCE_SHA256
    )
    run_identity = {
        "diagnostic_schema": REPRESENTATION_DIAGNOSTIC_SCHEMA,
        "diagnostic_implementation_sha256": {
            name: digest
            for name, digest in implementation_file_sha256().items()
            if name in {
                "scripts/experiment_2d5c.py",
                "scripts/experiment_2d5c_core.py",
            }
        },
        "experiment": EXPERIMENT,
        "label": args.label,
        "family": args.family,
        "local_update": local_update,
        "checkpoint_sha256": checkpoint_sha,
        "architecture_fingerprint": model.architecture_fingerprint(),
        "model_state_sha256": before_model["aggregate_sha256"],
        "optimizer_state_sha256": before_optimizer["state_aggregate_sha256"],
        "optimizer_model_binding_sha256": canonical_sha(optimizer_rebinding),
        "core_manifest_sha256": sha256(args.core_manifest),
        "core_subset_sha256": core_manifest["subset_sha256"],
        "diagnostic_subset_sha256": core_manifest["diagnostic_subset_sha256"],
    }
    run_identity["identity_sha256"] = canonical_sha(run_identity)
    identity_path = sequence_dir / "RUN_IDENTITY.json"
    if identity_path.exists():
        if read_json(identity_path) != run_identity:
            raise SystemExit("diagnostic resume model/checkpoint identity mismatch")
    else:
        durable_json(identity_path, run_identity)
    gates = {
        link: {
            "raw": float(model.gate_parameter(block).detach().float()),
            "effective": float(model.recurrent_scale(block).detach().float()),
            "effective_recurrent_scaling_actually_applied": float(model.recurrent_scale(block).detach().float()),
            "per_head": None,
        }
        for link, block in (("b3", 2), ("b5", 4))
    }
    rows = []
    val_path = base.validation_path(Path(args.data_root))
    for ordinal, selected in enumerate(identities):
        row_path = sequence_dir / f"sequence_{ordinal:02d}.json"
        if row_path.exists():
            row = read_json(row_path)
            if (
                row.get("combined_sha256") != selected["combined_sha256"]
                or row.get("run_identity_sha256") != run_identity["identity_sha256"]
            ):
                raise SystemExit("diagnostic resume identity mismatch")
            rows.append(row)
            continue
        cpu_x, cpu_y = batch_at_index(val_path, int(selected["batch_index"]))
        index = int(selected["sequence_index"])
        if tensor_sha256(cpu_x[index], cpu_y[index]) != selected["combined_sha256"]:
            raise SystemExit("frozen diagnostic token identity mismatch")
        # Clear the parameters of the model that actually executes.  This is
        # intentionally not delegated to an optimizer wrapper.
        model.zero_grad(set_to_none=True)
        result = one_sequence_representation_diagnostic(
            model, cpu_x[index].to(device), cpu_y[index].to(device),
            args.family, 1.0 / len(identities),
        )
        row = {
            "ordinal": ordinal,
            "batch_index": int(selected["batch_index"]),
            "sequence_index": index,
            "combined_sha256": selected["combined_sha256"],
            "run_identity_sha256": run_identity["identity_sha256"],
            **result,
        }
        durable_json(row_path, row)
        rows.append(row)
        model.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise SystemExit("representation diagnostic retained evaluator gradients")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"2D5C representation {args.label} sequence {ordinal + 1}/32", flush=True)
    after_model = parameter_manifest(model)
    after_optimizer = optimizer_manifest(model, optimizer)
    result = merge_representation_rows(rows, args.family, gates)
    invariance = {
        "model_before_sha256": before_model["aggregate_sha256"],
        "model_after_sha256": after_model["aggregate_sha256"],
        "model_unchanged": before_model["aggregate_sha256"] == after_model["aggregate_sha256"],
        "optimizer_before_sha256": before_optimizer["state_aggregate_sha256"],
        "optimizer_after_sha256": after_optimizer["state_aggregate_sha256"],
        "optimizer_unchanged": before_optimizer["state_aggregate_sha256"] == after_optimizer["state_aggregate_sha256"],
        "optimizer_steps_executed": 0,
        "optimizer_model_binding_verified": optimizer_rebinding["passed"],
        "no_parameter_gradients_retained": all(parameter.grad is None for parameter in model.parameters()),
    }
    result.update({
        "experiment": EXPERIMENT,
        "label": args.label,
        "local_update": local_update,
        "architecture_fingerprint": model.architecture_fingerprint(),
        "checkpoint_sha256": SOURCE_SHA256 if not args.checkpoint else sha256(args.checkpoint),
        "core_sha256": core_manifest["subset_sha256"],
        "diagnostic_subset_sha256": core_manifest["diagnostic_subset_sha256"],
        "load_checks": load_checks,
        "optimizer_model_binding": optimizer_rebinding,
        "run_identity": run_identity,
        "frozen_artifact_checks": freeze_checks,
        "state_invariance": invariance,
    })
    result["passed"] = (
        result["passed"]
        and all(load_checks.values())
        and all(freeze_checks.values())
        and invariance["model_unchanged"]
        and invariance["optimizer_unchanged"]
        and invariance["optimizer_steps_executed"] == 0
        and invariance["optimizer_model_binding_verified"]
        and invariance["no_parameter_gradients_retained"]
    )
    durable_json(args.output_json, result)
    if not result["passed"]:
        raise SystemExit("representation-pressure diagnostic audit failed")
    del model, optimizer, payload
    gc.collect()
    torch.cuda.empty_cache()
    print(f"EXPERIMENT_2D5C_REPRESENTATION_DIAGNOSTIC_COMPLETE {args.label}", flush=True)


CONDITION_CANONICAL_SUFFIX = {
    "all_real": "ALL_REAL",
    "b3_off": "B3_RECURRENCE_OFF",
    "b3_shuffled": "B3_SHUFFLED",
    "b5_off": "B5_RECURRENCE_OFF",
    "b5_shuffled": "B5_SHUFFLED",
    "b3_b5_off": "B3_B5_BOTH_OFF",
    "b3_b5_shuffled": "B3_B5_BOTH_SHUFFLED",
}


def all_real_sequence_ce(evaluation):
    return evaluation["conditions"]["all_real"]["per_sequence_ce"]


def nested_value(value, *keys):
    """Return a nested artifact value without letting malformed input crash audit."""
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def valid_sha256(value):
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def aggregate_identity_sha256(rows):
    try:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(bytes.fromhex(row["combined_sha256"]))
        return digest.hexdigest()
    except (KeyError, TypeError, ValueError):
        return None


def canonical_identity_matches(identity, exact_keys):
    if not isinstance(identity, dict) or set(identity) != set(exact_keys):
        return False
    claimed = identity.get("identity_sha256")
    payload = {key: value for key, value in identity.items()
               if key != "identity_sha256"}
    return valid_sha256(claimed) and claimed == canonical_sha(payload)


EVALUATION_IDENTITY_KEYS = frozenset({
    "family", "checkpoint_sha256", "local_update", "model_state_sha256",
    "architecture_fingerprint", "panel_manifest_path",
    "panel_manifest_sha256", "shuffle_manifest_path",
    "shuffle_manifest_sha256", "identity_sha256",
})

REPRESENTATION_IDENTITY_KEYS = frozenset({
    "diagnostic_schema", "diagnostic_implementation_sha256",
    "experiment", "label", "family", "local_update", "checkpoint_sha256",
    "architecture_fingerprint", "model_state_sha256",
    "optimizer_state_sha256", "optimizer_model_binding_sha256",
    "core_manifest_sha256", "core_subset_sha256",
    "diagnostic_subset_sha256", "identity_sha256",
})


def evaluation_artifact_identity_checks(evaluation, spec, panel,
                                        panel_manifest_sha256,
                                        shuffle_manifest_sha256,
                                        parallel=False):
    """Bind one raw evaluator artifact to its model and frozen input files."""
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    identity = evaluation.get("evaluation_identity", {})
    conditions = evaluation.get("conditions", {})
    identity = identity if isinstance(identity, dict) else {}
    conditions = conditions if isinstance(conditions, dict) else {}
    controls = list(spec["controls"])
    expected_batch_indices = panel.get("batch_indices_in_evaluation_order", [])
    expected_batch_identities = panel.get("batch_identities", [])
    expected_panel_sha = panel.get("panel_sha256", panel.get("subset_sha256"))
    expected_sequences = panel.get("sequence_count", panel.get("sequences"))
    expected_targets = panel.get("targets_per_condition")
    rows = [conditions.get(control, {}) for control in controls]
    checks = {
        "experiment_exact": evaluation.get("experiment") == EXPERIMENT,
        "family_exact": evaluation.get("family") == spec["family"],
        "local_update_exact": identity.get("local_update") == spec["local_update"],
        "checkpoint_sha256_exact": identity.get("checkpoint_sha256") == spec["checkpoint_sha256"],
        "architecture_fingerprint_exact": (
            evaluation.get("architecture_fingerprint", identity.get("architecture_fingerprint"))
            == spec["architecture_fingerprint"]
            and identity.get("architecture_fingerprint") == spec["architecture_fingerprint"]
        ),
        "model_state_sha256_present": valid_sha256(identity.get("model_state_sha256")),
        "identity_schema_and_digest_exact": canonical_identity_matches(
            identity, EVALUATION_IDENTITY_KEYS
        ),
        "panel_manifest_sha256_exact": (
            identity.get("panel_manifest_sha256") == panel_manifest_sha256
        ),
        "shuffle_manifest_sha256_exact": (
            identity.get("shuffle_manifest_sha256") == shuffle_manifest_sha256
        ),
        "manifest_paths_recorded": (
            isinstance(identity.get("panel_manifest_path"), str)
            and bool(identity.get("panel_manifest_path"))
            and isinstance(identity.get("shuffle_manifest_path"), str)
            and bool(identity.get("shuffle_manifest_path"))
        ),
        "panel_intrinsic_sha_exact": evaluation.get("panel_sha256") == expected_panel_sha,
        "controls_exact": evaluation.get("controls_requested") == controls,
        "condition_keys_exact": list(conditions) == controls,
        "status_complete": evaluation.get("status") == "complete",
        "artifact_passed": evaluation.get("passed") is True,
        "batch_identities_exact": evaluation.get("batch_identities") == expected_batch_identities,
        "same_sequence_order": evaluation.get("same_sequence_order_all_conditions") is True,
        "targets_exact_every_condition": all(
            row.get("validation_targets") == expected_targets for row in rows
        ),
        "sequences_exact_every_condition": all(
            row.get("paired_sequences") == expected_sequences for row in rows
        ),
        "per_sequence_arrays_exact": all(
            isinstance(row.get("per_sequence_nll"), list)
            and isinstance(row.get("per_sequence_ce"), list)
            and len(row["per_sequence_nll"]) == expected_sequences
            and len(row["per_sequence_ce"]) == expected_sequences
            for row in rows
        ),
        "validation_losses_finite": all(
            isinstance(row.get("validation_loss"), (int, float))
            and math.isfinite(float(row["validation_loss"])) for row in rows
        ),
    }
    if parallel:
        checks.update({
            "secondary_not_primary": evaluation.get("primary") is False,
            "parallel_mode_explicit": evaluation.get("evaluation_mode") in {
                "path_matched_parallel_2pass", "path_matched_parallel_3pass"
            },
        })
    else:
        checks.update({
            "batch_indices_exact": (
                evaluation.get("batch_indices_in_evaluation_order")
                == expected_batch_indices
            ),
            "completed_batch_indices_exact": (
                evaluation.get("completed_batch_indices") == expected_batch_indices
            ),
            "cache_reset_between_conditions": (
                evaluation.get("cache_reset_between_conditions") is True
            ),
            "terminal_sentinel_passed": nested_value(
                evaluation, "all_real_terminal_sentinel", "passed"
            ) is True,
            "deployment_cache_audit_passed": all(
                nested_value(row, "final_cache_audit", "passed") is True
                for row in rows
            ),
        })
    return {"checks": checks, "passed": all(checks.values())}


def representation_artifact_identity_checks(row, spec, core_manifest,
                                            core_manifest_sha256):
    """Bind one representation result to the same sealed model and core."""
    row = row if isinstance(row, dict) else {}
    identity = row.get("run_identity", {})
    state = row.get("state_invariance", {})
    identity = identity if isinstance(identity, dict) else {}
    state = state if isinstance(state, dict) else {}
    selected = core_manifest.get("diagnostic_sequence_identities", [])
    per_sequence = row.get("per_sequence", []) if isinstance(row, dict) else []
    expected_sequence_hashes = [item.get("combined_sha256") for item in selected]
    actual_sequence_hashes = [item.get("combined_sha256") for item in per_sequence] \
        if isinstance(per_sequence, list) else []
    current_implementation = implementation_file_sha256()
    expected_diagnostic_implementation = {
        name: current_implementation[name]
        for name in (
            "scripts/experiment_2d5c.py",
            "scripts/experiment_2d5c_core.py",
        )
    }
    checks = {
        "diagnostic_schema_exact": (
            identity.get("diagnostic_schema")
            == REPRESENTATION_DIAGNOSTIC_SCHEMA
        ),
        "diagnostic_implementation_exact": (
            identity.get("diagnostic_implementation_sha256")
            == expected_diagnostic_implementation
        ),
        "experiment_exact": row.get("experiment") == EXPERIMENT,
        "family_exact": row.get("family") == spec["family"],
        "local_update_exact": (
            row.get("local_update") == identity.get("local_update")
            == spec["local_update"]
        ),
        "checkpoint_sha256_exact": (
            row.get("checkpoint_sha256") == identity.get("checkpoint_sha256")
            == spec["checkpoint_sha256"]
        ),
        "architecture_fingerprint_exact": (
            row.get("architecture_fingerprint")
            == identity.get("architecture_fingerprint")
            == spec["architecture_fingerprint"]
        ),
        "model_state_sha256_present": valid_sha256(identity.get("model_state_sha256")),
        "optimizer_state_sha256_present": valid_sha256(identity.get("optimizer_state_sha256")),
        "optimizer_model_binding_sha256_present": valid_sha256(
            identity.get("optimizer_model_binding_sha256")
        ),
        "optimizer_model_binding_evidence_exact": (
            nested_value(row, "optimizer_model_binding", "passed") is True
            and canonical_sha(row.get("optimizer_model_binding"))
            == identity.get("optimizer_model_binding_sha256")
        ),
        "identity_schema_and_digest_exact": canonical_identity_matches(
            identity, REPRESENTATION_IDENTITY_KEYS
        ),
        "label_bound": row.get("label") == identity.get("label") and bool(row.get("label")),
        "core_manifest_sha256_exact": (
            identity.get("core_manifest_sha256") == core_manifest_sha256
        ),
        "core_subset_sha256_exact": (
            row.get("core_sha256") == identity.get("core_subset_sha256")
            == core_manifest.get("subset_sha256") == CORE_SHA256
        ),
        "diagnostic_subset_sha256_exact": (
            row.get("diagnostic_subset_sha256")
            == identity.get("diagnostic_subset_sha256")
            == core_manifest.get("diagnostic_subset_sha256")
        ),
        "diagnostic_subset_recomputed": (
            core_manifest.get("diagnostic_subset_sha256") == canonical_sha(selected)
        ),
        "diagnostic_sequences_exact": (
            row.get("diagnostic_sequences") == len(selected) == 32
            and row.get("targets") == 32 * base.T
        ),
        "diagnostic_sequence_identities_exact": (
            actual_sequence_hashes == expected_sequence_hashes
            and all(
                item.get("run_identity_sha256") == identity.get("identity_sha256")
                for item in per_sequence
            )
        ),
        "frozen_core_check_passed": nested_value(
            row, "frozen_artifact_checks", "core_manifest_sha256"
        ) is True,
        "model_unchanged": state.get("model_unchanged") is True,
        "optimizer_unchanged": state.get("optimizer_unchanged") is True,
        "optimizer_steps_zero": state.get("optimizer_steps_executed") == 0,
        "parameter_gradients_cleared": state.get("no_parameter_gradients_retained") is True,
        "artifact_passed": row.get("passed") is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def analysis_input_identity_audit(core_evaluations, large_evaluations,
                                  parallel_evaluations, representation_rows,
                                  core_manifest, large_manifest, shuffle_manifest,
                                  milestones, freeze, evidence, file_sha256s):
    """Fail-closed identity audit for every scientific analysis input."""
    core_manifest = core_manifest if isinstance(core_manifest, dict) else {}
    large_manifest = large_manifest if isinstance(large_manifest, dict) else {}
    shuffle_manifest = shuffle_manifest if isinstance(shuffle_manifest, dict) else {}
    milestones = milestones if isinstance(milestones, dict) else {}
    freeze = freeze if isinstance(freeze, dict) else {}
    milestone_checks = {
        "keys_exact": set(milestones) == {"48", "96", "144", "191"},
        "file_sha256_matches_training_complete": (
            file_sha256s["milestone_manifest"]
            == nested_value(
                evidence, "training", "artifact_identity", "milestone_manifest",
                "sha256",
            )
        ),
    }
    for update in MILESTONES:
        row = milestones.get(str(update), {})
        milestone_checks[f"update_{update}_exact"] = (
            valid_sha256(row.get("sha256"))
            and row.get("local_update") == update
            and row.get("global_update") == SOURCE_UPDATES + update
            and row.get("cumulative_targets") == MILESTONE_TARGETS[update]
            and row.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C
            and valid_sha256(row.get("model_state_sha256"))
            and valid_sha256(row.get("optimizer_state_sha256"))
            and isinstance(row.get("checkpoint"), str)
            and Path(row.get("checkpoint", "")).name == checkpoint_name(update)
            and nested_value(row, "strict_reopen", "passed") is True
        )

    core_identities = core_manifest.get("batch_identities", [])
    large_identities = large_manifest.get("batch_identities", [])
    manifest_checks = {
        "core_experiment": core_manifest.get("experiment") == EXPERIMENT,
        "core_subset_sha256": core_manifest.get("subset_sha256") == CORE_SHA256,
        "core_subset_recomputed": aggregate_identity_sha256(core_identities) == CORE_SHA256,
        "core_batch_indices": core_manifest.get("batch_indices_in_evaluation_order") == [0, 1, 2, 3],
        "core_size": core_manifest.get("sequences") == 256
        and core_manifest.get("targets_per_condition") == 262_144,
        "core_diagnostic_size": len(core_manifest.get("diagnostic_sequence_identities", [])) == 32,
        "large_experiment": large_manifest.get("experiment") == EXPERIMENT,
        "large_frozen_before_training": large_manifest.get("frozen_before_training") is True,
        "large_seed": large_manifest.get("selection_seed") == LARGE_SELECTION_SEED,
        "large_panel_sha256": valid_sha256(large_manifest.get("panel_sha256")),
        "large_panel_recomputed": (
            aggregate_identity_sha256(large_identities) == large_manifest.get("panel_sha256")
        ),
        "large_size": large_manifest.get("sequence_count") == 2_048
        and large_manifest.get("targets_per_condition") == 2_097_152
        and len(large_manifest.get("batch_indices_in_evaluation_order", [])) == 32,
        "large_disjointness": large_manifest.get("all_required_disjointness_passed") is True,
        "shuffle_seed": shuffle_manifest.get("seed") == SHUFFLE_SEED,
        "shuffle_passed": shuffle_manifest.get("passed") is True,
    }
    for size in (32, 64):
        permutation = nested_value(shuffle_manifest, "permutations", str(size), "values")
        manifest_checks[f"shuffle_{size}_exact_derangement"] = (
            isinstance(permutation, list) and len(permutation) == size
            and sorted(permutation) == list(range(size))
            and all(index != donor for index, donor in enumerate(permutation))
            and nested_value(shuffle_manifest, "permutations", str(size), "sha256")
            == canonical_sha(permutation)
        )

    frozen = freeze.get("artifact_sha256", {})
    replay = evidence["replay"]
    preflight = evidence["preflight"]
    freeze_checks = {
        "freeze_passed": freeze.get("passed") is True,
        "frozen_artifact_set_exact": set(frozen) == {
            "data_replay_ledger", "data_replay_audit", "core_panel_manifest",
            "large_panel_manifest", "shuffle_manifest", "scope_lock",
            "source_provenance", "fixed_control_provenance",
        },
        "core_manifest_file": frozen.get("core_panel_manifest") == file_sha256s["core_manifest"],
        "large_manifest_file": frozen.get("large_panel_manifest") == file_sha256s["large_manifest"],
        "shuffle_manifest_file": frozen.get("shuffle_manifest") == file_sha256s["shuffle_manifest"],
        "scope_lock_file": frozen.get("scope_lock") == file_sha256s["scope_lock"],
        "source_provenance_file": frozen.get("source_provenance") == file_sha256s["source_provenance"],
        "control_provenance_file": frozen.get("fixed_control_provenance") == file_sha256s["control_provenance"],
        "replay_audit_file": frozen.get("data_replay_audit") == file_sha256s["replay_audit"],
        "replay_ledger_digest": frozen.get("data_replay_ledger") == replay.get("ledger_sha256"),
        "preflight_authorized": preflight.get("authorized") is True,
        "preflight_freeze_manifest_exact": preflight.get("frozen_artifact_sha256") == frozen,
        "source_provenance_exact": evidence["source"].get("passed") is True
        and evidence["source"].get("sha256") == SOURCE_SHA256,
        "control_provenance_exact": evidence["control"].get("passed") is True
        and evidence["control"].get("sha256") == CONTROL_SHA256,
    }

    specs = {
        "parent": {"family": "Parent", "local_update": 0,
                   "checkpoint_sha256": SOURCE_SHA256,
                   "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_FIXED,
                   "controls": ("all_real",)},
        "c0": {"family": "C0", "local_update": 0,
               "checkpoint_sha256": SOURCE_SHA256,
               "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
               "controls": CONTROLS},
        "c48": {"family": "C", "local_update": 48,
                "checkpoint_sha256": nested_value(milestones, "48", "sha256"),
                "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
                "controls": CONTROLS},
        "c96": {"family": "C", "local_update": 96,
                "checkpoint_sha256": nested_value(milestones, "96", "sha256"),
                "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
                "controls": CONTROLS},
        "c144": {"family": "C", "local_update": 144,
                 "checkpoint_sha256": nested_value(milestones, "144", "sha256"),
                 "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
                 "controls": CONTROLS},
        "c191": {"family": "C", "local_update": 191,
                 "checkpoint_sha256": nested_value(milestones, "191", "sha256"),
                 "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_C,
                 "controls": CONTROLS},
        "fixed100m": {"family": "Fixed", "local_update": 191,
                      "checkpoint_sha256": CONTROL_SHA256,
                      "architecture_fingerprint": ARCHITECTURE_FINGERPRINT_FIXED,
                      "controls": CONTROLS},
    }
    artifact_set_checks = {
        "core_set_exact": set(core_evaluations) == set(specs),
        "large_set_exact": set(large_evaluations) == {"c191", "fixed100m"},
        "parallel_set_exact": set(parallel_evaluations) == {"c0", "c96", "c191"},
        "representation_set_exact": set(representation_rows) == set(specs),
    }
    evaluation_checks = {
        name: evaluation_artifact_identity_checks(
            row, specs[name], core_manifest, file_sha256s["core_manifest"],
            file_sha256s["shuffle_manifest"], parallel=False,
        )
        for name, row in core_evaluations.items() if name in specs
    }
    large_checks = {
        name: evaluation_artifact_identity_checks(
            row, specs[name], large_manifest, file_sha256s["large_manifest"],
            file_sha256s["shuffle_manifest"], parallel=False,
        )
        for name, row in large_evaluations.items() if name in specs
    }
    parallel_checks = {
        name: evaluation_artifact_identity_checks(
            row, specs[name], core_manifest, file_sha256s["core_manifest"],
            file_sha256s["shuffle_manifest"], parallel=True,
        )
        for name, row in parallel_evaluations.items() if name in specs
    }
    representation_checks = {
        name: representation_artifact_identity_checks(
            row, specs[name], core_manifest, file_sha256s["core_manifest"]
        )
        for name, row in representation_rows.items() if name in specs
    }

    def model_sha(evaluations, name):
        return nested_value(evaluations, name, "evaluation_identity", "model_state_sha256")

    seal = evidence["seal"]
    source_model_sha = nested_value(
        evidence, "source", "model_state_manifest", "aggregate_sha256"
    )
    source_optimizer_sha = nested_value(
        evidence, "source", "optimizer_state_manifest", "state_aggregate_sha256"
    )
    cross_checks = {
        "source_parent_model_state": model_sha(core_evaluations, "parent") == source_model_sha,
        "source_c0_model_state": model_sha(core_evaluations, "c0") == source_model_sha,
        "source_parent_optimizer_state": nested_value(
            representation_rows, "parent", "run_identity", "optimizer_state_sha256"
        ) == source_optimizer_sha,
        "source_c0_optimizer_state": nested_value(
            representation_rows, "c0", "run_identity", "optimizer_state_sha256"
        ) == source_optimizer_sha,
        "final_seal_milestone_sha": (
            seal.get("sealed") is True
            and seal.get("local_update") == LOCAL_UPDATES
            and seal.get("checkpoint_sha256") == nested_value(milestones, "191", "sha256")
            and seal.get("architecture_fingerprint") == ARCHITECTURE_FINGERPRINT_C
            and nested_value(seal, "milestone_binding", "passed") is True
        ),
        "final_c191_model_state": model_sha(core_evaluations, "c191")
        == seal.get("model_state_sha256"),
        "large_c191_same_model": model_sha(large_evaluations, "c191")
        == model_sha(core_evaluations, "c191"),
        "large_fixed_same_model": model_sha(large_evaluations, "fixed100m")
        == model_sha(core_evaluations, "fixed100m"),
        "midpoint_c96_model_state": model_sha(core_evaluations, "c96")
        == nested_value(evidence, "restart", "pre_exit", "model_aggregate_sha256"),
        "final_seal_model_state_matches_milestone": seal.get("model_state_sha256")
        == nested_value(milestones, "191", "model_state_sha256"),
        "final_seal_optimizer_matches_milestone": seal.get("optimizer_state_sha256")
        == nested_value(milestones, "191", "optimizer_state_sha256"),
        "parallel_modes_path_matched": len({
            row.get("evaluation_mode") for row in parallel_evaluations.values()
        }) == 1,
    }
    for name in ("c0", "c96", "c191"):
        cross_checks[f"parallel_{name}_same_identity"] = (
            parallel_evaluations.get(name, {}).get("evaluation_identity")
            == core_evaluations.get(name, {}).get("evaluation_identity")
        )
    for name in specs:
        cross_checks[f"representation_{name}_same_model"] = (
            nested_value(representation_rows, name, "run_identity", "model_state_sha256")
            == model_sha(core_evaluations, name)
        )
    for name, update in (("c48", "48"), ("c96", "96"),
                         ("c144", "144"), ("c191", "191")):
        cross_checks[f"core_{name}_model_matches_milestone"] = (
            model_sha(core_evaluations, name)
            == nested_value(milestones, update, "model_state_sha256")
        )
        cross_checks[f"representation_{name}_optimizer_matches_milestone"] = (
            nested_value(
                representation_rows, name, "run_identity", "optimizer_state_sha256"
            ) == nested_value(milestones, update, "optimizer_state_sha256")
        )
    cross_checks["representation_c191_optimizer_matches_seal"] = (
        nested_value(representation_rows, "c191", "run_identity", "optimizer_state_sha256")
        == seal.get("optimizer_state_sha256")
    )

    groups = {
        "milestone_manifest": {"checks": milestone_checks,
                               "passed": all(milestone_checks.values())},
        "frozen_manifests": {"checks": manifest_checks,
                             "passed": all(manifest_checks.values())},
        "pretrain_freeze": {"checks": freeze_checks,
                            "passed": all(freeze_checks.values())},
        "artifact_sets": {"checks": artifact_set_checks,
                          "passed": all(artifact_set_checks.values())},
        "core_evaluations": evaluation_checks,
        "large_evaluations": large_checks,
        "secondary_parallel_evaluations": parallel_checks,
        "representation_diagnostics": representation_checks,
        "cross_artifact_binding": {"checks": cross_checks,
                                   "passed": all(cross_checks.values())},
    }
    nested_artifact_groups_pass = all(
        len(group) == len(expected)
        and all(item.get("passed") is True for item in group.values())
        for group, expected in (
            (evaluation_checks, specs),
            (large_checks, {"c191", "fixed100m"}),
            (parallel_checks, {"c0", "c96", "c191"}),
            (representation_checks, specs),
        )
    )
    passed = (
        groups["milestone_manifest"]["passed"]
        and groups["frozen_manifests"]["passed"]
        and groups["pretrain_freeze"]["passed"]
        and groups["artifact_sets"]["passed"]
        and groups["cross_artifact_binding"]["passed"]
        and nested_artifact_groups_pass
    )
    return {
        "experiment": EXPERIMENT,
        "purpose": "fail-closed binding of every analysis input to frozen manifests and sealed checkpoints",
        "bound_file_sha256": dict(file_sha256s),
        "bound_panel_identity": {
            "core_subset_sha256": core_manifest.get("subset_sha256"),
            "large_panel_sha256": large_manifest.get("panel_sha256"),
            "diagnostic_subset_sha256": core_manifest.get("diagnostic_subset_sha256"),
            "shuffle_manifest_sha256": file_sha256s["shuffle_manifest"],
        },
        "bound_checkpoint_sha256": {
            name: spec["checkpoint_sha256"] for name, spec in specs.items()
        },
        "bound_model_state_sha256": {
            name: model_sha(core_evaluations, name) for name in specs
        },
        "secondary_parallel_modes": {
            name: row.get("evaluation_mode")
            for name, row in parallel_evaluations.items()
        },
        "groups": groups,
        "passed": passed,
    }


def large_loss_artifact(c_evaluation, fixed_evaluation):
    rows = {}
    for prefix, evaluation in (("C", c_evaluation), ("F", fixed_evaluation)):
        for source, suffix in CONDITION_CANONICAL_SUFFIX.items():
            row = evaluation["conditions"][source]
            rows[f"{prefix}_{suffix}"] = {
                "validation_loss": row["validation_loss"],
                "validation_targets": row["validation_targets"],
                "paired_sequences": row["paired_sequences"],
                "per_sequence_nll": row["per_sequence_nll"],
                "per_sequence_ce": row["per_sequence_ce"],
            }
    return {
        "experiment": EXPERIMENT,
        "condition_order": list(analysis.FINAL_CONDITIONS),
        "panel_sha256": c_evaluation["panel_sha256"],
        "same_panel_fixed_and_c": c_evaluation["panel_sha256"] == fixed_evaluation["panel_sha256"],
        "same_sequence_order_all_conditions": (
            c_evaluation["same_sequence_order_all_conditions"]
            and fixed_evaluation["same_sequence_order_all_conditions"]
            and c_evaluation["batch_identities"] == fixed_evaluation["batch_identities"]
        ),
        "conditions": rows,
    }


def combine_representation_artifacts(paths, rows=None):
    expected = {
        "parent", "c0", "c48", "c96", "c144", "c191", "fixed100m"
    }
    if set(paths) != expected:
        raise SystemExit(
            f"representation diagnostic set mismatch: {sorted(set(paths) ^ expected)}"
        )
    rows = ({name: read_json(path) for name, path in paths.items()}
            if rows is None else dict(rows))
    checks = {
        "exact_models": set(rows) == expected,
        "all_passed": all(row.get("passed") is True for row in rows.values()),
        "all_32_sequences": all(row.get("diagnostic_sequences") == 32 for row in rows.values()),
        "all_same_subset": len({row.get("diagnostic_subset_sha256") for row in rows.values()}) == 1,
        "model_optimizer_unchanged": all(
            row["state_invariance"]["model_unchanged"]
            and row["state_invariance"]["optimizer_unchanged"]
            and row["state_invariance"]["optimizer_steps_executed"] == 0
            for row in rows.values()
        ),
        "attached_b3_b5_writer_gradients": all(
            row["links"][link]["actual_writer_gradient"]["nonzero_back_to_actual_writer"]
            for row in rows.values() for link in ("b3", "b5")
        ),
    }
    return {
        "experiment": EXPERIMENT,
        "models": rows,
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation_guardrails": [
            "OFF and SHUFFLED CE interventions are the causal evidence.",
            "Attention probabilities are normalized within their own branch.",
            "Norms, ratios, cosine, entropy, and gate size are descriptive only.",
        ],
    }


def scientific_audit_checks(evidence, large_losses, large_analysis,
                            longitudinal, representation, memory):
    scope = evidence["scope"]
    source = evidence["source"]
    control = evidence["control"]
    replay = evidence["replay"]
    preflight = evidence["preflight"]
    tests = evidence["preflight_tests"]
    training = evidence["training"]
    restart = evidence["restart"]
    seal = evidence["seal"]
    panel = evidence["large_panel"]
    backup = evidence["local_backup"]
    c_core = evidence["c_core"]
    fixed_core = evidence["fixed_core"]
    c_large = evidence["c_large"]
    fixed_large = evidence["fixed_large"]
    core_complete = (
        longitudinal["passed"]
        and set(longitudinal["c"]) == {"0", "48", "96", "144", "191"}
        and set(longitudinal["fixed_available"]) == {"191"}
        and all(row.get("passed") for row in c_core.values())
        and fixed_core.get("passed")
    )
    exact_large_targets = all(
        row["validation_targets"] == 2_097_152
        and row["paired_sequences"] == 2_048
        for row in large_losses["conditions"].values()
    )
    historical_disjointness_rows = [
        row for row in panel["disjointness"].values()
        if "historical_subset_sha256" in row
    ]
    historical_dataset_and_span_evidence_exact = (
        len(historical_disjointness_rows) == 3
        and all(
            row["verified"]
            and row["dataset_identity"]["verified"]
            and row["canonical_span_nonoverlap_verified"]
            and row["canonical_target_span_intersections"] == []
            and (
                row["per_sequence_history_available"]
                or row["sequence_hash_intersection"] is None
            )
            for row in historical_disjointness_rows
        )
    )
    checks = {
        "source_checkpoint_sha_exact": source["sha256"] == SOURCE_SHA256 and source["passed"],
        "c_started_from_2d3a_source": seal["parent_checkpoint_sha256"] == SOURCE_SHA256,
        "fixed_control_checkpoint_sha_exact": control["sha256"] == CONTROL_SHA256 and control["passed"],
        "exactly_one_newly_trained_arm": scope["exactly_one_newly_trained_arm"] == ["C"],
        "fixed_control_optimizer_steps_zero": scope["fixed_control_optimizer_steps"] == 0,
        "c_optimizer_steps_exact": training.get("passed") is True
        and training["checks"]["optimizer_updates_exact"],
        "c_new_targets_exact": training["checks"]["targets_exact"],
        "final_global_update_exact": training["checks"]["final_global_update"] and seal["global_update"] == FINAL_GLOBAL_UPDATE,
        "final_cumulative_targets_exact": training["checks"]["final_cumulative_targets"] and seal["cumulative_targets"] == FINAL_CUMULATIVE_TARGETS,
        "replay_191_batches_exact": replay["rows"] == LOCAL_UPDATES and training["checks"]["batch_replay_exact"],
        "replay_chain_hash_exact": replay["passed"] and seal["replay_ledger_sha256"] == replay["ledger_sha256"],
        "initial_terminal_loader_cursor_hashes_exact": (
            source["next_batch_sha256"] == SOURCE_NEXT_BATCH
            and source["next_stream_sha256"] == SOURCE_NEXT_STREAM
            and seal["next_global_batch_sha256"] == CONTROL_NEXT_BATCH
            and seal["next_global_stream_sha256"] == CONTROL_NEXT_STREAM
        ),
        "pass_cadence_exact": replay["checks"]["pass_cadence"] and training["checks"]["pass_cadence_exact"],
        "optimizer_continuity": restart["checks"]["optimizer_digest"]
        and restart["checks"]["optimizer_steps"]
        and tests["optimizer_name_rebinding"]["passed"],
        "scheduler_continuity": (
            source["scheduler_key_present"]
            and seal["scheduler"] == source["scheduler_state"]
        ),
        "rng_continuity": restart["checks"]["rng_digests"] and seal["rng_digests"] is not None,
        "midpoint_fresh_process_restart_success": restart["passed"] and restart["checks"]["fresh_process"],
        "parameter_count_unchanged": tests.get("passed") is True
        and seal["parameter_count"] == PARAMETERS
        and tests["construction"]["checks"]["parameter_count"]
        and tests["forbidden_components"]["passed"],
        "state_dict_keys_exact": tests["construction"]["checks"]["parameter_names_exact"]
        and tests["construction"]["checks"]["parameter_shapes_exact"]
        and tests["forbidden_components"]["passed"],
        "fixed_writers_preserved": tests["forbidden_components"]["fixed_writer_sources"] == [list(row) for row in core.FIXED_WRITERS] or tests["forbidden_components"]["fixed_writer_sources"] == core.FIXED_WRITERS,
        "b3_b5_lag_coverage_exact": tests["lag_coverage_nonoverlap"]["passed"],
        "local_recurrent_nonoverlap": tests["lag_coverage_nonoverlap"]["passed"],
        "causality_tests_passed": tests["causality"]["passed"]
        and tests["ring_index_mapping"]["passed"],
        "deployment_cache_tests_passed": tests["incremental_cache_reload_equivalence"]["passed"]
        and tests["incremental_cache_reload_equivalence"]["all_kv_caches_exact"]
        and tests["incremental_cache_reload_equivalence"]["all_h7_h8_h10_h12_rings_exact"]
        and tests["cache_capacity_eviction"]["passed"],
        "control_specificity_tests_passed": tests["control_specificity"]["passed"],
        "ce_only_objective": tests["attached_gradients"]["no_auxiliary_loss"] and all(row.get("ce_only") for row in evidence["training_rows"]),
        "attached_writer_gradients": tests["attached_gradients"]["passed"]
        and tests["representation_diagnostic_production_shape_feasibility"]["passed"]
        and representation["checks"]["attached_b3_b5_writer_gradients"],
        "analysis_input_identities_exact": evidence["analysis_inputs"]["passed"],
        "secondary_parallel_c0_c96_c191_completed": evidence["analysis_inputs"]["groups"]["secondary_parallel_evaluations"]
        and all(
            row.get("passed") is True
            for row in evidence["analysis_inputs"]["groups"]["secondary_parallel_evaluations"].values()
        ),
        "all_required_core_conditions_completed": core_complete,
        "all_14_large_conditions_completed": large_analysis.get("passed") is True
        and large_analysis["condition_validation"]["condition_count"] == 14
        and c_large["passed"] and fixed_large["passed"],
        "per_sequence_pairing_intact": large_losses["same_sequence_order_all_conditions"] and large_analysis["condition_validation"]["passed"],
        "large_targets_exact_every_condition": exact_large_targets,
        "historical_panel_disjointness_checked_where_possible": panel["all_required_disjointness_passed"]
        and historical_dataset_and_span_evidence_exact,
        "fourteen_condition_evaluation_not_reduced": len(large_losses["conditions"]) == 14,
        "memory_accounting_completed": memory["passed"],
        "final_checkpoint_strict_reopen_passed": seal["sealed"] and seal["checks"]["strict_reopen"],
        "remote_local_checkpoint_sha_match": backup["remote_sha256"] == backup["local_sha256"] == seal["checkpoint_sha256"],
        # Filled by the postflight command after scientific-results push/tag.
        "git_branch_commit_tag_pushed_verified": False,
        "worktree_clean": True,
        "no_a_b_or_250m_training": (
            scope["A_prohibited"] and scope["B_prohibited"]
            and scope["continuation_beyond_100m_prohibited"]
            and training["checks"]["no_training_beyond"]
        ),
    }
    return checks


def run_analyze(args):
    require_branch(clean=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Read every scientific input exactly once, then bind the in-memory values
    # to their frozen manifests and sealed checkpoint identities before doing
    # any calculation that could produce a scientific classification.
    parent_core = read_json(args.parent_core)
    c_core = {
        "0": read_json(args.c0_core),
        "48": read_json(args.c48_core),
        "96": read_json(args.c96_core),
        "144": read_json(args.c144_core),
        "191": read_json(args.c191_core),
    }
    fixed_core = read_json(args.fixed_core)
    c_large = read_json(args.c_large)
    fixed_large = read_json(args.fixed_large)
    representation_paths = {
        "parent": args.representation_parent,
        "c0": args.representation_c0,
        "c48": args.representation_c48,
        "c96": args.representation_c96,
        "c144": args.representation_c144,
        "c191": args.representation_c191,
        "fixed100m": args.representation_fixed,
    }
    representation_rows = {
        name: read_json(path) for name, path in representation_paths.items()
    }
    core_evaluations = {
        "parent": parent_core,
        "c0": c_core["0"],
        "c48": c_core["48"],
        "c96": c_core["96"],
        "c144": c_core["144"],
        "c191": c_core["191"],
        "fixed100m": fixed_core,
    }
    large_evaluations = {"c191": c_large, "fixed100m": fixed_large}
    parallel_evaluations = {
        "c0": read_json(args.c0_parallel),
        "c96": read_json(args.c96_parallel),
        "c191": read_json(args.c191_parallel),
    }
    core_manifest = read_json(args.core_panel_manifest)
    large_manifest = read_json(args.large_panel_manifest)
    shuffle_manifest = read_json(args.shuffle_manifest)
    milestones = read_json(args.milestone_manifest)
    freeze = read_json(args.pretrain_freeze_audit)
    training_rows = [
        json.loads(line) for line in Path(args.training_log).read_text().splitlines()
        if line.strip()
    ]
    evidence = {
        "scope": read_json(args.scope_lock),
        "source": read_json(args.source_provenance),
        "control": read_json(args.control_provenance),
        "replay": read_json(args.replay_audit),
        "preflight": read_json(args.preflight_audit),
        "preflight_tests": read_json(args.preflight_tests),
        "training": read_json(args.training_complete),
        "training_rows": training_rows,
        "restart": read_json(args.restart_audit),
        "seal": read_json(args.final_checkpoint_seal),
        "large_panel": large_manifest,
        "local_backup": read_json(args.local_backup_audit),
        "c_core": c_core,
        "fixed_core": fixed_core,
        "c_large": c_large,
        "fixed_large": fixed_large,
    }
    file_sha256s = {
        "core_manifest": sha256(args.core_panel_manifest),
        "large_manifest": sha256(args.large_panel_manifest),
        "shuffle_manifest": sha256(args.shuffle_manifest),
        "scope_lock": sha256(args.scope_lock),
        "source_provenance": sha256(args.source_provenance),
        "control_provenance": sha256(args.control_provenance),
        "replay_audit": sha256(args.replay_audit),
        "milestone_manifest": sha256(args.milestone_manifest),
    }
    input_audit = analysis_input_identity_audit(
        core_evaluations, large_evaluations, parallel_evaluations,
        representation_rows, core_manifest, large_manifest, shuffle_manifest,
        milestones, freeze, evidence, file_sha256s,
    )
    durable_json(output / "ANALYSIS_INPUT_IDENTITY_AUDIT.json", input_audit)
    if not input_audit["passed"]:
        raise SystemExit(
            "analysis input identity audit failed; scientific classification withheld"
        )
    evidence["analysis_inputs"] = input_audit

    longitudinal = analysis.longitudinal_core_summary(
        c_core, fixed_milestones={191: fixed_core}
    )
    longitudinal["parent0_all_real_ce"] = parent_core["conditions"]["all_real"]["validation_loss"]
    durable_json(output / "TRUE_INCREMENTAL_LONGITUDINAL_CORE.json", longitudinal)
    recovery = analysis.adaptation_recovery_summary(
        all_real_sequence_ce(parent_core), all_real_sequence_ce(c_core["0"]),
        all_real_sequence_ce(c_core["191"]), all_real_sequence_ce(fixed_core),
    )
    durable_json(output / "ADAPTATION_RECOVERY.json", recovery)

    losses = large_loss_artifact(c_large, fixed_large)
    durable_json(output / "LARGE_FINAL_PER_SEQUENCE_LOSSES.json", losses)
    large = analysis.analyze_final_large_panel(
        c_large["conditions"], fixed_large["conditions"],
        seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
    )
    durable_json(output / "LARGE_FINAL_BOOTSTRAP.json", large)

    representation = combine_representation_artifacts(
        representation_paths, rows=representation_rows
    )
    durable_json(output / "REPRESENTATION_PRESSURE_DIAGNOSTICS.json", representation)

    memory_raw = read_json(args.memory_audit)
    memory = analysis.compare_bf16_persistent_state(
        memory_raw["fixed"], memory_raw["c"], batch_size=1, width=768
    )
    durable_json(output / "BF16_PERSISTENT_STATE_AUDIT.json", memory)

    checks = scientific_audit_checks(
        evidence, losses, large, longitudinal, representation, memory
    )
    protocol_critical_passed = all(
        checks[name] for name in analysis.CRITICAL_INVALID_AUDIT_CHECKS
    )
    post_analysis_operational = {"git_branch_commit_tag_pushed_verified"}
    substantive_failures = [
        name for name, value in checks.items()
        if not value and name not in post_analysis_operational
    ]
    scientific_integrity_passed = not substantive_failures
    critical_passed = protocol_critical_passed and scientific_integrity_passed
    provisional_audit = {
        "experiment": EXPERIMENT,
        "phase": "scientific-results-pretag",
        "checks": checks,
        "protocol_critical_checks_passed": protocol_critical_passed,
        "critical_scientific_checks_passed": critical_passed,
        "all_substantive_scientific_integrity_checks_passed": scientific_integrity_passed,
        "substantive_failures": substantive_failures,
        "passed": all(checks.values()),
        "pending_operational_checks": [
            name for name, value in checks.items()
            if not value and name in post_analysis_operational
        ],
    }
    durable_json(output / "SCIENTIFIC_AUDIT_PRETAG.json", provisional_audit)
    if not critical_passed:
        classification = analysis.classification_decision(
            large["bootstrap"], audit_passed=False
        )
    else:
        recovery_row = recovery["recovery_fraction"]
        meaningful_recovery = bool(
            recovery_row.get("defined")
            and recovery_row.get("point_estimate") is not None
            and recovery_row["point_estimate"] >= 0.5
            and recovery_row.get("lower_95") is not None
            and recovery_row["lower_95"] > 0.0
        )
        classification = analysis.classification_decision(
            large["bootstrap"], audit_passed=True,
            meaningful_recovery=meaningful_recovery,
        )
        classification["meaningful_recovery_frozen_rule"] = {
            "point_estimate_at_least": 0.5,
            "paired_95_ci_lower_above": 0.0,
            "result": meaningful_recovery,
        }
    recommendation = analysis.recommendation_after_c(
        classification["classification"], large["bootstrap"]
    )
    classification["recommendation"] = recommendation
    durable_json(output / "CLASSIFICATION.json", classification)
    if not critical_passed:
        raise SystemExit(
            "substantive scientific/integrity audit failed; classification withheld"
        )

    contrasts = large["bootstrap"]["contrasts"]
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification["classification"],
        "recommendation": recommendation,
        "fixed_all_real_ce": fixed_large["conditions"]["all_real"]["validation_loss"],
        "c_all_real_ce": c_large["conditions"]["all_real"]["validation_loss"],
        "bootstrap": large["bootstrap"],
        "longitudinal": longitudinal,
        "recovery": recovery,
        "bf16_persistent_state": memory,
        "representation_diagnostics_path": "REPRESENTATION_PRESSURE_DIAGNOSTICS.json",
        "final_checkpoint": {
            "path": evidence["seal"]["checkpoint"],
            "sha256": evidence["seal"]["checkpoint_sha256"],
            "local_archive_path": evidence["local_backup"]["local_path"],
        },
        "audit": provisional_audit,
        "git_branch": BRANCH,
        "git_commit": git("rev-parse", "HEAD"),
        "git_tag": FINAL_TAG,
        "runpod_status": "RUNNING",
        "pod_id": POD_ID,
        "volume_id": VOLUME_ID,
        "primary_contrasts": {
            name: contrasts[name] for name in (
                "architecture_fixed_minus_c", "architecture_c_minus_fixed_penalty",
                "c_b3_off_gain", "c_b3_sequence_gap", "c_b5_off_gain",
                "c_b5_sequence_gap", "c_combined_off_gain",
                "c_combined_sequence_gap", "b3_off_gain_lift",
                "b3_sequence_gap_lift", "b5_off_gain_lift",
                "b5_sequence_gap_lift", "combined_off_gain_lift",
                "combined_sequence_gap_lift",
            )
        },
    }
    durable_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    print(f"EXPERIMENT_2D5C_ANALYSIS_COMPLETE {classification['classification']}", flush=True)


def ci_text(row):
    return f"[{row['lower_95']:+.12f}, {row['upper_95']:+.12f}]"


def run_render_report(args):
    if finalizer.FINAL_PHRASE != "STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS":
        raise SystemExit("terminal report phrase binding failed")
    finalizer.render_report(
        summary_path=args.summary,
        representation_path=args.representation,
        postflight_audit_path=args.postflight_audit,
        output_path=args.output_path,
    )
    print("EXPERIMENT_2D5C_FINAL_REPORT_RENDERED", flush=True)


def run_postflight_audit(args):
    finalizer.run_postflight(
        provisional_audit_path=args.provisional_audit,
        summary_path=args.summary,
        representation_path=args.representation,
        git_verification_path=args.git_verification,
        stop_verification_path=args.stop_verification,
        guard_authorization_path=args.guard_authorization,
        guard_trigger_path=args.guard_trigger,
        output_path=args.output_path,
    )
    print("EXPERIMENT_2D5C_FINAL_AUDIT_PASS", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fail-closed Experiment 2D5C scientific driver"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.set_defaults(handler=run_prepare)
    for name in (
        "output_dir", "source_checkpoint", "control_checkpoint",
        "control_search_root", "fixed_replay_manifest", "data_root",
        "d3a_large_manifest", "d4a_large_manifest", "d4a250_large_manifest",
    ):
        prepare.add_argument(f"--{name.replace('_', '-')}", required=True)
    prepare.add_argument("--pod-id", default=POD_ID, choices=(POD_ID,))
    prepare.add_argument("--pod-name", default=POD_NAME, choices=(POD_NAME,))
    prepare.add_argument("--volume-id", default=VOLUME_ID, choices=(VOLUME_ID,))

    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(handler=run_preflight)
    for name in ("output_dir", "pretrain_dir", "source_checkpoint", "data_root"):
        preflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    preflight.add_argument("--stop-capability-verified", action="store_true")
    preflight.add_argument("--storage-inventory-verified", action="store_true")
    preflight.add_argument("--network-volume-free-bytes", required=True, type=int)
    preflight.add_argument("--pod-id", default=POD_ID, choices=(POD_ID,))
    preflight.add_argument("--pod-name", default=POD_NAME, choices=(POD_NAME,))
    preflight.add_argument("--volume-id", default=VOLUME_ID, choices=(VOLUME_ID,))

    train = subparsers.add_parser("train")
    train.set_defaults(handler=run_train)
    train.add_argument("--arm", required=True, choices=("C",))
    train.add_argument("--end-local-update", required=True, type=int, choices=(96, 191))
    for name in (
        "output_dir", "preflight_audit", "replay_ledger", "replay_audit",
        "source_checkpoint", "large_panel", "scientific_checkpoint_dir",
    ):
        train.add_argument(f"--{name.replace('_', '-')}", required=True)
    train.add_argument("--resume-checkpoint")
    train.add_argument("--midpoint-preexit")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.set_defaults(handler=run_evaluate)
    evaluate.add_argument("--family", required=True, choices=("Parent", "C0", "C", "Fixed"))
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--source-checkpoint", required=True)
    evaluate.add_argument("--expected-local-update", required=True, type=int)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--panel-manifest", required=True)
    evaluate.add_argument("--shuffle-manifest", required=True)
    evaluate.add_argument("--pretrain-freeze-audit", required=True)
    evaluate.add_argument("--panel-kind", required=True, choices=("core", "large"))
    evaluate.add_argument("--output-path", required=True)
    evaluate.add_argument("--all-real-only", action="store_true")
    evaluate.add_argument("--parallel-output")
    evaluate.add_argument("--parallel-passes", type=int, choices=(2, 3), default=2)
    evaluate.add_argument("--final-checkpoint-seal")
    evaluate.add_argument("--milestone-manifest")

    seal = subparsers.add_parser("seal-final")
    seal.set_defaults(handler=run_seal_final)
    for name in (
        "checkpoint", "source_checkpoint", "training_complete", "training_log",
        "training_replay_actual", "replay_ledger", "replay_audit",
        "midpoint_restart_preexit", "midpoint_restart_audit",
        "milestone_manifest", "output_path",
    ):
        seal.add_argument(f"--{name.replace('_', '-')}", required=True)

    memory = subparsers.add_parser("memory-audit")
    memory.set_defaults(handler=run_memory_audit)
    for name in (
        "fixed_checkpoint", "c_checkpoint", "source_checkpoint",
        "output_json", "output_table", "milestone_manifest",
        "final_checkpoint_seal",
    ):
        memory.add_argument(f"--{name.replace('_', '-')}", required=True)

    representation = subparsers.add_parser("representation-diagnostics")
    representation.set_defaults(handler=run_representation_diagnostics)
    representation.add_argument("--family", required=True, choices=("Parent", "C0", "C", "Fixed"))
    representation.add_argument("--checkpoint")
    representation.add_argument("--source-checkpoint", required=True)
    representation.add_argument("--expected-local-update", required=True, type=int)
    representation.add_argument("--label", required=True)
    representation.add_argument("--data-root", required=True)
    representation.add_argument("--core-manifest", required=True)
    representation.add_argument("--pretrain-freeze-audit", required=True)
    representation.add_argument("--output-dir", required=True)
    representation.add_argument("--output-json", required=True)
    representation.add_argument("--milestone-manifest")
    representation.add_argument("--final-checkpoint-seal")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.set_defaults(handler=run_analyze)
    for name in (
        "output_dir", "parent_core", "c0_core", "c48_core", "c96_core",
        "c144_core", "c191_core", "fixed_core", "c_large", "fixed_large",
        "representation_parent", "representation_c0", "representation_c48",
        "representation_c96", "representation_c144", "representation_c191",
        "representation_fixed", "memory_audit", "training_log", "scope_lock",
        "source_provenance", "control_provenance", "replay_audit",
        "preflight_audit", "preflight_tests", "training_complete",
        "restart_audit", "final_checkpoint_seal", "large_panel_manifest",
        "local_backup_audit", "pretrain_freeze_audit", "core_panel_manifest",
        "shuffle_manifest", "milestone_manifest", "c0_parallel",
        "c96_parallel", "c191_parallel",
    ):
        analyze_parser.add_argument(f"--{name.replace('_', '-')}", required=True)

    report = subparsers.add_parser("render-report")
    report.set_defaults(handler=run_render_report)
    for name in ("summary", "representation", "postflight_audit", "output_path"):
        report.add_argument(f"--{name.replace('_', '-')}", required=True)

    postflight = subparsers.add_parser("postflight-audit")
    postflight.set_defaults(handler=run_postflight_audit)
    for name in (
        "provisional_audit", "summary", "representation", "git_verification",
        "stop_verification", "guard_authorization", "guard_trigger", "output_path",
    ):
        postflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if args.end_local_update == 96 and (args.resume_checkpoint or args.midpoint_preexit):
            raise SystemExit("segment 0->96 cannot accept resume artifacts")
        if args.end_local_update == 191 and not (args.resume_checkpoint and args.midpoint_preexit):
            raise SystemExit("segment 96->191 requires both sealed midpoint artifacts")
    if args.command == "evaluate":
        if (args.panel_kind == "large" and args.family == "C"
                and not args.final_checkpoint_seal):
            raise SystemExit("C large evaluation requires --final-checkpoint-seal")
        protocol = evaluation_protocol_checks(
            args.family, args.panel_kind, args.expected_local_update,
            args.all_real_only, args.parallel_output,
            args.final_checkpoint_seal, args.milestone_manifest,
        )
        if not protocol["passed"]:
            raise SystemExit(f"evaluation protocol matrix failed: {protocol['checks']}")
    if args.command == "representation-diagnostics":
        if args.family == "C" and not args.milestone_manifest:
            raise SystemExit("C diagnostics require --milestone-manifest")
        if args.family != "C" and (args.milestone_manifest or args.final_checkpoint_seal):
            raise SystemExit("checkpoint binding arguments apply only to C diagnostics")
    args.handler(args)


if __name__ == "__main__":
    main()
