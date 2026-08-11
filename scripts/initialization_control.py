#!/usr/bin/env python3
"""Create and verify the canonical seed-1337 GPT-2 step-0 initialization."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_TAG = "baseline-gpt2-124m-10b"
BASELINE_COMMIT = "a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb"
SEED = 1337


def definition_prefix(source):
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    if marker not in source:
        raise RuntimeError("could not find training launch marker")
    return source.split(marker)[0]


def load_symbols(source, filename, namespace_name):
    namespace = {
        "__name__": namespace_name,
        "__file__": filename,
        "master_process": True,
    }
    sys.path.insert(0, str(REPO_ROOT))
    exec(compile(definition_prefix(source), filename, "exec"), namespace)
    namespace["master_process"] = True
    return namespace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiment_artifacts/baseline_init_seed1337.pt")
    parser.add_argument("--report", default="reports/initialization_equivalence.json")
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", f"{BASELINE_TAG}^{{commit}}"], cwd=REPO_ROOT, text=True).strip()
    if head != BASELINE_COMMIT:
        raise SystemExit(f"baseline tag mismatch: expected {BASELINE_COMMIT}, found {head}")

    baseline_source = subprocess.check_output(
        ["git", "show", f"{BASELINE_TAG}:train_gpt2.py"],
        cwd=REPO_ROOT,
        text=True,
    )
    baseline = load_symbols(baseline_source, f"{BASELINE_TAG}:train_gpt2.py", "frozen_baseline")

    torch.manual_seed(SEED)
    baseline_model = baseline["GPT"](baseline["GPTConfig"](vocab_size=50304))
    baseline_state = baseline_model.state_dict()

    checkpoint_path = REPO_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": baseline_state,
        "config": {
            "block_size": 1024,
            "vocab_size": 50304,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "residual_mode": "standard",
        },
        "seed": SEED,
        "baseline_tag": BASELINE_TAG,
        "baseline_commit": BASELINE_COMMIT,
    }, checkpoint_path)

    current_source = (REPO_ROOT / "train_gpt2.py").read_text()
    current = load_symbols(current_source, str(REPO_ROOT / "train_gpt2.py"), "experiment_model")
    standard_model = current["GPT"](current["GPTConfig"](vocab_size=50304, residual_mode="standard"))
    standard_model.load_state_dict(baseline_state, strict=True)

    full_model = current["GPT"](current["GPTConfig"](vocab_size=50304, residual_mode="full_attnres"))
    full_model.load_shared_baseline_state(baseline_state)

    full_state = full_model.state_dict()
    shared_keys = sorted(baseline_state)
    exact = 0
    mismatches = []
    maximum_difference = 0.0
    for key in shared_keys:
        baseline_tensor = baseline_state[key]
        full_tensor = full_state[key]
        if torch.equal(baseline_tensor, full_tensor):
            exact += 1
        else:
            mismatches.append(key)
            maximum_difference = max(
                maximum_difference,
                (baseline_tensor.float() - full_tensor.float()).abs().max().item(),
            )

    query_tensors = [value for key, value in full_state.items() if key.endswith(".query")]
    norm_tensors = [value for key, value in full_state.items() if ".attnres." in key and key.endswith(".norm.weight")]
    if len(query_tensors) != 25 or not all(torch.count_nonzero(value) == 0 for value in query_tensors):
        raise SystemExit("AttnRes pseudo-query initialization is not exactly zero")
    if len(norm_tensors) != 25 or not all(torch.equal(value, torch.ones_like(value)) for value in norm_tensors):
        raise SystemExit("AttnRes RMSNorm initialization is not exactly one")

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    full_params = sum(p.numel() for p in full_model.parameters())
    query_params = sum(value.numel() for value in query_tensors)
    norm_params = sum(value.numel() for value in norm_tensors)
    report = {
        "baseline_commit": BASELINE_COMMIT,
        "baseline_tag": BASELINE_TAG,
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "shared_tensors": len(shared_keys),
        "exact_matches": exact,
        "mismatches": len(mismatches),
        "mismatch_keys": mismatches,
        "maximum_absolute_difference": maximum_difference,
        "baseline_parameters": baseline_params,
        "full_attnres_parameters": full_params,
        "added_parameters": full_params - baseline_params,
        "percentage_increase": 100 * (full_params - baseline_params) / baseline_params,
        "parameter_breakdown": {
            "all_depth_queries": query_params,
            "all_rmsnorm_scales": norm_params,
            "pre_sublayer_queries": query_params - 768,
            "pre_sublayer_rmsnorm_scales": norm_params - 768,
            "final_aggregator_query_and_rmsnorm": 2 * 768,
            "other": full_params - baseline_params - query_params - norm_params,
        },
    }

    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit("shared GPT-2 initialization mismatch")


if __name__ == "__main__":
    main()
