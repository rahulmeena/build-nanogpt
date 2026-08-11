#!/usr/bin/env python3
"""Compare uninterrupted and save/restart Experiment 1B trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def tensor_mapping_comparison(left, right):
    left_keys = set(left)
    right_keys = set(right)
    report = {
        "left_keys": len(left_keys),
        "right_keys": len(right_keys),
        "missing_from_left": sorted(right_keys - left_keys),
        "missing_from_right": sorted(left_keys - right_keys),
        "shape_or_dtype_mismatches": [],
        "unequal_tensors": 0,
        "unequal_elements": 0,
        "maximum_absolute_difference": 0.0,
    }
    for key in sorted(left_keys & right_keys, key=str):
        a = left[key]
        b = right[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"{key!r} is not a tensor in both mappings")
        if a.shape != b.shape or a.dtype != b.dtype:
            report["shape_or_dtype_mismatches"].append(str(key))
            continue
        if torch.equal(a, b):
            continue
        report["unequal_tensors"] += 1
        report["unequal_elements"] += int(torch.count_nonzero(a != b).item())
        if a.numel() and (a.is_floating_point() or a.is_complex()):
            difference = (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
            report["maximum_absolute_difference"] = max(
                report["maximum_absolute_difference"], difference
            )
    report["bit_exact"] = not any([
        report["missing_from_left"],
        report["missing_from_right"],
        report["shape_or_dtype_mismatches"],
        report["unequal_tensors"],
    ])
    return report


def optimizer_tensor_mapping(checkpoint):
    output = {}
    for parameter_id, state in checkpoint["optimizer"]["state"].items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                output[f"{parameter_id}:{name}"] = value
    return output


def nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def read_metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-a-run", required=True)
    parser.add_argument("--path-b-run", required=True)
    parser.add_argument("--path-a-checkpoint", required=True)
    parser.add_argument("--path-b-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left = load_checkpoint(args.path_a_checkpoint)
    right = load_checkpoint(args.path_b_checkpoint)
    model = tensor_mapping_comparison(left["model"], right["model"])
    optimizer = tensor_mapping_comparison(
        optimizer_tensor_mapping(left), optimizer_tensor_mapping(right)
    )

    left_rows = {row["step"]: row for row in read_metrics(Path(args.path_a_run) / "metrics.jsonl") if row["kind"] == "train"}
    right_rows = {row["step"]: row for row in read_metrics(Path(args.path_b_run) / "metrics.jsonl") if row["kind"] == "train"}
    trajectory = []
    for step in sorted(set(left_rows) | set(right_rows)):
        a = left_rows.get(step)
        b = right_rows.get(step)
        trajectory.append({
            "step": step,
            "present_in_both": a is not None and b is not None,
            "processed_tokens_equal": a is not None and b is not None and a["tokens"] == b["tokens"],
            "lr_equal": a is not None and b is not None and a["lr"] == b["lr"],
            "loss_equal": a is not None and b is not None and a["train_loss"] == b["train_loss"],
            "loss_absolute_difference": None if a is None or b is None else abs(a["train_loss"] - b["train_loss"]),
            "global_batch_hash_equal": a is not None and b is not None and a.get("global_batch_sha256") == b.get("global_batch_sha256"),
        })

    training_state_equal = nested_equal(left["training_state"], right["training_state"])
    dataloader_equal = nested_equal(left["dataloader_states"], right["dataloader_states"])
    rng_equal = nested_equal(left["rng_states"], right["rng_states"])
    optimizer_groups_equal = nested_equal(
        left["optimizer"]["param_groups"], right["optimizer"]["param_groups"]
    )
    next_hash_equal = left["next_global_batch_sha256"] == right["next_global_batch_sha256"]
    trajectory_exact = all(
        row["present_in_both"]
        and row["processed_tokens_equal"]
        and row["lr_equal"]
        and row["loss_equal"]
        and row["global_batch_hash_equal"]
        for row in trajectory
    )
    report = {
        "path_a_checkpoint": str(Path(args.path_a_checkpoint).resolve()),
        "path_b_checkpoint": str(Path(args.path_b_checkpoint).resolve()),
        "completed_updates_a": left["training_state"]["completed_updates"],
        "completed_updates_b": right["training_state"]["completed_updates"],
        "processed_tokens_a": left["training_state"]["processed_tokens"],
        "processed_tokens_b": right["training_state"]["processed_tokens"],
        "model": model,
        "optimizer_tensors": optimizer,
        "optimizer_param_groups_bit_exact": optimizer_groups_equal,
        "training_state_bit_exact": training_state_equal,
        "dataloader_states_bit_exact": dataloader_equal,
        "rng_states_bit_exact": rng_equal,
        "next_global_batch_hash_equal": next_hash_equal,
        "trajectory": trajectory,
        "trajectory_bit_exact": trajectory_exact,
    }
    report["passed"] = all([
        left["training_state"]["completed_updates"] == 10,
        right["training_state"]["completed_updates"] == 10,
        model["bit_exact"],
        optimizer["bit_exact"],
        optimizer_groups_equal,
        training_state_equal,
        dataloader_equal,
        rng_equal,
        next_hash_equal,
        trajectory_exact,
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("exact-resume comparison failed")


if __name__ == "__main__":
    main()
