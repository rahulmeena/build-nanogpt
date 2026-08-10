#!/usr/bin/env python3
"""Inspect edu_fineweb10B shards using the same split discovery as train_gpt2.py."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tiktoken


SHARD_SIZE = int(1e8)


def human_bytes(nbytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def upstream_shards(data_dir, split):
    filenames = os.listdir(data_dir)
    shards = [s for s in filenames if split in s]
    shards = sorted(shards)
    return [Path(data_dir) / s for s in shards]


def inspect_shard(path):
    arr = np.load(path, mmap_mode="r")
    return {
        "filename": path.name,
        "path": str(path),
        "tokens": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "disk_size_bytes": path.stat().st_size,
        "disk_size": human_bytes(path.stat().st_size),
    }


def shard_index(path):
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except Exception:
        return 10**12


def decode_sample(arr, start, length):
    enc = tiktoken.get_encoding("gpt2")
    start = max(0, min(start, max(0, arr.shape[0] - 1)))
    end = min(arr.shape[0], start + length)
    token_ids = arr[start:end].astype(np.int64).tolist()
    return {
        "start": int(start),
        "end": int(end),
        "token_ids": token_ids,
        "decoded": enc.decode(token_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="edu_fineweb10B")
    parser.add_argument("--out", help="write report JSON to this file")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--expected-shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--min-total-tokens", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    failures = []
    if not data_dir.exists():
        raise SystemExit(f"dataset directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"dataset path is not a directory: {data_dir}")

    val_paths = upstream_shards(data_dir, "val")
    train_paths = upstream_shards(data_dir, "train")
    all_paths = sorted(val_paths + train_paths, key=shard_index)
    if not all_paths:
        failures.append("no .npy shards found by upstream split discovery")

    shard_infos = [inspect_shard(path) for path in all_paths]
    total_tokens = sum(info["tokens"] for info in shard_infos)
    total_disk = sum(info["disk_size_bytes"] for info in shard_infos)
    dtypes = sorted({info["dtype"] for info in shard_infos})

    if args.strict:
        if len(val_paths) != 1:
            failures.append(f"expected exactly one validation shard, found {len(val_paths)}")
        if len(train_paths) == 0:
            failures.append("expected at least one training shard")
        if dtypes != ["uint16"]:
            failures.append(f"expected only uint16 shards, found {dtypes}")
        if len(all_paths) >= 2:
            for info in shard_infos[:-1]:
                if info["tokens"] != args.expected_shard_size:
                    failures.append(f"{info['filename']} has {info['tokens']} tokens, expected {args.expected_shard_size}")
            if shard_infos[-1]["tokens"] > args.expected_shard_size:
                failures.append(f"last shard has {shard_infos[-1]['tokens']} tokens, exceeds {args.expected_shard_size}")
        if args.min_total_tokens is not None and total_tokens < args.min_total_tokens:
            failures.append(f"total tokens {total_tokens} below required minimum {args.min_total_tokens}")
        if args.max_total_tokens is not None and total_tokens > args.max_total_tokens:
            failures.append(f"total tokens {total_tokens} above required maximum {args.max_total_tokens}")

    samples = {}
    if all_paths:
        first = np.load(all_paths[0], mmap_mode="r")
        samples["first_32_token_ids"] = first[:32].astype(np.int64).tolist()
        samples["decoded_first_sample"] = decode_sample(first, 0, 160)
    if len(all_paths) >= 2:
        a = np.load(all_paths[0], mmap_mode="r")
        b = np.load(all_paths[1], mmap_mode="r")
        boundary_ids = np.concatenate([a[-16:].astype(np.int64), b[:16].astype(np.int64)]).tolist()
        enc = tiktoken.get_encoding("gpt2")
        samples["near_first_shard_boundary"] = {
            "token_ids": boundary_ids,
            "decoded": enc.decode(boundary_ids),
        }
    if train_paths:
        later_path = train_paths[min(5, len(train_paths) - 1)]
        later = np.load(later_path, mmap_mode="r")
        samples["decoded_later_sample"] = {
            "filename": later_path.name,
            **decode_sample(later, min(10000, max(0, later.shape[0] - 160)), 160),
        }

    report = {
        "data_dir": str(data_dir),
        "number_of_shard_files": len(all_paths),
        "validation_shards_upstream": [path.name for path in val_paths],
        "training_shards_upstream": [path.name for path in train_paths],
        "shards": shard_infos,
        "dtype": dtypes,
        "total_token_count": total_tokens,
        "total_disk_size_bytes": total_disk,
        "total_disk_size": human_bytes(total_disk),
        "samples": samples,
        "failures": failures,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    if failures:
        raise SystemExit("dataset verification failed")


if __name__ == "__main__":
    main()
