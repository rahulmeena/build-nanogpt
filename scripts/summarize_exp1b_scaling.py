#!/usr/bin/env python3
"""Summarize four-GPU scaling runs against the frozen one-GPU references."""

import argparse
import csv
import json
import statistics
from pathlib import Path


REFERENCES = {"standard": 168645.0, "full_attnres": 21257.0}


def metrics(path):
    return [
        json.loads(line)
        for line in (Path(path) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def monitor(path):
    rows = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            try:
                rows.append({
                    "gpu_utilization_percent": float(row["utilization.gpu [%]"].split()[0]),
                    "memory_used_mib": float(row["memory.used [MiB]"].split()[0]),
                    "power_watts": float(row["power.draw [W]"].split()[0]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def summarize(mode, run_dir, monitor_path):
    train = [row for row in metrics(run_dir) if row["kind"] == "train"]
    steady = [row for row in train if row["step"] >= 2]
    samples = monitor(monitor_path)
    throughput = statistics.mean(row["tokens_per_second"] for row in steady)
    reference = REFERENCES[mode]
    return {
        "mode": mode,
        "updates": len(train),
        "steady_state_updates": len(steady),
        "four_gpu_tokens_per_second": throughput,
        "four_gpu_seconds_per_update": statistics.mean(row["step_time_ms"] for row in steady) / 1000,
        "one_gpu_reference_tokens_per_second": reference,
        "speedup_over_one_gpu": throughput / reference,
        "four_gpu_scaling_efficiency": throughput / (4 * reference),
        "peak_allocated_mib_per_gpu": max(row["peak_allocated_mb_per_gpu_max"] for row in train),
        "peak_reserved_mib_per_gpu": max(row["peak_reserved_mb_per_gpu_max"] for row in train),
        "monitor_samples": len(samples),
        "mean_gpu_utilization_percent": statistics.mean(row["gpu_utilization_percent"] for row in samples) if samples else None,
        "minimum_gpu_utilization_percent": min(row["gpu_utilization_percent"] for row in samples) if samples else None,
        "maximum_memory_used_mib": max(row["memory_used_mib"] for row in samples) if samples else None,
        "mean_power_watts": statistics.mean(row["power_watts"] for row in samples) if samples else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-run", required=True)
    parser.add_argument("--standard-monitor", required=True)
    parser.add_argument("--attnres-run", required=True)
    parser.add_argument("--attnres-monitor", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {
        "standard": summarize("standard", args.standard_run, args.standard_monitor),
        "full_attnres": summarize("full_attnres", args.attnres_run, args.attnres_monitor),
    }
    report["passed"] = all(
        arm["updates"] == 10
        and arm["steady_state_updates"] == 8
        and arm["four_gpu_scaling_efficiency"] >= 0.25
        for arm in report.values()
        if isinstance(arm, dict)
    )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("pathological four-GPU scaling detected")


if __name__ == "__main__":
    main()
