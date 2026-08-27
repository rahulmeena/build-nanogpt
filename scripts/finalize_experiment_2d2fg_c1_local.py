#!/usr/bin/env python3
"""Finish 2D2FG-C1 plots and publication after the GPU pod is stopped.

The scientific evaluator deliberately remains the single source of truth for
plot and report rendering.  This utility extracts only those two pure
post-processing functions from its AST, so it needs NumPy/Matplotlib but not
PyTorch or a CUDA runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


CONSTANTS = {
    "REQUIRED_FILES",
    "F_STATE_BYTES",
    "G_STATE_BYTES",
    "T",
    "BATCH",
    "BATCHES",
    "SEQUENCES",
    "TARGETS",
}
FUNCTIONS = {"make_plots", "report_text"}
POSITION_ORDER = ("1-31", "32-63", "64-127", "128-255", "256-511", "512-767", "768-1023")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def load_renderers(source_path: Path) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = []
    for node in tree.body:
        name = assignment_name(node)
        if name in CONSTANTS or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            body.append(node)
    namespace = {"Path": Path, "np": np, "json": json}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    missing = (CONSTANTS | FUNCTIONS) - namespace.keys()
    if missing:
        raise SystemExit(f"renderer extraction missing: {sorted(missing)}")
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluator-source", required=True)
    parser.add_argument("--stopped-utc", required=True)
    parser.add_argument("--results-commit")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    renderers = load_renderers(Path(args.evaluator_source).resolve())
    required_files = renderers["REQUIRED_FILES"]
    summary = json.loads((output / "result_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise SystemExit("scientific FINAL_AUDIT.json must pass before local finalization")

    summary["artifact_path"] = str(output)
    summary["position_bins"] = {label: summary["position_bins"][label] for label in POSITION_ORDER}
    summary["pod"].update(
        {
            "status": "EXITED",
            "runtime_status": "stopped",
            "runtime_status_reason": "stopped_by_user",
            "stopped_utc": args.stopped_utc,
            "persistent_volume_retained": True,
        }
    )
    audit["pod_terminal_status"] = {
        "id": summary["pod"]["id"],
        "name": summary["pod"]["name"],
        "desired_status": "EXITED",
        "runtime_status": "stopped",
        "runtime_status_reason": "stopped_by_user",
        "stopped_utc": args.stopped_utc,
        "persistent_volume_id": summary["pod"]["persistent_volume_id"],
        "persistent_volume_retained": True,
    }
    if args.results_commit:
        summary["git"]["results_commit"] = args.results_commit
        audit["results_commit"] = args.results_commit

    atomic_json(output / "result_summary.json", summary)
    atomic_json(output / "FINAL_AUDIT.json", audit)
    atomic_json(output / "runpod_terminal_status.json", audit["pod_terminal_status"])
    renderers["make_plots"](output, summary)

    report = renderers["report_text"](summary, audit, args.results_commit)
    pending = (
        f"- Pod `{summary['pod']['id']}` is `EXITED` pending terminal publication/backup stop."
    )
    terminal = (
        f"- Pod `{summary['pod']['id']}` was stopped at `{args.stopped_utc}` after a verified local backup."
    )
    if pending not in report:
        raise SystemExit("expected report pod-status sentence was not found")
    report = report.replace(pending, terminal)
    disclosure = (
        "## Post-GPU finalization disclosure\n\n"
        "The frozen scientific evaluation and terminal scientific audit completed on the pod. "
        "The pod image lacked Matplotlib, so the already-written scientific JSON was checksum-copied locally, "
        "the pod was stopped, and only plots/report publication were completed locally. No model execution, "
        "training, checkpoint, subset, or scientific metric changed.\n\n"
    )
    report = report.replace("## Integrity, Git, and runtime\n", disclosure + "## Integrity, Git, and runtime\n")
    atomic_text(output / "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md", report)
    atomic_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        "# 2D2FG-C1 final handoff\n\n"
        f"Scientific evaluation and final audit passed. Classification: **{summary['absolute_quality_classification']}**.\n\n"
        f"Pod `{summary['pod']['id']}` is stopped. Persistent volume `{summary['pod']['persistent_volume_id']}` is retained. "
        "The complete artifact set is backed up locally and publication work is CPU-only.\n",
    )

    missing = [
        name for name in required_files
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing:
        raise SystemExit(f"required artifacts missing after local finalization: {missing}")

    heartbeat = {
        "schema": "2d2fg_c1_heartbeat_v1",
        "experiment": "2D2FG-C1",
        "phase": "results_sealed" if args.results_commit else "local_reporting_complete",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": True,
    }
    if args.results_commit:
        heartbeat["results_commit"] = args.results_commit
    atomic_json(output / "HEARTBEAT.json", heartbeat)

    if args.results_commit:
        inventory = {
            name: {"bytes": (output / name).stat().st_size, "sha256": file_sha256(output / name)}
            for name in required_files
        }
        atomic_json(output / "artifact_inventory.json", {"files": inventory, "passed": True})
    print(output / "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md")


if __name__ == "__main__":
    main()
