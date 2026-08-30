#!/usr/bin/env python3
"""Supervise, stop, finalize, and terminally commit Experiment 2D5C.

This is the stdlib-only host entrypoint used after the implementation and
pretraining-input freeze commits are already pushed.  The RunPod guard
supervises the complete scientific child workflow and writes its final report
to a dedicated JSON file, separate from inherited child stdout.  Only a
successful, exact stopped-pod report is allowed to enter the local finalizer.

The immutable experiment tag intentionally remains on the sealed scientific
results commit.  A subsequent terminal postflight commit adds the verified
stop evidence, FINAL_AUDIT, and final report, matching the accepted 2D4A
scientific-tag/terminal-postflight pattern.  This script verifies that the
branch points at the terminal commit, the tag still peels to the scientific
commit, and the worktree is clean; that last verification is stored outside
the Git worktree to avoid a self-referential dirty-tree artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


EXPERIMENT = "2D5C"
BRANCH = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m"
FINAL_TAG = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final"
LOCAL_REPO = Path(
    "/Users/rahul/Documents/GPT-2 Enhancement/"
    "parallel_2d2_master_dev/2d3a_1b"
)
LOCAL_RESULTS = (
    LOCAL_REPO / "results" /
    "experiment_2d5c_fixed_writer_b3_b5_w2_matched_100m"
)
LOCAL_ARCHIVE = Path(
    "/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/"
    "experiment_2d5c_fixed_writer_b3_b5_w2_matched_100m"
)


class CompletionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(argv: Sequence[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        [str(value) for value in argv],
        cwd=None if cwd is None else str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode != 0:
        if capture and result.stdout:
            print(result.stdout, end="", flush=True)
        raise CompletionError(
            f"command failed with exit {result.returncode}: {argv[0]}"
        )
    return "" if result.stdout is None else result.stdout.strip()


def git(*args: str) -> str:
    return run(["git", *args], cwd=LOCAL_REPO, capture=True)


def require_fresh_path(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise CompletionError(f"{label} path must be absolute")
    if path.exists():
        raise CompletionError(f"refusing to reuse existing {label}: {path}")
    if not path.parent.is_dir():
        raise CompletionError(f"missing {label} parent: {path.parent}")


def run_guarded_workflow(args: argparse.Namespace) -> None:
    require_fresh_path(args.trigger_file, "guard trigger")
    require_fresh_path(args.stop_report, "guard stop report")
    if not args.authorization_artifact.is_file():
        raise CompletionError("missing private guard authorization")
    if git("branch", "--show-current") != BRANCH or git("status", "--porcelain"):
        raise CompletionError("completion entry requires the exact clean scientific branch")
    command = [
        sys.executable,
        str(LOCAL_REPO / "scripts" / "experiment_2d5c_runpod_guard.py"),
        "watchdog",
        "--authorization-artifact", str(args.authorization_artifact),
        "--trigger-file", str(args.trigger_file),
        "--report-artifact", str(args.stop_report),
        "--watch-timeout-seconds", str(args.watch_timeout_seconds),
        "--stop-timeout-seconds", str(args.stop_timeout_seconds),
        "--poll-interval-seconds", "5",
        "--",
        sys.executable,
        str(LOCAL_REPO / "scripts" / "experiment_2d5c_workflow.py"),
        "--authorization-artifact", str(args.authorization_artifact),
        "--runtime-log", str(args.runtime_log),
    ]
    result = subprocess.run(command, cwd=str(LOCAL_REPO), check=False)
    if not args.stop_report.is_file():
        raise CompletionError("guard returned without a dedicated stop report")
    stop = json.loads(args.stop_report.read_text(encoding="utf-8"))
    if result.returncode != 0:
        raise CompletionError(
            "scientific child failed; guard stop report was preserved at "
            f"{args.stop_report} with child exit {result.returncode}"
        )
    if not (
        stop.get("passed") is True
        and stop.get("terminal_outcome") == "success"
        and stop.get("pod", {}).get("desiredStatus") == "EXITED"
        and stop.get("network_volume", {}).get("id") == "yhzyb27fb5"
    ):
        raise CompletionError("guard did not prove successful exact-pod stop")


def finalize_and_commit(args: argparse.Namespace) -> dict:
    summary = LOCAL_RESULTS / "SCIENTIFIC_RESULT_SUMMARY.json"
    representation = LOCAL_RESULTS / "REPRESENTATION_PRESSURE_DIAGNOSTICS.json"
    provisional = LOCAL_RESULTS / "SCIENTIFIC_AUDIT_PRETAG.json"
    git_verification = LOCAL_RESULTS / "GIT_VERIFICATION.json"
    final_audit = LOCAL_RESULTS / "FINAL_AUDIT.json"
    final_report = LOCAL_RESULTS / "EXPERIMENT_2D5C_FINAL_REPORT.md"
    for path in (summary, representation, provisional, git_verification):
        if not path.is_file():
            raise CompletionError(f"missing post-stop finalizer input: {path}")
    for path in (final_audit, final_report):
        if path.exists():
            raise CompletionError(f"refusing to overwrite terminal artifact: {path}")

    finalizer = LOCAL_REPO / "scripts" / "experiment_2d5c_finalizer.py"
    run([
        sys.executable, str(finalizer), "postflight-audit",
        "--provisional-audit", str(provisional),
        "--summary", str(summary),
        "--representation", str(representation),
        "--git-verification", str(git_verification),
        "--stop-verification", str(args.stop_report),
        "--guard-authorization", str(args.authorization_artifact),
        "--guard-trigger", str(args.trigger_file),
        "--output-path", str(final_audit),
    ])
    run([
        sys.executable, str(finalizer), "render-report",
        "--summary", str(summary),
        "--representation", str(representation),
        "--postflight-audit", str(final_audit),
        "--output-path", str(final_report),
    ])
    if args.runtime_log.is_file():
        shutil.copy2(args.runtime_log, LOCAL_RESULTS / "WORKFLOW_RUNTIME.jsonl")

    guard_identity = {
        "experiment": EXPERIMENT,
        "authorization": {
            "sha256": sha256(args.authorization_artifact),
            "mode": oct(args.authorization_artifact.stat().st_mode & 0o777),
            "committed": False,
        },
        "trigger": {
            "sha256": sha256(args.trigger_file),
            "mode": oct(args.trigger_file.stat().st_mode & 0o777),
            "committed": False,
        },
        "stop_report": {
            "sha256": sha256(args.stop_report),
            "committed_path": str(args.stop_report),
        },
        "private_guard_artifacts_retained_outside_git": True,
    }
    write_json_atomic(LOCAL_RESULTS / "GUARD_ARTIFACT_IDENTITIES.json", guard_identity)

    git("add", "--", str(LOCAL_RESULTS.relative_to(LOCAL_REPO)))
    if not git("status", "--porcelain"):
        raise CompletionError("no terminal artifacts were staged")
    run(
        ["git", "commit", "-m", "Record terminal Experiment 2D5C postflight"],
        cwd=LOCAL_REPO,
    )
    terminal_commit = git("rev-parse", "HEAD")
    run(["git", "push", "origin", BRANCH], cwd=LOCAL_REPO)
    scientific = json.loads(git_verification.read_text(encoding="utf-8"))[
        "scientific_results_commit"
    ]
    origin_branch = git("ls-remote", "origin", f"refs/heads/{BRANCH}")
    origin_tag = git("ls-remote", "origin", f"refs/tags/{FINAL_TAG}^{{}}")
    branch_sha = origin_branch.split()[0] if origin_branch.split() else None
    tag_sha = origin_tag.split()[0] if origin_tag.split() else None
    clean = not bool(git("status", "--porcelain"))
    verification = {
        "experiment": EXPERIMENT,
        "branch": BRANCH,
        "terminal_postflight_commit": terminal_commit,
        "origin_branch_commit": branch_sha,
        "scientific_results_commit": scientific,
        "final_tag": FINAL_TAG,
        "origin_tag_commit": tag_sha,
        "worktree_clean_after_terminal_commit": clean,
        "final_audit_sha256": sha256(final_audit),
        "final_report_sha256": sha256(final_report),
        "passed": branch_sha == terminal_commit and tag_sha == scientific and clean,
        "verified_at_unix": time.time(),
    }
    if not verification["passed"]:
        raise CompletionError(f"terminal Git verification failed: {verification}")
    write_json_atomic(args.terminal_git_verification, verification)
    return verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded terminal completion for official Experiment 2D5C"
    )
    parser.add_argument("--authorization-artifact", type=Path, required=True)
    parser.add_argument("--trigger-file", type=Path, required=True)
    parser.add_argument("--stop-report", type=Path, required=True)
    parser.add_argument(
        "--runtime-log", type=Path,
        default=LOCAL_ARCHIVE / "WORKFLOW_RUNTIME.jsonl",
    )
    parser.add_argument(
        "--terminal-git-verification", type=Path,
        default=LOCAL_ARCHIVE / "TERMINAL_GIT_VERIFICATION.json",
    )
    parser.add_argument("--watch-timeout-seconds", type=float, default=7 * 24 * 60 * 60)
    parser.add_argument("--stop-timeout-seconds", type=float, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "authorization_artifact", "trigger_file", "stop_report", "runtime_log",
        "terminal_git_verification",
    ):
        setattr(args, name, getattr(args, name).resolve())
    try:
        require_fresh_path(args.terminal_git_verification, "terminal Git verification")
        run_guarded_workflow(args)
        verification = finalize_and_commit(args)
    except Exception as error:
        print(f"EXPERIMENT_2D5C_COMPLETION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(
        "EXPERIMENT_2D5C_TERMINAL_COMPLETE "
        f"{verification['terminal_postflight_commit']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
