#!/usr/bin/env python3
"""Guarded terminal completion for the adjudicated 2D5C post-training flow.

The accepted C training was executed at ``TRAINING_FREEZE_COMMIT``.  This
stdlib-only host entrypoint does not train or adjudicate anything itself.  It
places the dedicated post-training workflow under the existing exact-pod
RunPod watchdog, using fresh artifacts for this recovery attempt.  The guard
creates a stop-capable trigger and stops the pod only after the child exits
successfully.  A child failure retains the pod and cannot enter terminal
finalization.

After a successful exact stopped-pod report, the established 2D5C
``finalize_and_commit`` implementation performs the postflight audit, report,
terminal commit, push, and Git verification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from experiment_2d5c_complete import (
    BRANCH,
    LOCAL_REPO,
    CompletionError,
    finalize_and_commit,
    git,
    require_fresh_path,
    sha256,
)


EXPERIMENT = "2D5C"
TRAINING_FREEZE_COMMIT = "4df3cfaaa486a7162485862ea521e69c47d5147d"
GUARD_REPORT_SCHEMA = "experiment_2d5c_runpod_guard_report_v1"
POD_ID = "rvgztsr0azrwyo"
POD_NAME = "happy_apricot_stork"
NETWORK_VOLUME_ID = "yhzyb27fb5"
NETWORK_VOLUME_NAME = "unlikely_lime_flamingo"


def exact_training_freeze_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError(
            "training freeze commit must be one lowercase 40-hex commit"
        )
    if value != TRAINING_FREEZE_COMMIT:
        raise argparse.ArgumentTypeError(
            "training freeze commit does not equal the sealed 2D5C training commit"
        )
    return value


def ssh_host(value: str) -> str:
    if (
        not value
        or value.startswith("-")
        or "@" in value
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise argparse.ArgumentTypeError(
            "SSH host must be a bare host/address without a user prefix"
        )
    return value


def ssh_port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SSH port must be an integer") from error
    if parsed < 1 or parsed > 65_535:
        raise argparse.ArgumentTypeError("SSH port must be in 1..65535")
    return parsed


def positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be numeric") from error
    if not math.isfinite(parsed) or not parsed > 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return parsed


def _inside_local_repo(path: Path) -> bool:
    try:
        path.relative_to(LOCAL_REPO.resolve())
    except ValueError:
        return False
    return True


def validate_artifact_paths(args: argparse.Namespace) -> None:
    paths = {
        "authorization": args.authorization_artifact,
        "guard trigger": args.trigger_file,
        "guard stop report": args.stop_report,
        "post-training runtime log": args.runtime_log,
        "terminal Git verification": args.terminal_git_verification,
    }
    if len(set(paths.values())) != len(paths):
        raise CompletionError("completion artifact paths must be pairwise distinct")
    for label, path in paths.items():
        if not path.is_absolute():
            raise CompletionError(f"{label} path must be absolute")
        if _inside_local_repo(path):
            raise CompletionError(f"{label} must remain outside the Git worktree")

    authorization = args.authorization_artifact
    if authorization.is_symlink() or not authorization.is_file():
        raise CompletionError("missing regular private guard authorization")
    if stat.S_IMODE(os.lstat(authorization).st_mode) != 0o600:
        raise CompletionError("private guard authorization must have mode 0600")

    require_fresh_path(args.trigger_file, "post-training guard trigger")
    require_fresh_path(args.stop_report, "post-training guard stop report")
    require_fresh_path(args.runtime_log, "post-training runtime log")
    require_fresh_path(
        args.terminal_git_verification,
        "post-training terminal Git verification",
    )


def validate_git_lineage(training_freeze_commit: str) -> None:
    if training_freeze_commit != TRAINING_FREEZE_COMMIT:
        raise CompletionError("unexpected 2D5C training freeze commit")
    if git("branch", "--show-current") != BRANCH:
        raise CompletionError("post-training completion requires the exact branch")
    if git("status", "--porcelain"):
        raise CompletionError("post-training completion requires a clean worktree")
    head = git("rev-parse", "HEAD")
    if git("rev-parse", f"{training_freeze_commit}^{{commit}}") != training_freeze_commit:
        raise CompletionError("training freeze commit does not resolve exactly")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", training_freeze_commit, head],
        cwd=str(LOCAL_REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestry.returncode != 0:
        raise CompletionError("current post-training commit does not descend from training freeze")
    if git("rev-parse", f"origin/{BRANCH}") != head:
        raise CompletionError("post-training completion commit is not pushed")


def _read_guard_report(path: Path) -> dict:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise CompletionError("guard stop report is unavailable") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise CompletionError("guard stop report is not a private regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompletionError("guard stop report is not readable JSON") from error
    if not isinstance(payload, dict):
        raise CompletionError("guard stop report is not a JSON object")
    return payload


def run_guarded_posttrain_workflow(args: argparse.Namespace) -> dict:
    """Run the whole recovery workflow under success-only stop supervision."""

    validate_artifact_paths(args)
    validate_git_lineage(args.training_freeze_commit)
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
        str(LOCAL_REPO / "scripts" / "experiment_2d5c_posttrain_workflow.py"),
        "--authorization-artifact", str(args.authorization_artifact),
        "--runtime-log", str(args.runtime_log),
        "--ssh-host", args.ssh_host,
        "--ssh-port", str(args.ssh_port),
        "--training-freeze-commit", args.training_freeze_commit,
    ]
    result = subprocess.run(command, cwd=str(LOCAL_REPO), check=False)
    if not args.stop_report.is_file():
        raise CompletionError(
            "post-training guard returned without a dedicated stop report"
        )
    stop = _read_guard_report(args.stop_report)
    if result.returncode != 0:
        if (
            stop.get("terminal_outcome") == "failure"
            and (
                args.trigger_file.exists()
                or stop.get("trigger_artifact_created") is not False
                or stop.get("pod_stop_attempted") is not False
            )
        ):
            raise CompletionError(
                "failed post-training child produced unsafe stop/trigger evidence"
            )
        raise CompletionError(
            "post-training child failed; the guard report and runtime were retained "
            f"at {args.stop_report} and {args.runtime_log}"
        )

    checks = {
        "schema": stop.get("schema") == GUARD_REPORT_SCHEMA,
        "mode": stop.get("mode") == "watchdog_supervise",
        "passed": stop.get("passed") is True,
        "child_success": stop.get("terminal_outcome") == "success"
        and stop.get("child_exit_code") == 0,
        "trigger_exists": args.trigger_file.is_file(),
        "trigger_private": args.trigger_file.is_file()
        and not args.trigger_file.is_symlink()
        and stat.S_IMODE(os.lstat(args.trigger_file).st_mode) == 0o600,
        "authorization_bound": stop.get("authorization_sha256")
        == sha256(args.authorization_artifact),
        "trigger_bound": args.trigger_file.is_file()
        and stop.get("trigger_sha256") == sha256(args.trigger_file),
        "exact_pod": stop.get("pod", {}).get("id") == POD_ID
        and stop.get("pod", {}).get("name") == POD_NAME,
        "pod_stopped": stop.get("pod", {}).get("desiredStatus") == "EXITED"
        and stop.get("pod", {}).get("runtimeStatus") == "stopped",
        "volume_retained": stop.get("network_volume", {}).get("id")
        == NETWORK_VOLUME_ID
        and stop.get("network_volume", {}).get("name") == NETWORK_VOLUME_NAME,
        "safe_status": stop.get("status") in {
            "stopped_and_volume_retained_verified",
            "already_stopped_verified",
        },
        "no_secret": stop.get("secret_recorded") is False,
        "runtime_written": args.runtime_log.is_file()
        and args.runtime_log.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise CompletionError(
            f"guard did not prove successful exact-pod post-training stop: {checks}"
        )
    return stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded terminal completion for adjudicated Experiment 2D5C"
    )
    parser.add_argument("--authorization-artifact", type=Path, required=True)
    parser.add_argument("--trigger-file", type=Path, required=True)
    parser.add_argument("--stop-report", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--terminal-git-verification", type=Path, required=True)
    parser.add_argument("--ssh-host", type=ssh_host, required=True)
    parser.add_argument("--ssh-port", type=ssh_port, required=True)
    parser.add_argument(
        "--training-freeze-commit",
        type=exact_training_freeze_commit,
        required=True,
    )
    parser.add_argument(
        "--watch-timeout-seconds",
        type=positive_seconds,
        default=7 * 24 * 60 * 60,
    )
    parser.add_argument(
        "--stop-timeout-seconds", type=positive_seconds, default=900
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "authorization_artifact",
        "trigger_file",
        "stop_report",
        "runtime_log",
        "terminal_git_verification",
    ):
        setattr(args, name, getattr(args, name).resolve())
    try:
        run_guarded_posttrain_workflow(args)
        verification = finalize_and_commit(args)
    except Exception as error:
        print(
            f"EXPERIMENT_2D5C_POSTTRAIN_COMPLETION_FAILURE: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        "EXPERIMENT_2D5C_POSTTRAIN_TERMINAL_COMPLETE "
        f"{verification['terminal_postflight_commit']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
