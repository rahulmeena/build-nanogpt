#!/usr/bin/env python3
"""Supervised host-side workflow for the official Experiment 2D5C run.

This module is intentionally stdlib-only.  It is designed to be the child of
``experiment_2d5c_runpod_guard.py watchdog`` so every terminal child outcome
causes the guard to preserve the retained volume and stop exactly the frozen
pod identity.  It never calls RunPod itself and cannot start another arm.

Preparation/freeze is deliberately outside this workflow: the caller must
first commit and push the implementation, run ``prepare`` on the A100, commit
and push the frozen pretraining manifests, and update the remote checkout to
that exact clean commit.  This workflow then performs preflight, the two
authorized C-only training processes, all required evaluations/diagnostics,
local checkpoint backup, analysis, and the scientific-results commit/tag.
The stopped-state audit and terminal report are produced locally only after
the guard has returned its verified stop report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT = "2D5C"
BRANCH = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m"
FINAL_TAG = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final"
GIT_VERIFICATION_SCHEMA = "experiment_2d5c_git_verification_v1"

IDENTITY_FILE = Path("/Users/rahul/.ssh/id_ed25519")
REMOTE_REPO = Path("/tmp/exp2d5c-official")
RUN_ROOT = Path("/workspace/exp2d5c_w2w2_100m")
PRETRAIN = RUN_ROOT / "pretrain"
PREFLIGHT = RUN_ROOT / "preflight"
RESULTS = RUN_ROOT / "results"
CHECKPOINTS = RUN_ROOT / "checkpoints"

SOURCE = Path(
    "/workspace/exp2d3a_run/checkpoints/"
    "scientific_cumulative_001000341504.pt"
)
CONTROL = Path(
    "/workspace/exp2d4a_fixed_run/checkpoints/scientific_local_0191.pt"
)
DATA_ROOT = Path("/workspace/edu_fineweb10B")

SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
CONTROL_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
MILESTONE_NAMES = {
    48: "scientific_cumulative_001025507328.pt",
    96: "scientific_cumulative_001050673152.pt",
    144: "scientific_cumulative_001075838976.pt",
    191: "scientific_cumulative_001100480512.pt",
}
VOLUME_CAPACITY_BYTES = 150_000_000_000
MIN_PREFLIGHT_FREE_BYTES = 12 * 1024**3

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


class WorkflowError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_runtime(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Workflow:
    def __init__(self, runtime_log: Path, ssh_host: str, ssh_port: int):
        if not ssh_host or any(character.isspace() for character in ssh_host):
            raise WorkflowError("SSH host must be a nonempty address without whitespace")
        if not 1 <= int(ssh_port) <= 65_535:
            raise WorkflowError("SSH port must be between 1 and 65535")
        self.runtime_log = runtime_log.resolve()
        self.ssh_host = ssh_host
        self.ssh_port = int(ssh_port)
        self.remote = f"root@{ssh_host}"

    def ssh_prefix(self) -> list[str]:
        return [
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6", "-i", str(IDENTITY_FILE),
            "-p", str(self.ssh_port), self.remote,
        ]

    def rsync_shell(self) -> str:
        return shlex.join([
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6", "-i", str(IDENTITY_FILE),
            "-p", str(self.ssh_port),
        ])

    def run(
        self,
        argv: Sequence[str],
        stage: str,
        *,
        cwd: Path | None = None,
        capture: bool = False,
    ) -> str:
        command = [str(value) for value in argv]
        started = time.time()
        append_runtime(self.runtime_log, {
            "experiment": EXPERIMENT,
            "stage": stage,
            "event": "start",
            "started_at_unix": started,
            "argv": command,
            "cwd": None if cwd is None else str(cwd.resolve()),
        })
        print(f"[2D5C workflow] {stage}", flush=True)
        result = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
        finished = time.time()
        append_runtime(self.runtime_log, {
            "experiment": EXPERIMENT,
            "stage": stage,
            "event": "finish",
            "finished_at_unix": finished,
            "wall_seconds": finished - started,
            "returncode": result.returncode,
        })
        if result.returncode != 0:
            if capture and result.stdout:
                print(result.stdout, end="", flush=True)
            raise WorkflowError(f"stage {stage!r} failed with exit code {result.returncode}")
        return "" if result.stdout is None else result.stdout.strip()

    def remote_capture(self, script: str, stage: str) -> str:
        return self.run([*self.ssh_prefix(), script], stage, capture=True)

    def driver(self, stage: str, *arguments: str) -> None:
        remote_command = shlex.join([
            "cd", str(REMOTE_REPO), "&&", "exec", "python3", "-u",
            "scripts/experiment_2d5c.py", *arguments,
        ])
        # shlex.join quotes the shell metacharacter, so construct the fixed cd
        # prefix separately while retaining strict quoting for every path/arg.
        remote_command = (
            f"cd {shlex.quote(str(REMOTE_REPO))} && "
            + shlex.join([
                "exec", "python3", "-u", "scripts/experiment_2d5c.py",
                *arguments,
            ])
        )
        self.run([*self.ssh_prefix(), remote_command], stage)

    def rsync_from(self, remote_path: Path | str, local_path: Path | str, stage: str) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.run([
            "rsync", "-a", "--partial", "--human-readable", "--progress",
            "-e", self.rsync_shell(), f"{self.remote}:{remote_path}", str(local_path),
        ], stage)

    def rsync_to(self, local_path: Path, remote_path: Path, stage: str) -> None:
        self.run([
            "rsync", "-a", "--partial", "-e", self.rsync_shell(),
            str(local_path), f"{self.remote}:{remote_path}",
        ], stage)


def checkpoint(update: int) -> Path:
    return CHECKPOINTS / MILESTONE_NAMES[int(update)]


def evaluate_args(
    family: str,
    label: str,
    local_update: int,
    *,
    checkpoint_path: Path | None = None,
    panel: str = "core",
    all_real_only: bool = False,
    parallel: bool = False,
    final_seal: bool = False,
) -> list[str]:
    panel_path = PRETRAIN / (
        "PANEL_MANIFEST_CORE.json" if panel == "core"
        else "PANEL_MANIFEST_LARGE.json"
    )
    arguments = [
        "evaluate", "--family", family,
        "--source-checkpoint", str(SOURCE),
        "--expected-local-update", str(local_update),
        "--data-root", str(DATA_ROOT),
        "--panel-manifest", str(panel_path),
        "--shuffle-manifest", str(PRETRAIN / "SHUFFLE_MANIFEST.json"),
        "--pretrain-freeze-audit", str(PRETRAIN / "PRETRAIN_FREEZE_AUDIT.json"),
        "--panel-kind", panel,
        "--output-path", str(RESULTS / f"{label}.json"),
    ]
    if checkpoint_path is not None:
        arguments.extend(["--checkpoint", str(checkpoint_path)])
    if family == "C":
        arguments.extend([
            "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json")
        ])
    if final_seal:
        arguments.extend([
            "--final-checkpoint-seal",
            str(RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"),
        ])
    if all_real_only:
        arguments.append("--all-real-only")
    if parallel:
        arguments.extend([
            "--parallel-output", str(RESULTS / f"{label}_PARALLEL.json"),
            "--parallel-passes", "2",
        ])
    return arguments


def representation_args(
    family: str,
    label: str,
    local_update: int,
    checkpoint_path: Path | None = None,
    final_seal: bool = False,
) -> list[str]:
    arguments = [
        "representation-diagnostics", "--family", family,
        "--source-checkpoint", str(SOURCE),
        "--expected-local-update", str(local_update),
        "--label", label,
        "--data-root", str(DATA_ROOT),
        "--core-manifest", str(PRETRAIN / "PANEL_MANIFEST_CORE.json"),
        "--pretrain-freeze-audit", str(PRETRAIN / "PRETRAIN_FREEZE_AUDIT.json"),
        "--output-dir", str(RESULTS / "representation_rows" / label.lower()),
        "--output-json", str(RESULTS / f"REPRESENTATION_{label}.json"),
    ]
    if checkpoint_path is not None:
        arguments.extend(["--checkpoint", str(checkpoint_path)])
    if family == "C":
        arguments.extend([
            "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json")
        ])
    if final_seal:
        arguments.extend([
            "--final-checkpoint-seal",
            str(RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"),
        ])
    return arguments


def assert_private_authorization(path: Path) -> None:
    if not path.is_absolute() or not path.is_file():
        raise WorkflowError("the guard authorization artifact must be an existing absolute file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise WorkflowError("the guard authorization artifact is not private")


def git_output(workflow: Workflow, *args: str) -> str:
    return workflow.run(
        ["git", *args], f"git {' '.join(args[:2])}", cwd=LOCAL_REPO,
        capture=True,
    )


def verify_entry(workflow: Workflow, authorization: Path) -> tuple[str, int]:
    assert_private_authorization(authorization)
    if not IDENTITY_FILE.is_file():
        raise WorkflowError(f"missing SSH identity {IDENTITY_FILE}")
    if git_output(workflow, "branch", "--show-current") != BRANCH:
        raise WorkflowError("local scientific branch mismatch")
    if git_output(workflow, "status", "--porcelain"):
        raise WorkflowError("local repository must be clean at workflow entry")
    local_head = git_output(workflow, "rev-parse", "HEAD")
    if git_output(workflow, "rev-parse", f"origin/{BRANCH}") != local_head:
        raise WorkflowError("local freeze commit is not pushed")
    live_branch = git_output(
        workflow, "ls-remote", "origin", f"refs/heads/{BRANCH}"
    ).split()
    if not live_branch or live_branch[0] != local_head:
        raise WorkflowError("live origin branch does not equal the local freeze commit")
    local_tag = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{FINAL_TAG}"],
        cwd=LOCAL_REPO, check=False, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    remote_tag = git_output(
        workflow, "ls-remote", "origin", f"refs/tags/{FINAL_TAG}*"
    )
    if local_tag or remote_tag:
        raise WorkflowError(f"final tag already exists: {FINAL_TAG}")
    remote_head = workflow.remote_capture(
        f"cd {shlex.quote(str(REMOTE_REPO))} && "
        "test -z \"$(git status --porcelain)\" && "
        "git rev-parse HEAD && git rev-parse origin/"
        + shlex.quote(BRANCH),
        "verify clean remote freeze checkout",
    ).splitlines()
    if remote_head != [local_head, local_head]:
        raise WorkflowError(f"remote freeze commit mismatch: {remote_head}")
    required = [
        SOURCE, CONTROL, PRETRAIN / "PRETRAIN_FREEZE_AUDIT.json",
        PRETRAIN / "DATA_REPLAY_LEDGER.jsonl",
        PRETRAIN / "DATA_REPLAY_AUDIT.json",
        PRETRAIN / "PANEL_MANIFEST_CORE.json",
        PRETRAIN / "PANEL_MANIFEST_LARGE.json",
        PRETRAIN / "SHUFFLE_MANIFEST.json",
    ]
    script = "set -eu; " + "; ".join(
        f"test -f {shlex.quote(str(path))}" for path in required
    )
    workflow.remote_capture(script, "verify frozen remote inputs")
    inventory_script = (
        "import hashlib,json,sys; from pathlib import Path; r=Path(sys.argv[1]); "
        "h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); "
        "print(json.dumps({str(p.relative_to(r)):h(p) for p in sorted(r.rglob('*')) "
        "if p.is_file()},sort_keys=True,separators=(',',':')))"
    )
    remote_inventory = json.loads(workflow.remote_capture(
        shlex.join(["python3", "-c", inventory_script, str(PRETRAIN)]),
        "bind local frozen-result inventory to remote pretrain inventory",
    ))
    if not LOCAL_RESULTS.is_dir():
        raise WorkflowError("local frozen result directory is missing")
    local_inventory = {
        str(path.relative_to(LOCAL_RESULTS)): sha256(path)
        for path in sorted(LOCAL_RESULTS.rglob("*")) if path.is_file()
    }
    if local_inventory != remote_inventory:
        raise WorkflowError("local frozen result inventory differs from remote pretrain")
    stale_archive = [
        str(path) for name in MILESTONE_NAMES.values()
        for path in (
            LOCAL_ARCHIVE / name,
            LOCAL_ARCHIVE / f"{name}.sha256",
            LOCAL_ARCHIVE / f"{name}.verification.json",
        ) if path.exists()
    ]
    if stale_archive:
        raise WorkflowError(f"local C checkpoint archive is not fresh: {stale_archive}")
    official_outputs = (PREFLIGHT, RESULTS, CHECKPOINTS)
    output_script = "set -eu; " + "; ".join(
        "test ! -e {path} || test -z \"$(find {path} -mindepth 1 -maxdepth 1 "
        "-print -quit)\"".format(path=shlex.quote(str(path)))
        for path in official_outputs
    )
    workflow.remote_capture(
        output_script, "verify official preflight/training outputs are fresh"
    )
    remote_hashes = workflow.remote_capture(
        "sha256sum " + shlex.join([str(SOURCE), str(CONTROL)]),
        "rehash source and Fixed control",
    ).splitlines()
    observed = [line.split()[0] for line in remote_hashes]
    if observed != [SOURCE_SHA256, CONTROL_SHA256]:
        raise WorkflowError(f"remote lineage hash mismatch: {observed}")
    storage = workflow.remote_capture(
        "set -eu; du -sb /workspace | awk '{print $1}'; "
        "du -sB1 /workspace | awk '{print $1}'; "
        "df -B1 --output=avail /workspace | tail -1 | tr -d ' '",
        "measure retained-volume apparent, allocated, and filesystem headroom",
    ).splitlines()
    if len(storage) != 3 or not all(row.isdigit() for row in storage):
        raise WorkflowError(f"invalid retained-volume storage evidence: {storage}")
    apparent_used, allocated_used, filesystem_available = map(int, storage)
    quota_available = VOLUME_CAPACITY_BYTES - max(apparent_used, allocated_used)
    free = min(quota_available, filesystem_available)
    if free < MIN_PREFLIGHT_FREE_BYTES:
        raise WorkflowError(
            f"retained-volume headroom {free} is below {MIN_PREFLIGHT_FREE_BYTES}"
        )
    return local_head, free


def backup_checkpoints(workflow: Workflow) -> dict:
    LOCAL_ARCHIVE.mkdir(parents=True, exist_ok=True)
    rows = {}
    for update, name in MILESTONE_NAMES.items():
        remote_path = CHECKPOINTS / name
        local_path = LOCAL_ARCHIVE / name
        workflow.rsync_from(remote_path, local_path, f"backup C update {update} checkpoint")
        for suffix in (".sha256", ".verification.json"):
            workflow.rsync_from(
                Path(str(remote_path) + suffix),
                Path(str(local_path) + suffix),
                f"backup C update {update}{suffix}",
            )
        remote_digest = workflow.remote_capture(
            f"sha256sum {shlex.quote(str(remote_path))} | awk '{{print $1}}'",
            f"independently hash remote C update {update}",
        )
        local_digest = sha256(local_path)
        sidecar = json.loads(Path(str(local_path) + ".verification.json").read_text())
        row = {
            "local_update": update,
            "remote_path": str(remote_path),
            "local_path": str(local_path),
            "remote_sha256": remote_digest,
            "local_sha256": local_digest,
            "verification_sidecar_sha256": sidecar.get("sha256"),
            "bytes": local_path.stat().st_size,
        }
        row["passed"] = (
            remote_digest == local_digest == sidecar.get("sha256")
            and sidecar.get("local_update") == update
            and sidecar.get("strict_reopen", {}).get("passed") is True
        )
        if not row["passed"]:
            raise WorkflowError(f"checkpoint backup verification failed: {row}")
        rows[str(update)] = row
    final = rows["191"]
    audit = {
        "experiment": EXPERIMENT,
        "all_milestones_backed_up": True,
        "milestones": rows,
        "remote_path": final["remote_path"],
        "local_path": final["local_path"],
        "remote_sha256": final["remote_sha256"],
        "local_sha256": final["local_sha256"],
        "passed": all(row["passed"] for row in rows.values()),
        "completed_at_unix": time.time(),
    }
    local_audit = LOCAL_ARCHIVE / "LOCAL_BACKUP_AUDIT.json"
    write_json_atomic(local_audit, audit)
    workflow.rsync_to(
        local_audit, RESULTS / "LOCAL_BACKUP_AUDIT.json",
        "copy local backup audit to scientific result set",
    )
    return audit


def run_scientific_workflow(workflow: Workflow, authorization: Path) -> dict:
    freeze_head, free_bytes = verify_entry(workflow, authorization)
    workflow.driver(
        "A100 preflight and disposable validation",
        "preflight",
        "--output-dir", str(PREFLIGHT),
        "--pretrain-dir", str(PRETRAIN),
        "--source-checkpoint", str(SOURCE),
        "--data-root", str(DATA_ROOT),
        "--stop-capability-verified",
        "--storage-inventory-verified",
        "--network-volume-free-bytes", str(free_bytes),
    )
    workflow.driver(
        "official C training process 1: updates 0 through 96",
        "train", "--arm", "C", "--end-local-update", "96",
        "--output-dir", str(RESULTS),
        "--preflight-audit", str(PREFLIGHT / "PREFLIGHT_AUDIT.json"),
        "--replay-ledger", str(PRETRAIN / "DATA_REPLAY_LEDGER.jsonl"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--source-checkpoint", str(SOURCE),
        "--large-panel", str(PRETRAIN / "PANEL_MANIFEST_LARGE.json"),
        "--scientific-checkpoint-dir", str(CHECKPOINTS),
    )
    workflow.driver(
        "official C training process 2: fresh restart, updates 97 through 191",
        "train", "--arm", "C", "--end-local-update", "191",
        "--output-dir", str(RESULTS),
        "--preflight-audit", str(PREFLIGHT / "PREFLIGHT_AUDIT.json"),
        "--replay-ledger", str(PRETRAIN / "DATA_REPLAY_LEDGER.jsonl"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--source-checkpoint", str(SOURCE),
        "--large-panel", str(PRETRAIN / "PANEL_MANIFEST_LARGE.json"),
        "--scientific-checkpoint-dir", str(CHECKPOINTS),
        "--resume-checkpoint", str(checkpoint(96)),
        "--midpoint-preexit", str(RESULTS / "MIDPOINT_RESTART_PREEXIT.json"),
    )
    workflow.driver(
        "fresh-process final C checkpoint seal",
        "seal-final",
        "--checkpoint", str(checkpoint(191)),
        "--source-checkpoint", str(SOURCE),
        "--training-complete", str(RESULTS / "TRAINING_COMPLETE.json"),
        "--training-log", str(RESULTS / "TRAINING_LOG.jsonl"),
        "--training-replay-actual", str(RESULTS / "TRAINING_REPLAY_ACTUAL.jsonl"),
        "--replay-ledger", str(PRETRAIN / "DATA_REPLAY_LEDGER.jsonl"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--midpoint-restart-preexit", str(RESULTS / "MIDPOINT_RESTART_PREEXIT.json"),
        "--midpoint-restart-audit", str(RESULTS / "MIDPOINT_RESTART_AUDIT.json"),
        "--output-path", str(RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
    )
    backup = backup_checkpoints(workflow)
    workflow.driver(
        "exact BF16 persistent-state audit",
        "memory-audit",
        "--fixed-checkpoint", str(CONTROL),
        "--c-checkpoint", str(checkpoint(191)),
        "--source-checkpoint", str(SOURCE),
        "--output-json", str(RESULTS / "BF16_PERSISTENT_STATE_RAW.json"),
        "--output-table", str(RESULTS / "BF16_PERSISTENT_STATE_RAW.md"),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
        "--final-checkpoint-seal", str(RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"),
    )

    evaluations = [
        ("Parent core", evaluate_args("Parent", "PARENT_CORE", 0, all_real_only=True)),
        ("C0 core and secondary parallel", evaluate_args("C0", "C0_CORE", 0, parallel=True)),
        ("C48 core", evaluate_args("C", "C48_CORE", 48, checkpoint_path=checkpoint(48))),
        ("C96 core and secondary parallel", evaluate_args("C", "C96_CORE", 96, checkpoint_path=checkpoint(96), parallel=True)),
        ("C144 core", evaluate_args("C", "C144_CORE", 144, checkpoint_path=checkpoint(144))),
        ("C191 core and secondary parallel", evaluate_args("C", "C191_CORE", 191, checkpoint_path=checkpoint(191), parallel=True, final_seal=True)),
        ("Fixed100M core", evaluate_args("Fixed", "FIXED_CORE", 191, checkpoint_path=CONTROL)),
        ("C191 final large", evaluate_args("C", "C191_LARGE", 191, checkpoint_path=checkpoint(191), panel="large", final_seal=True)),
        ("Fixed100M final large", evaluate_args("Fixed", "FIXED_LARGE", 191, checkpoint_path=CONTROL, panel="large")),
    ]
    for stage, arguments in evaluations:
        workflow.driver(f"true-incremental evaluation: {stage}", *arguments)

    diagnostics = [
        ("Parent", representation_args("Parent", "PARENT", 0)),
        ("C0", representation_args("C0", "C0", 0)),
        ("C48", representation_args("C", "C48", 48, checkpoint(48))),
        ("C96", representation_args("C", "C96", 96, checkpoint(96))),
        ("C144", representation_args("C", "C144", 144, checkpoint(144))),
        ("C191", representation_args("C", "C191", 191, checkpoint(191), final_seal=True)),
        ("Fixed100M", representation_args("Fixed", "FIXED100M", 191, CONTROL)),
    ]
    for stage, arguments in diagnostics:
        workflow.driver(f"representation-pressure diagnostics: {stage}", *arguments)

    analyze = [
        "analyze", "--output-dir", str(RESULTS),
        "--parent-core", str(RESULTS / "PARENT_CORE.json"),
        "--c0-core", str(RESULTS / "C0_CORE.json"),
        "--c48-core", str(RESULTS / "C48_CORE.json"),
        "--c96-core", str(RESULTS / "C96_CORE.json"),
        "--c144-core", str(RESULTS / "C144_CORE.json"),
        "--c191-core", str(RESULTS / "C191_CORE.json"),
        "--fixed-core", str(RESULTS / "FIXED_CORE.json"),
        "--c0-parallel", str(RESULTS / "C0_CORE_PARALLEL.json"),
        "--c96-parallel", str(RESULTS / "C96_CORE_PARALLEL.json"),
        "--c191-parallel", str(RESULTS / "C191_CORE_PARALLEL.json"),
        "--c-large", str(RESULTS / "C191_LARGE.json"),
        "--fixed-large", str(RESULTS / "FIXED_LARGE.json"),
        "--representation-parent", str(RESULTS / "REPRESENTATION_PARENT.json"),
        "--representation-c0", str(RESULTS / "REPRESENTATION_C0.json"),
        "--representation-c48", str(RESULTS / "REPRESENTATION_C48.json"),
        "--representation-c96", str(RESULTS / "REPRESENTATION_C96.json"),
        "--representation-c144", str(RESULTS / "REPRESENTATION_C144.json"),
        "--representation-c191", str(RESULTS / "REPRESENTATION_C191.json"),
        "--representation-fixed", str(RESULTS / "REPRESENTATION_FIXED100M.json"),
        "--memory-audit", str(RESULTS / "BF16_PERSISTENT_STATE_RAW.json"),
        "--training-log", str(RESULTS / "TRAINING_LOG.jsonl"),
        "--scope-lock", str(PRETRAIN / "SCOPE_LOCK.json"),
        "--source-provenance", str(PRETRAIN / "SOURCE_PROVENANCE.json"),
        "--control-provenance", str(PRETRAIN / "FIXED_CONTROL_PROVENANCE.json"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--pretrain-freeze-audit", str(PRETRAIN / "PRETRAIN_FREEZE_AUDIT.json"),
        "--preflight-audit", str(PREFLIGHT / "PREFLIGHT_AUDIT.json"),
        "--preflight-tests", str(PREFLIGHT / "PREFLIGHT_TESTS.json"),
        "--training-complete", str(RESULTS / "TRAINING_COMPLETE.json"),
        "--restart-audit", str(RESULTS / "MIDPOINT_RESTART_AUDIT.json"),
        "--final-checkpoint-seal", str(RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
        "--large-panel-manifest", str(PRETRAIN / "PANEL_MANIFEST_LARGE.json"),
        "--core-panel-manifest", str(PRETRAIN / "PANEL_MANIFEST_CORE.json"),
        "--shuffle-manifest", str(PRETRAIN / "SHUFFLE_MANIFEST.json"),
        "--local-backup-audit", str(RESULTS / "LOCAL_BACKUP_AUDIT.json"),
    ]
    workflow.driver("paired statistics, audit, and classification", *analyze)

    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    workflow.rsync_from(f"{PRETRAIN}/", f"{LOCAL_RESULTS}/", "copy frozen manifests to local result tree")
    workflow.rsync_from(f"{PREFLIGHT}/", f"{LOCAL_RESULTS}/", "copy preflight artifacts to local result tree")
    workflow.rsync_from(f"{RESULTS}/", f"{LOCAL_RESULTS}/", "copy scientific results to local result tree")
    shutil.copy2(workflow.runtime_log, LOCAL_RESULTS / "WORKFLOW_RUNTIME_PRESTOP.jsonl")
    git_output(workflow, "add", "--", str(LOCAL_RESULTS.relative_to(LOCAL_REPO)))
    status = git_output(workflow, "status", "--porcelain")
    if not status:
        raise WorkflowError("scientific result tree produced no Git changes")
    workflow.run(
        ["git", "commit", "-m", "Record Experiment 2D5C scientific results"],
        "commit scientific results", cwd=LOCAL_REPO,
    )
    scientific_commit = git_output(workflow, "rev-parse", "HEAD")
    workflow.run(
        ["git", "push", "origin", BRANCH], "push scientific-results branch",
        cwd=LOCAL_REPO,
    )
    if subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{FINAL_TAG}"],
        cwd=LOCAL_REPO, check=False, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise WorkflowError(f"refusing to replace existing final tag {FINAL_TAG}")
    workflow.run(
        ["git", "tag", "-a", FINAL_TAG, scientific_commit, "-m",
         "Experiment 2D5C sealed scientific results"],
        "create immutable scientific-results tag", cwd=LOCAL_REPO,
    )
    workflow.run(
        ["git", "push", "origin", FINAL_TAG], "push scientific-results tag",
        cwd=LOCAL_REPO,
    )
    origin_branch = git_output(workflow, "ls-remote", "origin", f"refs/heads/{BRANCH}")
    origin_tag_peeled = git_output(
        workflow, "ls-remote", "origin", f"refs/tags/{FINAL_TAG}^{{}}"
    )
    branch_sha = origin_branch.split()[0] if origin_branch.split() else None
    tag_sha = origin_tag_peeled.split()[0] if origin_tag_peeled.split() else None
    verification = {
        "schema": GIT_VERIFICATION_SCHEMA,
        "experiment": EXPERIMENT,
        "branch": BRANCH,
        "final_tag": FINAL_TAG,
        "freeze_commit": freeze_head,
        "implementation_commit": freeze_head,
        "scientific_results_commit": scientific_commit,
        "origin_branch_commit": branch_sha,
        "local_tag_commit": git_output(
            workflow, "rev-parse", f"{FINAL_TAG}^{{commit}}"
        ),
        "origin_tag_commit": tag_sha,
        "branch_push_verified": branch_sha == scientific_commit,
        "tag_push_verified": tag_sha == scientific_commit,
        "worktree_clean": not bool(git_output(workflow, "status", "--porcelain")),
        "verified_at_unix": time.time(),
    }
    verification["passed"] = (
        verification["branch_push_verified"]
        and verification["tag_push_verified"]
        and verification["local_tag_commit"] == scientific_commit
        and verification["worktree_clean"]
    )
    if not verification["passed"]:
        raise WorkflowError(f"Git push/tag verification failed: {verification}")
    write_json_atomic(LOCAL_RESULTS / "GIT_VERIFICATION.json", verification)
    return {
        "scientific_commit": scientific_commit,
        "git_verification": verification,
        "backup": backup,
    }


def preserve_failure_best_effort(workflow: Workflow, error: Exception) -> dict:
    """Preserve the latest strict evidence before the supervising guard stops."""
    stamp = int(time.time())
    attempt = LOCAL_ARCHIVE / f"failure_{stamp}"
    attempt.mkdir(parents=True, exist_ok=False)
    report_name = f"FAILURE_REPORT_{stamp}.json"
    remote_report = RESULTS / report_name
    report = {
        "experiment": EXPERIMENT,
        "status": "terminal_failure_before_scientific_completion",
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at_unix": time.time(),
        "pod_stop_delegated_to_independent_guard": True,
        "network_volume_retention_required": True,
        "no_substitute_arm_authorized": True,
        "preservation": {},
    }
    remote_writer = (
        "import json,os,sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
        "f=p.open('x',encoding='utf-8'); json.dump(json.loads(sys.argv[2]),f,sort_keys=True,indent=2); "
        "f.write('\\n'); f.flush(); os.fsync(f.fileno()); f.close()"
    )
    remote_command = shlex.join([
        "python3", "-c", remote_writer, str(remote_report),
        canonical_json(report),
    ])
    written = subprocess.run(
        [*workflow.ssh_prefix(), remote_command], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    report["preservation"]["remote_failure_report_written"] = written

    latest_result = subprocess.run(
        [*workflow.ssh_prefix(),
         f"find {shlex.quote(str(CHECKPOINTS))} -maxdepth 1 -type f "
         "-name '*.pt' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    latest = None
    if latest_result.returncode == 0 and latest_result.stdout.strip():
        latest = Path(latest_result.stdout.strip().split(" ", 1)[1])
        checkpoint_ok = subprocess.run([
            "rsync", "-a", "--partial", "-e", workflow.rsync_shell(),
            f"{workflow.remote}:{latest}", str(attempt / latest.name),
        ], check=False).returncode == 0
        sidecars = {}
        for suffix in (".sha256", ".verification.json"):
            sidecars[suffix] = subprocess.run([
                "rsync", "-a", "--partial", "-e", workflow.rsync_shell(),
                f"{workflow.remote}:{latest}{suffix}", str(attempt / f"{latest.name}{suffix}"),
            ], check=False).returncode == 0
        report["preservation"]["latest_checkpoint"] = {
            "remote_path": str(latest),
            "local_path": str(attempt / latest.name),
            "checkpoint_copied": checkpoint_ok,
            "sidecars": sidecars,
            "local_sha256": sha256(attempt / latest.name) if checkpoint_ok else None,
        }
    else:
        report["preservation"]["latest_checkpoint"] = None

    copied_roots = {}
    for label, remote_root in (
        ("pretrain", PRETRAIN), ("preflight", PREFLIGHT), ("results", RESULTS)
    ):
        destination = attempt / label
        destination.mkdir(parents=True, exist_ok=True)
        copied_roots[label] = subprocess.run([
            "rsync", "-a", "--partial", "-e", workflow.rsync_shell(),
            f"{workflow.remote}:{remote_root}/", f"{destination}/",
        ], check=False).returncode == 0
    report["preservation"]["artifact_roots_copied"] = copied_roots
    report["preservation"]["attempt_directory"] = str(attempt)
    write_json_atomic(attempt / report_name, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised host-side official Experiment 2D5C workflow"
    )
    parser.add_argument("--authorization-artifact", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument(
        "--runtime-log", type=Path,
        default=LOCAL_ARCHIVE / "WORKFLOW_RUNTIME.jsonl",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workflow = Workflow(args.runtime_log, args.ssh_host, args.ssh_port)
    try:
        result = run_scientific_workflow(workflow, args.authorization_artifact.resolve())
    except Exception as error:
        try:
            preservation = preserve_failure_best_effort(workflow, error)
            preservation_path = preservation["preservation"]["attempt_directory"]
        except Exception as preservation_error:
            preservation_path = f"preservation attempt also failed: {preservation_error}"
        append_runtime(workflow.runtime_log, {
            "experiment": EXPERIMENT,
            "event": "terminal_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "failure_preservation": preservation_path,
            "finished_at_unix": time.time(),
        })
        print(f"EXPERIMENT_2D5C_WORKFLOW_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    append_runtime(workflow.runtime_log, {
        "experiment": EXPERIMENT,
        "event": "terminal_success",
        "scientific_commit": result["scientific_commit"],
        "finished_at_unix": time.time(),
    })
    print(
        f"EXPERIMENT_2D5C_SCIENTIFIC_WORKFLOW_COMPLETE {result['scientific_commit']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
