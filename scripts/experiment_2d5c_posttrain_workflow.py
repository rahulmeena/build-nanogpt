#!/usr/bin/env python3
"""Append-only post-training completion workflow for Experiment 2D5C.

This host-side workflow is deliberately separate from the frozen 2D5C
workflow and driver.  It is authorized to start only after the exact official
191-update C training run has finished.  It has no preparation, preflight, or
training code path.

The frozen training checkout remains clean at its original implementation
commit.  A separately committed adjudicator records the inherited optimizer
step-vector continuity, the unchanged frozen driver is then required to fail
its legacy final-seal predicate in exactly the one known way, and the
adjudicator derives a truthful final seal from that complete evidence.  All
evaluation and diagnostic commands continue to execute from the unchanged
frozen checkout.

This script never calls RunPod.  It is intended to be the scientific child of
``experiment_2d5c_runpod_guard.py watchdog``: a non-zero outcome preserves
evidence and leaves the pod running, while success lets the independent guard
perform its exact-pod stop.
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
from typing import Sequence

import experiment_2d5c_workflow as frozen


EXPERIMENT = frozen.EXPERIMENT
BRANCH = frozen.BRANCH
FINAL_TAG = frozen.FINAL_TAG
GIT_VERIFICATION_SCHEMA = frozen.GIT_VERIFICATION_SCHEMA

TRAINING_FREEZE_COMMIT = "4df3cfaaa486a7162485862ea521e69c47d5147d"
FROZEN_DRIVER_SHA256 = (
    "204e589d4dfd68ed2136d70d2470c6bc06a85f5e89b706a7075706d039081b48"
)
ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256 = (
    "8fef8253f596d668462a8e4c313a762105c63a36bc23f7db3ee25fc9db04579c"
)
FINAL_CHECKPOINT_SHA256 = (
    "f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe"
)

REMOTE_REPO = frozen.REMOTE_REPO
RUN_ROOT = frozen.RUN_ROOT
PRETRAIN = frozen.PRETRAIN
PREFLIGHT = frozen.PREFLIGHT
RESULTS = frozen.RESULTS
CHECKPOINTS = frozen.CHECKPOINTS
SOURCE = frozen.SOURCE
CONTROL = frozen.CONTROL
DATA_ROOT = frozen.DATA_ROOT
LOCAL_REPO = frozen.LOCAL_REPO
LOCAL_RESULTS = frozen.LOCAL_RESULTS
LOCAL_ARCHIVE = frozen.LOCAL_ARCHIVE

ORIGINAL_TRAINING_COMPLETE = RESULTS / "TRAINING_COMPLETE.json"
ADJUDICATED_TRAINING_COMPLETE = RESULTS / "TRAINING_COMPLETE_ADJUDICATED.json"
LEGACY_FAILED_FINAL_SEAL = (
    RESULTS / "FINAL_CHECKPOINT_PROVENANCE_LEGACY_FAILED.json"
)
FINAL_SEAL = RESULTS / "FINAL_CHECKPOINT_PROVENANCE.json"
POSTTRAIN_PROVENANCE = RESULTS / "POSTTRAIN_RESUME_PROVENANCE.json"

LOCAL_ADJUDICATOR = (
    LOCAL_REPO / "scripts" / "experiment_2d5c_posttrain_adjudicator.py"
)
LOCAL_WORKFLOW = Path(__file__).resolve()


class PostTrainingWorkflowError(RuntimeError):
    """A fail-closed post-training workflow error."""


def safe_ssh_host(value: str) -> str:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def git_output(workflow: frozen.Workflow, *args: str) -> str:
    return workflow.run(
        ["git", *args],
        f"post-training git {' '.join(args[:2])}",
        cwd=LOCAL_REPO,
        capture=True,
    )


def remote_python(
    workflow: frozen.Workflow,
    source: str,
    arguments: Sequence[Path | str],
    stage: str,
) -> str:
    command = shlex.join(
        ["python3", "-c", source, *[str(value) for value in arguments]]
    )
    return workflow.remote_capture(command, stage)


def remote_path_exists(
    workflow: frozen.Workflow, path: Path, stage: str
) -> bool:
    result = workflow.remote_capture(
        "if test -e " + shlex.quote(str(path)) + "; then echo yes; else echo no; fi",
        stage,
    )
    if result not in {"yes", "no"}:
        raise PostTrainingWorkflowError(f"invalid remote path probe for {path}: {result}")
    return result == "yes"


def assert_private_authorization(path: Path) -> None:
    frozen.assert_private_authorization(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PostTrainingWorkflowError(
            "post-training guard authorization must have exact mode 0600"
        )


def verify_local_and_frozen_checkout(
    workflow: frozen.Workflow,
    authorization: Path,
    training_freeze_commit: str,
) -> str:
    """Bind the host tooling commit and unchanged remote scientific checkout."""
    assert_private_authorization(authorization)
    if training_freeze_commit != TRAINING_FREEZE_COMMIT:
        raise PostTrainingWorkflowError("unexpected 2D5C training freeze commit")
    if not frozen.IDENTITY_FILE.is_file():
        raise PostTrainingWorkflowError(f"missing SSH identity {frozen.IDENTITY_FILE}")
    if not LOCAL_ADJUDICATOR.is_file():
        raise PostTrainingWorkflowError(
            f"missing separately committed adjudicator {LOCAL_ADJUDICATOR}"
        )
    if LOCAL_WORKFLOW.parent != LOCAL_ADJUDICATOR.parent:
        raise PostTrainingWorkflowError("post-training tooling is outside scripts/")

    if git_output(workflow, "branch", "--show-current") != BRANCH:
        raise PostTrainingWorkflowError("local scientific branch mismatch")
    if git_output(workflow, "status", "--porcelain"):
        raise PostTrainingWorkflowError(
            "local repository must be clean before post-training completion"
        )
    adjudication_commit = git_output(workflow, "rev-parse", "HEAD")
    if git_output(workflow, "rev-parse", f"origin/{BRANCH}") != adjudication_commit:
        raise PostTrainingWorkflowError("post-training tooling commit is not pushed")
    if subprocess.run(
        [
            "git", "merge-base", "--is-ancestor", training_freeze_commit,
            adjudication_commit,
        ],
        cwd=LOCAL_REPO,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise PostTrainingWorkflowError(
            "post-training tooling commit does not descend from training freeze"
        )
    for relative in (
        "scripts/experiment_2d5c_posttrain_adjudicator.py",
        "scripts/experiment_2d5c_posttrain_workflow.py",
    ):
        tracked = git_output(workflow, "ls-files", "--error-unmatch", relative)
        if tracked != relative:
            raise PostTrainingWorkflowError(f"untracked post-training tool: {relative}")
    if subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{FINAL_TAG}"],
        cwd=LOCAL_REPO,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise PostTrainingWorkflowError(f"final tag already exists: {FINAL_TAG}")
    if git_output(workflow, "ls-remote", "origin", f"refs/tags/{FINAL_TAG}*"):
        raise PostTrainingWorkflowError(f"remote final tag already exists: {FINAL_TAG}")

    remote_state_source = r"""
import hashlib,json,subprocess,sys
from pathlib import Path
repo=Path(sys.argv[1]); expected=sys.argv[2]
driver=repo/'scripts'/'experiment_2d5c.py'
h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={
  'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
  'branch':subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),
  'status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),
  'driver_sha256':h(driver),
  'expected':expected,
}
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
"""
    remote_state = json.loads(
        remote_python(
            workflow,
            remote_state_source,
            (REMOTE_REPO, training_freeze_commit),
            "verify unchanged frozen 2D5C training checkout",
        )
    )
    if not (
        remote_state.get("head") == training_freeze_commit
        and remote_state.get("expected") == training_freeze_commit
        and remote_state.get("branch") == BRANCH
        and remote_state.get("status") == ""
        and remote_state.get("driver_sha256") == FROZEN_DRIVER_SHA256
    ):
        raise PostTrainingWorkflowError(
            f"frozen remote training checkout identity mismatch: {remote_state}"
        )
    return adjudication_commit


def verify_exact_completed_training(workflow: frozen.Workflow) -> dict:
    """Require the exact failed terminal audit and exact completed training."""
    final_checkpoint = frozen.checkpoint(191)
    source = r"""
import hashlib,json,sys
from pathlib import Path
results=Path(sys.argv[1]); preflight=Path(sys.argv[2]); checkpoints=Path(sys.argv[3])
final_checkpoint=Path(sys.argv[4]); expected_freeze=sys.argv[5]
expected_training_sha=sys.argv[6]; expected_checkpoint_sha=sys.argv[7]
h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
training_path=results/'TRAINING_COMPLETE.json'
log_path=results/'TRAINING_LOG.jsonl'
actual_path=results/'TRAINING_REPLAY_ACTUAL.jsonl'
milestone_path=results/'MILESTONE_CHECKPOINTS.json'
restart_pre=results/'MIDPOINT_RESTART_PREEXIT.json'
restart_audit=results/'MIDPOINT_RESTART_AUDIT.json'
heartbeat=results/'HEARTBEAT_C.json'
required=[training_path,log_path,actual_path,milestone_path,restart_pre,restart_audit,
          final_checkpoint,Path(str(final_checkpoint)+'.sha256'),
          Path(str(final_checkpoint)+'.verification.json'),
          preflight/'PREFLIGHT_AUDIT.json',preflight/'PREFLIGHT_TESTS.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('missing exact completed-training evidence: '+repr(missing))
training=json.loads(training_path.read_text())
rows=[json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
actual=[json.loads(x) for x in actual_path.read_text().splitlines() if x.strip()]
milestones=json.loads(milestone_path.read_text())
pre=json.loads((preflight/'PREFLIGHT_AUDIT.json').read_text())
sidecar=json.loads(Path(str(final_checkpoint)+'.verification.json').read_text())
expected_checkpoint_names=[
 'scientific_cumulative_001025507328.pt',
 'scientific_cumulative_001050673152.pt',
 'scientific_cumulative_001075838976.pt',
 'scientific_cumulative_001100480512.pt',
]
checkpoint_names=sorted(p.name for p in checkpoints.glob('*.pt') if p.is_file())
trainers=[]
for command_path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv=[part.decode('utf-8','replace') for part in command_path.read_bytes().split(b'\0') if part]
    except (OSError,PermissionError):
        continue
    if any(part.endswith('scripts/experiment_2d5c.py') for part in argv) and 'train' in argv:
        trainers.append({'pid':command_path.parent.name,'argv':argv})
false_checks=sorted(k for k,v in training.get('checks',{}).items() if v is not True)
lineage={
 'rows_exact':len(rows)==191 and [r.get('local_update') for r in rows]==list(range(1,192)),
 'actual_rows_exact':len(actual)==191 and [r.get('local_update') for r in actual]==list(range(1,192)),
 'increment_flags':len(rows)==191 and all(r.get('optimizer_step_increment_exact') is True for r in rows),
 'elementwise_increment':len(rows)==191 and all(r.get('optimizer_steps_after_summary')==[v+1 for v in r.get('optimizer_steps_before_summary',[])] for r in rows),
 'adjacent_chain':len(rows)==191 and all(rows[i].get('optimizer_steps_before_summary')==rows[i-1].get('optimizer_steps_after_summary') for i in range(1,191)),
 'first_exact':len(rows)==191 and rows[0].get('optimizer_steps_before_summary')==[1908,2386],
 'terminal_exact':len(rows)==191 and rows[-1].get('optimizer_steps_after_summary')==[2099,2577],
 'whole_run_delta':len(rows)==191 and rows[-1].get('optimizer_steps_after_summary')==[v+191 for v in rows[0].get('optimizer_steps_before_summary',[])],
}
payload={
 'training_complete_sha256':h(training_path),
 'final_checkpoint_sha256':h(final_checkpoint),
 'training_schema':training.get('schema'),
 'training_passed':training.get('passed'),
 'false_checks':false_checks,
 'optimizer_evidence':training.get('optimizer_evidence'),
 'lineage':lineage,
 'preflight_commit':pre.get('git_commit'),
 'preflight_authorized':pre.get('authorized'),
 'milestone_keys':sorted(milestones),
 'final_milestone_sha256':milestones.get('191',{}).get('sha256'),
 'final_sidecar_sha256':sidecar.get('sha256'),
 'final_sidecar_local_update':sidecar.get('local_update'),
 'final_sidecar_global_update':sidecar.get('global_update'),
 'final_sidecar_cumulative_targets':sidecar.get('cumulative_targets'),
 'final_sidecar_strict_reopen':sidecar.get('strict_reopen',{}).get('passed'),
 'checkpoint_names':checkpoint_names,
 'expected_checkpoint_names':expected_checkpoint_names,
 'active_training_processes':trainers,
 'expected_training_sha256':expected_training_sha,
 'expected_checkpoint_sha256':expected_checkpoint_sha,
 'heartbeat_status':json.loads(heartbeat.read_text()).get('status') if heartbeat.is_file() else None,
 'heartbeat_local_update':json.loads(heartbeat.read_text()).get('local_update') if heartbeat.is_file() else None,
}
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
"""
    snapshot = json.loads(
        remote_python(
            workflow,
            source,
            (
                RESULTS,
                PREFLIGHT,
                CHECKPOINTS,
                final_checkpoint,
                TRAINING_FREEZE_COMMIT,
                ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256,
                FINAL_CHECKPOINT_SHA256,
            ),
            "audit exact already-completed 191-update training",
        )
    )
    checks = {
        "original_training_complete_unchanged": snapshot.get(
            "training_complete_sha256"
        )
        == ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256,
        "final_checkpoint_unchanged": snapshot.get("final_checkpoint_sha256")
        == FINAL_CHECKPOINT_SHA256,
        "training_schema": snapshot.get("training_schema")
        == "experiment_2d5c_training_complete_v1",
        "legacy_failure_exact": snapshot.get("training_passed") is False
        and snapshot.get("false_checks") == ["optimizer_terminal_step_exact"],
        "optimizer_evidence_exact": snapshot.get("optimizer_evidence", {}).get(
            "step_summary"
        )
        == [2099, 2577],
        "optimizer_lineage_exact": bool(snapshot.get("lineage"))
        and all(snapshot["lineage"].values()),
        "preflight_exact": snapshot.get("preflight_commit")
        == TRAINING_FREEZE_COMMIT
        and snapshot.get("preflight_authorized") is True,
        "milestones_exact": snapshot.get("milestone_keys")
        == ["144", "191", "48", "96"],
        "checkpoint_bindings_exact": snapshot.get("final_milestone_sha256")
        == snapshot.get("final_sidecar_sha256")
        == FINAL_CHECKPOINT_SHA256,
        "checkpoint_terminal_counts_exact": snapshot.get(
            "final_sidecar_local_update"
        )
        == 191
        and snapshot.get("final_sidecar_global_update") == 2099
        and snapshot.get("final_sidecar_cumulative_targets") == 1_100_480_512,
        "checkpoint_strict_reopen": snapshot.get(
            "final_sidecar_strict_reopen"
        )
        is True,
        "checkpoint_set_exact": snapshot.get("checkpoint_names")
        == snapshot.get("expected_checkpoint_names"),
        "no_training_process_active": snapshot.get("active_training_processes")
        == [],
        "terminal_heartbeat_exact": snapshot.get("heartbeat_status")
        == "checkpoint_verified"
        and snapshot.get("heartbeat_local_update") == 191,
    }
    if not all(checks.values()):
        raise PostTrainingWorkflowError(
            f"exact completed-training entry audit failed: {checks}"
        )
    snapshot["checks"] = checks
    snapshot["passed"] = True
    return snapshot


def install_and_bind_adjudicator(
    workflow: frozen.Workflow, adjudication_commit: str
) -> Path:
    """Copy the committed adjudicator outside the frozen checkout and hash-bind it."""
    local_digest = sha256(LOCAL_ADJUDICATOR)
    remote_dir = Path("/tmp/exp2d5c-posttrain-tools") / adjudication_commit
    remote_path = remote_dir / LOCAL_ADJUDICATOR.name
    workflow.remote_capture(
        "set -eu; mkdir -p " + shlex.quote(str(remote_dir)),
        "create isolated post-training adjudicator directory",
    )
    if remote_path_exists(
        workflow, remote_path, "probe existing remote post-training adjudicator"
    ):
        remote_digest = workflow.remote_capture(
            f"sha256sum {shlex.quote(str(remote_path))} | awk '{{print $1}}'",
            "verify existing remote post-training adjudicator",
        )
        if remote_digest != local_digest:
            raise PostTrainingWorkflowError(
                "remote adjudicator path exists with a different byte identity"
            )
    else:
        workflow.rsync_to(
            LOCAL_ADJUDICATOR,
            remote_path,
            "copy committed adjudicator outside frozen checkout",
        )
    remote_digest = workflow.remote_capture(
        f"sha256sum {shlex.quote(str(remote_path))} | awk '{{print $1}}'",
        "bind remote adjudicator byte identity",
    )
    if remote_digest != local_digest:
        raise PostTrainingWorkflowError("remote adjudicator SHA-256 mismatch")
    return remote_path


def adjudicator_arguments() -> list[str]:
    return [
        "--source-checkpoint", str(SOURCE),
        "--final-checkpoint", str(frozen.checkpoint(191)),
        "--training-log", str(RESULTS / "TRAINING_LOG.jsonl"),
        "--training-complete-original", str(ORIGINAL_TRAINING_COMPLETE),
        "--training-replay-actual", str(RESULTS / "TRAINING_REPLAY_ACTUAL.jsonl"),
        "--replay-ledger", str(PRETRAIN / "DATA_REPLAY_LEDGER.jsonl"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
        "--midpoint-restart-preexit", str(RESULTS / "MIDPOINT_RESTART_PREEXIT.json"),
        "--midpoint-restart-audit", str(RESULTS / "MIDPOINT_RESTART_AUDIT.json"),
    ]


def run_remote_adjudicator(
    workflow: frozen.Workflow,
    remote_adjudicator: Path,
    command: str,
    arguments: Sequence[str],
    stage: str,
) -> None:
    remote_command = (
        f"cd {shlex.quote(str(REMOTE_REPO))} && "
        + shlex.join(
            ["exec", "python3", "-u", str(remote_adjudicator), command, *arguments]
        )
    )
    workflow.run([*workflow.ssh_prefix(), remote_command], stage)


def remote_adjudication_state(workflow: frozen.Workflow) -> dict:
    source = r"""
import hashlib,json,sys
from pathlib import Path
paths=[Path(value) for value in sys.argv[1:]]
h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={'exists':[path.is_file() for path in paths]}
if paths[0].is_file():
    row=json.loads(paths[0].read_text()); embedded=row.get('posttrain_adjudication',{}); payload['training_sha256']=h(paths[0]); payload['training_passed']=row.get('passed'); payload['training_false']=sorted(k for k,v in row.get('checks',{}).items() if v is not True); payload['training_optimizer']=row.get('optimizer_evidence',{}).get('step_summary'); payload['training_embedded_passed']=embedded.get('passed'); payload['training_original_sha']=embedded.get('original_training_complete',{}).get('sha256'); payload['training_tool_sha']=embedded.get('groups',{}).get('frozen_implementation',{}).get('adjudicator',{}).get('sha256')
if paths[1].is_file():
    row=json.loads(paths[1].read_text()); payload['legacy_file_sha256']=h(paths[1]); payload['legacy_sealed']=row.get('sealed'); payload['legacy_false']=sorted(k for k,v in row.get('checks',{}).items() if v is not True); payload['legacy_sha']=row.get('checkpoint_sha256'); payload['legacy_training_sha']=row.get('artifact_identity',{}).get('training_complete',{}).get('sha256')
if paths[2].is_file():
    row=json.loads(paths[2].read_text()); embedded=row.get('posttrain_adjudication',{}); artifacts=row.get('artifact_identity',{}); fresh=embedded.get('fresh_optimizer_and_row_adjudication',{}); payload['final_sealed']=row.get('sealed'); payload['final_false']=sorted(k for k,v in row.get('checks',{}).items() if v is not True); payload['final_sha']=row.get('checkpoint_sha256'); payload['final_update']=row.get('local_update'); payload['final_embedded_passed']=embedded.get('passed'); payload['final_training_sha']=artifacts.get('training_adjudication',{}).get('sha256'); payload['final_legacy_sha']=artifacts.get('legacy_failed_seal',{}).get('sha256'); payload['final_original_sha']=artifacts.get('original_training_complete',{}).get('sha256'); payload['final_tool_sha']=fresh.get('groups',{}).get('frozen_implementation',{}).get('adjudicator',{}).get('sha256')
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
"""
    return json.loads(
        remote_python(
            workflow,
            source,
            (
                ADJUDICATED_TRAINING_COMPLETE,
                LEGACY_FAILED_FINAL_SEAL,
                FINAL_SEAL,
            ),
            "inspect append-only post-training adjudication state",
        )
    )


def require_valid_adjudicated_training(state: dict) -> None:
    if not (
        state.get("training_passed") is True
        and state.get("training_false") == []
        and state.get("training_optimizer") == [2099, 2577]
        and state.get("training_embedded_passed") is True
        and state.get("training_original_sha")
        == ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256
        and state.get("training_tool_sha") == sha256(LOCAL_ADJUDICATOR)
    ):
        raise PostTrainingWorkflowError(
            f"existing adjudicated training artifact is invalid: {state}"
        )


def require_valid_legacy_seal(state: dict) -> None:
    if not (
        state.get("legacy_sealed") is False
        and state.get("legacy_false") == ["optimizer_step_exact"]
        and state.get("legacy_sha") == FINAL_CHECKPOINT_SHA256
        and state.get("legacy_training_sha") == state.get("training_sha256")
    ):
        raise PostTrainingWorkflowError(
            f"legacy frozen seal did not fail in exactly the known way: {state}"
        )


def require_valid_final_seal(state: dict) -> None:
    if not (
        state.get("final_sealed") is True
        and state.get("final_false") == []
        and state.get("final_sha") == FINAL_CHECKPOINT_SHA256
        and state.get("final_update") == 191
        and state.get("final_embedded_passed") is True
        and state.get("final_training_sha") == state.get("training_sha256")
        and state.get("final_legacy_sha") == state.get("legacy_file_sha256")
        and state.get("final_original_sha")
        == ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256
        and state.get("final_tool_sha") == sha256(LOCAL_ADJUDICATOR)
    ):
        raise PostTrainingWorkflowError(
            f"adjudicated final seal is invalid: {state}"
        )


def run_frozen_seal_expect_known_failure(workflow: frozen.Workflow) -> None:
    arguments = [
        "seal-final",
        "--checkpoint", str(frozen.checkpoint(191)),
        "--source-checkpoint", str(SOURCE),
        "--training-complete", str(ADJUDICATED_TRAINING_COMPLETE),
        "--training-log", str(RESULTS / "TRAINING_LOG.jsonl"),
        "--training-replay-actual", str(RESULTS / "TRAINING_REPLAY_ACTUAL.jsonl"),
        "--replay-ledger", str(PRETRAIN / "DATA_REPLAY_LEDGER.jsonl"),
        "--replay-audit", str(PRETRAIN / "DATA_REPLAY_AUDIT.json"),
        "--midpoint-restart-preexit", str(RESULTS / "MIDPOINT_RESTART_PREEXIT.json"),
        "--midpoint-restart-audit", str(RESULTS / "MIDPOINT_RESTART_AUDIT.json"),
        "--output-path", str(LEGACY_FAILED_FINAL_SEAL),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
    ]
    remote_command = (
        f"cd {shlex.quote(str(REMOTE_REPO))} && "
        + shlex.join(
            ["exec", "python3", "-u", "scripts/experiment_2d5c.py", *arguments]
        )
    )
    started = time.time()
    frozen.append_runtime(
        workflow.runtime_log,
        {
            "experiment": EXPERIMENT,
            "stage": "unchanged frozen final seal (known legacy predicate)",
            "event": "start",
            "started_at_unix": started,
            "argv": [*workflow.ssh_prefix(), remote_command],
            "expected_returncode": 1,
            "frozen_commit": TRAINING_FREEZE_COMMIT,
        },
    )
    print(
        "[2D5C post-training workflow] unchanged frozen final seal; "
        "expecting exactly one adjudicated legacy predicate failure",
        flush=True,
    )
    result = subprocess.run(
        [*workflow.ssh_prefix(), remote_command],
        check=False,
        text=True,
    )
    frozen.append_runtime(
        workflow.runtime_log,
        {
            "experiment": EXPERIMENT,
            "stage": "unchanged frozen final seal (known legacy predicate)",
            "event": "finish",
            "finished_at_unix": time.time(),
            "wall_seconds": time.time() - started,
            "returncode": result.returncode,
            "expected_returncode": 1,
        },
    )
    if result.returncode != 1:
        raise PostTrainingWorkflowError(
            "unchanged frozen final seal did not exit with the exact expected code 1"
        )


def complete_adjudication(
    workflow: frozen.Workflow, remote_adjudicator: Path
) -> None:
    """Create or verify each append-only adjudication stage in order."""
    state = remote_adjudication_state(workflow)
    exists = state.get("exists")
    if not isinstance(exists, list) or len(exists) != 3:
        raise PostTrainingWorkflowError(f"invalid adjudication state: {state}")
    if exists[1] and not exists[0]:
        raise PostTrainingWorkflowError("legacy seal exists without training adjudication")
    if exists[2] and not (exists[0] and exists[1]):
        raise PostTrainingWorkflowError("final seal exists without its append-only ancestors")

    if not exists[0]:
        run_remote_adjudicator(
            workflow,
            remote_adjudicator,
            "adjudicate-training",
            [
                *adjudicator_arguments(),
                "--output", str(ADJUDICATED_TRAINING_COMPLETE),
            ],
            "adjudicate exact inherited optimizer-step lineage",
        )
        state = remote_adjudication_state(workflow)
    require_valid_adjudicated_training(state)

    if not state["exists"][1]:
        run_frozen_seal_expect_known_failure(workflow)
        state = remote_adjudication_state(workflow)
    require_valid_legacy_seal(state)

    if not state["exists"][2]:
        run_remote_adjudicator(
            workflow,
            remote_adjudicator,
            "adjudicate-seal",
            [
                *adjudicator_arguments(),
                "--training-adjudication", str(ADJUDICATED_TRAINING_COMPLETE),
                "--legacy-failed-seal", str(LEGACY_FAILED_FINAL_SEAL),
                "--output", str(FINAL_SEAL),
            ],
            "derive final checkpoint seal from exact legacy failure evidence",
        )
        state = remote_adjudication_state(workflow)
    require_valid_final_seal(state)


def evaluation_stages() -> list[tuple[str, list[str]]]:
    return [
        (
            "Parent core",
            frozen.evaluate_args("Parent", "PARENT_CORE", 0, all_real_only=True),
        ),
        (
            "C0 core and secondary parallel",
            frozen.evaluate_args("C0", "C0_CORE", 0, parallel=True),
        ),
        (
            "C48 core",
            frozen.evaluate_args(
                "C", "C48_CORE", 48, checkpoint_path=frozen.checkpoint(48)
            ),
        ),
        (
            "C96 core and secondary parallel",
            frozen.evaluate_args(
                "C", "C96_CORE", 96,
                checkpoint_path=frozen.checkpoint(96), parallel=True,
            ),
        ),
        (
            "C144 core",
            frozen.evaluate_args(
                "C", "C144_CORE", 144, checkpoint_path=frozen.checkpoint(144)
            ),
        ),
        (
            "C191 core and secondary parallel",
            frozen.evaluate_args(
                "C", "C191_CORE", 191,
                checkpoint_path=frozen.checkpoint(191), parallel=True,
                final_seal=True,
            ),
        ),
        (
            "Fixed100M core",
            frozen.evaluate_args(
                "Fixed", "FIXED_CORE", 191, checkpoint_path=CONTROL
            ),
        ),
        (
            "C191 final large",
            frozen.evaluate_args(
                "C", "C191_LARGE", 191,
                checkpoint_path=frozen.checkpoint(191), panel="large",
                final_seal=True,
            ),
        ),
        (
            "Fixed100M final large",
            frozen.evaluate_args(
                "Fixed", "FIXED_LARGE", 191,
                checkpoint_path=CONTROL, panel="large",
            ),
        ),
    ]


def representation_stages() -> list[tuple[str, list[str]]]:
    return [
        ("Parent", frozen.representation_args("Parent", "PARENT", 0)),
        ("C0", frozen.representation_args("C0", "C0", 0)),
        (
            "C48",
            frozen.representation_args("C", "C48", 48, frozen.checkpoint(48)),
        ),
        (
            "C96",
            frozen.representation_args("C", "C96", 96, frozen.checkpoint(96)),
        ),
        (
            "C144",
            frozen.representation_args(
                "C", "C144", 144, frozen.checkpoint(144)
            ),
        ),
        (
            "C191",
            frozen.representation_args(
                "C", "C191", 191, frozen.checkpoint(191), final_seal=True
            ),
        ),
        (
            "Fixed100M",
            frozen.representation_args("Fixed", "FIXED100M", 191, CONTROL),
        ),
    ]


def preserve_partial_parallel_evaluation(
    workflow: frozen.Workflow,
    arguments: Sequence[str],
    stage: str,
) -> None:
    """Move a non-resumable partial parallel artifact aside, never delete it.

    The primary true-incremental evaluator is exactly resumable.  The frozen
    secondary parallel evaluator intentionally accepts an existing artifact
    only when it is already complete.  A watchdog retry after interruption
    therefore needs a fresh parallel path, while the partial evidence remains
    retained under ``results/failed_attempts``.
    """
    if "--parallel-output" not in arguments:
        return
    index = list(arguments).index("--parallel-output")
    if index + 1 >= len(arguments):
        raise PostTrainingWorkflowError(
            f"parallel-output argument is malformed for {stage}"
        )
    parallel_path = Path(arguments[index + 1])
    source = r"""
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1]); stage=sys.argv[2]
result={'path':str(path),'action':'absent'}
if path.exists():
    payload=json.loads(path.read_text())
    if payload.get('status')=='complete':
        result={'path':str(path),'action':'complete_retained'}
    else:
        root=path.parent/'failed_attempts'/f"posttrain_parallel_{int(time.time())}_{os.getpid()}"
        root.mkdir(parents=True,exist_ok=False)
        moved=[]
        for candidate in (path,path.with_suffix('.audit.json')):
            if candidate.exists():
                destination=root/candidate.name
                os.replace(candidate,destination)
                moved.append(str(destination))
        record=root/'PRESERVATION.json'
        record.write_text(json.dumps({'stage':stage,'original':str(path),'moved':moved},sort_keys=True,indent=2)+'\n')
        result={'path':str(path),'action':'partial_preserved','directory':str(root),'moved':moved}
print(json.dumps(result,sort_keys=True,separators=(',',':')))
"""
    result = json.loads(
        remote_python(
            workflow,
            source,
            (parallel_path, stage),
            f"inspect resumability of secondary parallel evaluation: {stage}",
        )
    )
    if result.get("action") not in {
        "absent", "complete_retained", "partial_preserved",
    }:
        raise PostTrainingWorkflowError(
            f"invalid parallel evaluation preservation outcome: {result}"
        )


def analysis_arguments() -> list[str]:
    return [
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
        "--training-complete", str(ADJUDICATED_TRAINING_COMPLETE),
        "--restart-audit", str(RESULTS / "MIDPOINT_RESTART_AUDIT.json"),
        "--final-checkpoint-seal", str(FINAL_SEAL),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
        "--large-panel-manifest", str(PRETRAIN / "PANEL_MANIFEST_LARGE.json"),
        "--core-panel-manifest", str(PRETRAIN / "PANEL_MANIFEST_CORE.json"),
        "--shuffle-manifest", str(PRETRAIN / "SHUFFLE_MANIFEST.json"),
        "--local-backup-audit", str(RESULTS / "LOCAL_BACKUP_AUDIT.json"),
    ]


def write_and_copy_posttrain_provenance(
    workflow: frozen.Workflow,
    adjudication_commit: str,
    training_snapshot: dict,
) -> dict:
    local_path = LOCAL_ARCHIVE / "POSTTRAIN_RESUME_PROVENANCE.json"
    payload = {
        "schema": "experiment_2d5c_posttrain_resume_provenance_v1",
        "experiment": EXPERIMENT,
        "training_implementation_commit": TRAINING_FREEZE_COMMIT,
        "frozen_driver_sha256": FROZEN_DRIVER_SHA256,
        "adjudication_tooling_commit": adjudication_commit,
        "adjudicator_sha256": sha256(LOCAL_ADJUDICATOR),
        "posttrain_workflow_sha256": sha256(LOCAL_WORKFLOW),
        "original_failed_training_complete_sha256": (
            ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256
        ),
        "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "training_entry_audit": training_snapshot,
        "original_failed_training_artifact_retained": True,
        "legacy_failed_seal_retained": True,
        "no_training_command_available_or_invoked": True,
        "posttraining_only": True,
        "completed_at_unix": time.time(),
    }
    if local_path.exists():
        existing = json.loads(local_path.read_text(encoding="utf-8"))
        if existing != payload:
            # The timestamp is intentionally retained from the first durable
            # write.  All scientific/provenance content must otherwise match.
            comparable_existing = dict(existing)
            comparable_payload = dict(payload)
            comparable_payload["completed_at_unix"] = comparable_existing.get(
                "completed_at_unix"
            )
            if comparable_existing != comparable_payload:
                raise PostTrainingWorkflowError(
                    "existing local post-training provenance has different content"
                )
            payload = existing
    else:
        frozen.write_json_atomic(local_path, payload)

    if remote_path_exists(
        workflow,
        POSTTRAIN_PROVENANCE,
        "probe existing remote post-training provenance",
    ):
        remote_digest = workflow.remote_capture(
            f"sha256sum {shlex.quote(str(POSTTRAIN_PROVENANCE))} | awk '{{print $1}}'",
            "verify existing remote post-training provenance",
        )
        if remote_digest != sha256(local_path):
            raise PostTrainingWorkflowError(
                "existing remote post-training provenance differs from local evidence"
            )
    else:
        workflow.rsync_to(
            local_path,
            POSTTRAIN_PROVENANCE,
            "copy post-training adjudication provenance to scientific results",
        )
    return payload


def commit_scientific_results(
    workflow: frozen.Workflow,
    adjudication_commit: str,
) -> tuple[str, dict]:
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    workflow.rsync_from(
        f"{PRETRAIN}/", f"{LOCAL_RESULTS}/",
        "copy frozen manifests to local post-training result tree",
    )
    workflow.rsync_from(
        f"{PREFLIGHT}/", f"{LOCAL_RESULTS}/",
        "copy frozen preflight artifacts to local post-training result tree",
    )
    workflow.rsync_from(
        f"{RESULTS}/", f"{LOCAL_RESULTS}/",
        "copy adjudicated scientific results to local result tree",
    )
    shutil.copy2(
        workflow.runtime_log,
        LOCAL_RESULTS / "WORKFLOW_RUNTIME_PRESTOP.jsonl",
    )
    git_output(workflow, "add", "--", str(LOCAL_RESULTS.relative_to(LOCAL_REPO)))
    if not git_output(workflow, "status", "--porcelain"):
        raise PostTrainingWorkflowError("post-training scientific result tree is empty")
    workflow.run(
        ["git", "commit", "-m", "Record Experiment 2D5C scientific results"],
        "commit adjudicated scientific results",
        cwd=LOCAL_REPO,
    )
    scientific_commit = git_output(workflow, "rev-parse", "HEAD")
    workflow.run(
        ["git", "push", "origin", BRANCH],
        "push adjudicated scientific-results branch",
        cwd=LOCAL_REPO,
    )
    workflow.run(
        [
            "git", "tag", "-a", FINAL_TAG, scientific_commit, "-m",
            "Experiment 2D5C sealed scientific results",
        ],
        "create immutable adjudicated scientific-results tag",
        cwd=LOCAL_REPO,
    )
    workflow.run(
        ["git", "push", "origin", FINAL_TAG],
        "push adjudicated scientific-results tag",
        cwd=LOCAL_REPO,
    )
    origin_branch = git_output(
        workflow, "ls-remote", "origin", f"refs/heads/{BRANCH}"
    )
    origin_tag = git_output(
        workflow, "ls-remote", "origin", f"refs/tags/{FINAL_TAG}^{{}}"
    )
    branch_sha = origin_branch.split()[0] if origin_branch.split() else None
    tag_sha = origin_tag.split()[0] if origin_tag.split() else None
    verification = {
        "schema": GIT_VERIFICATION_SCHEMA,
        "experiment": EXPERIMENT,
        "branch": BRANCH,
        "final_tag": FINAL_TAG,
        "freeze_commit": TRAINING_FREEZE_COMMIT,
        # The scientific/evaluation implementation stayed frozen at this commit.
        "implementation_commit": TRAINING_FREEZE_COMMIT,
        "training_implementation_commit": TRAINING_FREEZE_COMMIT,
        "adjudication_tooling_commit": adjudication_commit,
        "adjudicator_sha256": sha256(LOCAL_ADJUDICATOR),
        "posttrain_workflow_sha256": sha256(LOCAL_WORKFLOW),
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
        raise PostTrainingWorkflowError(
            f"post-training Git push/tag verification failed: {verification}"
        )
    frozen.write_json_atomic(LOCAL_RESULTS / "GIT_VERIFICATION.json", verification)
    return scientific_commit, verification


def run_posttraining_workflow(
    workflow: frozen.Workflow,
    authorization: Path,
    training_freeze_commit: str,
) -> dict:
    adjudication_commit = verify_local_and_frozen_checkout(
        workflow, authorization, training_freeze_commit
    )
    training_snapshot = verify_exact_completed_training(workflow)
    remote_adjudicator = install_and_bind_adjudicator(
        workflow, adjudication_commit
    )
    complete_adjudication(workflow, remote_adjudicator)

    backup = frozen.backup_checkpoints(workflow)
    workflow.driver(
        "post-training exact BF16 persistent-state audit",
        "memory-audit",
        "--fixed-checkpoint", str(CONTROL),
        "--c-checkpoint", str(frozen.checkpoint(191)),
        "--source-checkpoint", str(SOURCE),
        "--output-json", str(RESULTS / "BF16_PERSISTENT_STATE_RAW.json"),
        "--output-table", str(RESULTS / "BF16_PERSISTENT_STATE_RAW.md"),
        "--milestone-manifest", str(RESULTS / "MILESTONE_CHECKPOINTS.json"),
        "--final-checkpoint-seal", str(FINAL_SEAL),
    )
    for stage, arguments in evaluation_stages():
        preserve_partial_parallel_evaluation(workflow, arguments, stage)
        workflow.driver(
            f"post-training true-incremental evaluation: {stage}", *arguments
        )
    for stage, arguments in representation_stages():
        workflow.driver(
            f"post-training representation-pressure diagnostics: {stage}",
            *arguments,
        )
    workflow.driver(
        "post-training paired statistics, audit, and classification",
        *analysis_arguments(),
    )
    provenance = write_and_copy_posttrain_provenance(
        workflow, adjudication_commit, training_snapshot
    )
    scientific_commit, git_verification = commit_scientific_results(
        workflow, adjudication_commit
    )
    return {
        "scientific_commit": scientific_commit,
        "git_verification": git_verification,
        "backup": backup,
        "posttrain_provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append-only post-training completion for exact Experiment 2D5C "
            "191-update artifacts"
        )
    )
    parser.add_argument("--authorization-artifact", type=Path, required=True)
    parser.add_argument("--ssh-host", type=safe_ssh_host, required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument(
        "--training-freeze-commit",
        required=True,
        choices=(TRAINING_FREEZE_COMMIT,),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workflow = frozen.Workflow(
        args.runtime_log.resolve(), args.ssh_host, args.ssh_port
    )
    try:
        result = run_posttraining_workflow(
            workflow,
            args.authorization_artifact.resolve(),
            args.training_freeze_commit,
        )
    except Exception as error:
        try:
            preservation = frozen.preserve_failure_best_effort(workflow, error)
            preservation_path = preservation["preservation"]["attempt_directory"]
        except Exception as preservation_error:
            preservation_path = (
                "post-training preservation attempt also failed: "
                f"{preservation_error}"
            )
        frozen.append_runtime(
            workflow.runtime_log,
            {
                "experiment": EXPERIMENT,
                "event": "posttraining_terminal_failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "failure_preservation": preservation_path,
                "pod_retention_required": True,
                "finished_at_unix": time.time(),
            },
        )
        print(
            f"EXPERIMENT_2D5C_POSTTRAIN_WORKFLOW_FAILURE: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    frozen.append_runtime(
        workflow.runtime_log,
        {
            "experiment": EXPERIMENT,
            "event": "posttraining_terminal_success",
            "scientific_commit": result["scientific_commit"],
            "finished_at_unix": time.time(),
        },
    )
    print(
        "EXPERIMENT_2D5C_POSTTRAIN_SCIENTIFIC_WORKFLOW_COMPLETE "
        f"{result['scientific_commit']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
