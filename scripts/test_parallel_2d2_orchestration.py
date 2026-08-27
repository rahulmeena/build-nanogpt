#!/usr/bin/env python3
"""Focused CPU-only tests for run scoping and supervisor terminal semantics."""

from __future__ import annotations

import json
import hashlib
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parallel_2d2_supervisor as supervisor
import parallel_2d2_heartbeat as heartbeat


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def passing_finalization_boundary() -> dict:
    return {
        "git": {"manifest": {"sha256": "a" * 64}, "passed": True},
        "report": {"manifest": {"sha256": "b" * 64}, "passed": True},
        "report_git_binding": {
            "checks": {name: True for name in supervisor.FINAL_GIT_REPOSITORIES},
            "passed": True,
        },
        "backup": {"manifest": {"sha256": "c" * 64}, "passed": True},
        "final_science": {"all_gpus_compute_idle": True},
        "scientific_processes": [],
        "recorded_lane_processes": {"passed": True},
        "authenticated_stop_identity": {"passed": True},
        "timestamp_order": {"passed": True},
        "pod_stop_invoked": False,
        "passed": True,
    }


def passing_prepublication_fingerprint() -> dict:
    return {
        "captured_after_final_process_gate": True,
        "all_required_sources_match_signed_backup_inventory": True,
        "stable_file_count": 1,
        "stable_file_inventory_sha256": "d" * 64,
        "validated_boundary_serialized_sha256": "e" * 64,
        "backup_manifest_sha256": "f" * 64,
        "backup_signature_sha256": "0" * 64,
        "passed": True,
    }


def file_evidence(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_standard_final_checkpoint(path: Path, experiment: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"scientific checkpoint {experiment}".encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    contract = supervisor.FINAL_CHECKPOINT_CONTRACTS[experiment]
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    base = {
        "checkpoint": str(path.resolve()),
        "sha256": digest,
        "bytes": size,
        contract["updates_key"]: 191,
        contract["targets_key"]: contract["targets"],
        "next_global_batch_sha256": "a" * 64,
        "next_global_batch_stream_sha256": "b" * 64,
        "strict_reopen": {key: True for key in contract["strict_keys"]},
        "passed": True,
    }
    verification = dict(base)
    if experiment == "2D2F":
        verification.update(
            local_stage={
                "path": (
                    "/tmp/parallel_2d2_ephemeral/2d2f/"
                    ".scientific_update_0191.local-stage.pt"
                ),
                "sha256": digest, "bytes": size, "strict_reopen_passed": True,
            },
            persistent_copy_lock=supervisor.CHECKPOINT_PERSIST_LOCK,
            persistent_copy_sha_verified=True,
        )
    elif experiment == "2D2H":
        local = dict(base)
        local["checkpoint"] = f"/tmp/parallel_2d2_ephemeral/2d2h/{path.name}"
        verification.update(local_stage=local, persisted_under_global_lock=True)
    elif experiment == "2D2I":
        verification.update(
            local_staged_checkpoint=f"/tmp/parallel_2d2_ephemeral/2d2i/{path.name}",
            local_and_persistent_sha_match=True,
            persist_lock=supervisor.CHECKPOINT_PERSIST_LOCK,
        )
    write_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def finalization_manifest_base(kind: str, run_id: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "run_id": run_id,
        "pod": supervisor.EXPECTED_POD,
        "created_utc": "2026-08-27T00:00:00+00:00",
    }


def synthetic_master_claims() -> dict:
    answers = {}
    for index, name in enumerate(supervisor.MASTER_SCIENTIFIC_QUESTIONS, 1):
        answers[name] = {
            "answer_code": supervisor.MASTER_SCIENTIFIC_ANSWER_CODES[name],
            "evidence": {"synthetic_exact_value": index},
            "conclusion": supervisor.MASTER_SCIENTIFIC_CONCLUSIONS[name],
        }
    matrix = {}
    for index, name in enumerate(
        ("2D2B", "2D2D", "2D2E", "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"),
        1,
    ):
        matrix[name] = {
            "classification": f"SYNTHETIC CLASSIFICATION {name}",
            "parallel": {
                "gain": index / 1000.0,
                "sequence_gap": index / 2000.0,
            } if name != "2D2E-C1" else {},
            "true_incremental": {
                "real_validation_ce": 3.0 + index / 1000.0,
                "gain": index / 10000.0,
                "sequence_gap": index / 20000.0,
                "wins_vs_off": index,
                "wins_vs_shuffled": index + 1,
            },
            "inference_state_bytes_b1": 1_000_000 + index,
            "runtime_seconds": 100.0 + index,
            "gpu_hours": (100.0 + index) / 3600.0,
            "comparison_architecture": {
                "parent": "SYNTHETIC",
                "local_window": {"B1": 2, "B2": 32, "B3": 64, "B4": 128},
                "recurrence": {
                    "B1": "B12→B1", "B2": None, "B3": "B10→B3", "B4": None,
                },
                "new_link_gate_tanh": index / 100.0,
                "passed": True,
            },
        }
    return {
        "schema": "parallel_2d2_master_scientific_claims_v1",
        "source_artifacts": {},
        "matrix": matrix,
        "answers": answers,
        "recommended_next_experiment": supervisor.MASTER_SCIENTIFIC_CONCLUSIONS["M15"],
    }


def synthetic_master_report(claims: dict, classification_line: str) -> str:
    sections = []
    for name, question in supervisor.MASTER_SCIENTIFIC_QUESTIONS.items():
        number = name[1:]
        answer = claims["answers"][name]
        sections.append(
            f"M{number}. {question}\n"
            f"ANSWER_CODE: {answer['answer_code']}\n"
            "EVIDENCE_JSON: "
            + supervisor._canonical_json(answer["evidence"])
            + "\n"
            f"CONCLUSION: {answer['conclusion']}\n"
        )
    return (
        "2D2E-C1 2D2F 2D2G 2D2H 2D2I\n"
        + classification_line
        + "\n"
        + supervisor.MASTER_CLAIMS_BEGIN
        + "\n"
        + supervisor._canonical_json(claims)
        + "\n"
        + supervisor.MASTER_CLAIMS_END
        + "\n"
        + supervisor.MASTER_MATRIX_BEGIN
        + "\n"
        + supervisor._master_matrix_markdown(claims)
        + "\n"
        + supervisor.MASTER_MATRIX_END
        + "\n"
        + "".join(sections)
        + "Recommended next experiment: one only\n"
        + "# PARALLEL 4-GPU BATCH COMPLETE\n"
    )


def preflight(root: Path, run_id: str) -> dict:
    run_root = root / "runs" / run_id
    run_root.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "run_root": str(run_root.resolve()),
        "pod": {
            "id": supervisor.POD_ID,
            "name": supervisor.POD_NAME,
            "gpu_count": 4,
            "volume_id": supervisor.VOLUME_ID,
        },
        "checks": {name: True for name in supervisor.REQUIRED_CHECKS},
        "passed": True,
        "created_utc": "2026-08-27T00:00:00+00:00",
    }
    write_json(root / "MASTER_PREFLIGHT.json", payload)
    write_json(run_root / "MASTER_PREFLIGHT.json", payload)
    return payload


def recovery_fixture(root: Path, run_id: str, planned_commands: list[str]) -> tuple[SimpleNamespace, bytes]:
    preflight(root, run_id)
    run_root = (root / "runs" / run_id).resolve()
    lane_terminals = {}
    for index in range(4):
        lane = f"GPU{index}"
        if lane == "GPU0":
            returncode = 9
            status = "HARD_FAILURE"
            rationale = ["lane shell exited 9"]
        else:
            returncode = 0
            status = "SUCCESS"
            rationale = ["shell exited zero and exact science-complete marker passed"]
        lane_terminals[lane] = {
            "schema_version": 1,
            "run_id": run_id,
            "lane": lane,
            "returncode": returncode,
            "status": status,
            "rationale": rationale,
            "normalized_utc": "2026-08-27T00:00:00+00:00",
        }
        write_json(run_root / f"lane_gpu{index}.terminal.json", lane_terminals[lane])
    automatic = {
        "schema_version": 1,
        "run_id": run_id,
        "pod": supervisor.EXPECTED_POD,
        "all_four_lane_shells_exited": True,
        "all_lanes_terminal": True,
        "status": "HARD_FAILURE",
        "lanes": lane_terminals,
        "heartbeat_pid": os.getpid(),
        "heartbeat_process_group_id": os.getpgrp(),
        "heartbeat_left_running_for_finalization": True,
        "pod_stop_automated": False,
        "created_utc": "2026-08-27T00:00:00+00:00",
    }
    write_json(run_root / "MASTER_TERMINAL_STATUS.json", automatic)
    write_json(run_root / "MASTER_ALL_LANES_TERMINAL", automatic)
    write_json(root / "MASTER_TERMINAL_STATUS.json", automatic)
    write_json(root / "MASTER_ALL_LANES_TERMINAL", automatic)
    supervisor_state = {
        "schema_version": 1,
        "run_id": run_id,
        "run_root": str(run_root),
        "pod": supervisor.EXPECTED_POD,
        "supervisor_pid": 876543,
        "status": "HARD_FAILURE",
        "heartbeat": {
            "pid": os.getpid(),
            "process_group_id": os.getpgrp(),
            "status": "RUNNING_THROUGH_MASTER_FINALIZATION",
        },
        "lanes": {
            lane: {
                "status": row["status"],
                "pid": 800000 + index,
                "process_group_id": 800000 + index,
                "shell_exited": True,
                "returncode": row["returncode"],
            }
            for index, (lane, row) in enumerate(lane_terminals.items())
        },
    }
    write_json(run_root / "MASTER_SUPERVISOR.json", supervisor_state)
    write_json(root / "MASTER_STATUS.json", supervisor_state)
    original_master_bytes = (run_root / "MASTER_TERMINAL_STATUS.json").read_bytes()

    error_path = run_root / "lane_gpu0.error.json"
    write_json(
        error_path,
        {
            "run_id": run_id,
            "lane": "GPU0",
            "status": "HARD_FAILURE",
            "exit_code": 9,
        },
    )
    error_sha = hashlib.sha256(error_path.read_bytes()).hexdigest()
    recovery_reason = "audited deterministic test recovery"
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"strict frozen checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".pt.sha256").write_text(checkpoint_sha + "  checkpoint.pt\n")
    write_json(checkpoint.with_suffix(".pt.verification.json"), {"passed": True})
    write_json(
        run_root / "RECOVERY_PREFLIGHT.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "passed": True,
            "authorized_lanes": ["GPU0"],
            # A retained shell can use the current v2 evidence schema.  Only
            # legacy evidence is restricted to the explicitly retained GPU2.
            "retained_active_lanes": ["GPU0"],
            "recovery_evidence_schemas": {"GPU0": "v2_with_recovery_reason"},
            "recovery_reasons": {"GPU0": recovery_reason},
            "checks": {"all_exact": True},
            "lane_evidence": {
                "GPU0": {
                    "passed": True,
                    "prior_failure_marker_sha256": error_sha,
                    "base_checkpoint": str(checkpoint),
                    "base_checkpoint_sha256": checkpoint_sha,
                    "strict_checkpoint_reopen_passed": True,
                    "recovery_reason": recovery_reason,
                }
            },
        },
    )
    actual_commands = ["python first.py --exact", "python second.py --finalize"]
    shell_pid = 987654
    shell_pgid = 987654
    success = {
        "schema_version": 1,
        "run_id": run_id,
        "lane": "GPU0",
        "status": "RECOVERABLE_FAILURE_RESUMED",
        "phase": "SCIENCE_COMPLETE",
        "exit_code": 0,
        "shell_pid": shell_pid,
        "process_group_id": shell_pgid,
        "recovery_evidence": {
            "recovery_reason": recovery_reason,
            "prior_failure_marker_sha256": error_sha,
            "resume_checkpoint_sha256": checkpoint_sha,
            "resumed_command_records": actual_commands,
            "strict_checkpoint_reopen_passed": True,
            "recovery_preflight": str(run_root / "RECOVERY_PREFLIGHT.json"),
        },
    }
    write_json(run_root / "lane_gpu0.science_complete.json", success)
    write_json(
        run_root / "lane_gpu0.status.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "lane": "GPU0",
            "status": "RECOVERABLE_FAILURE_RESUMED",
            "phase": "SCIENCE_COMPLETE",
            "exit_code": 0,
            "shell_pid": shell_pid,
            "process_group_id": shell_pgid,
        },
    )
    (run_root / "lane_gpu0.recovery_commands.jsonl").write_text(
        "".join(json.dumps(command) + "\n" for command in actual_commands)
    )
    (run_root / "MASTER_COMMANDS.log").write_text(
        "".join(
            f"2026-08-27T00:00:0{number}Z run_id={run_id} lane=GPU0 "
            f"shell_pid={shell_pid} pgid={shell_pgid} phase=RECOVERY_{number} command={command}\n"
            for number, command in enumerate(actual_commands, 1)
        )
    )
    final_reports = {}
    final_audits = {}
    for experiment in supervisor.FINAL_REPORTS:
        report = root / f"{experiment}.final.md"
        report.write_text(f"final report for {experiment}\n")
        final_reports[experiment] = str(report)
        audit = root / f"{experiment}.FINAL_AUDIT.json"
        write_json(audit, {"passed": True})
        final_audits[experiment] = str(audit)
    final_checkpoints = {}
    for experiment in supervisor.FINAL_CHECKPOINTS:
        final = root / f"{experiment}.final.pt"
        final.write_bytes(f"verified final checkpoint {experiment}".encode())
        final_sha = hashlib.sha256(final.read_bytes()).hexdigest()
        final.with_suffix(".pt.sha256").write_text(
            f"{final_sha}  {final.name}\n"
        )
        write_json(final.with_suffix(".pt.verification.json"), {"passed": True})
        final_checkpoints[experiment] = str(final)
    plan_path = root / "recovery-plan.json"
    write_json(
        plan_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "recovered_lanes": {
                "GPU0": {
                    "recovery_reason": recovery_reason,
                    "recovery_evidence_schema": "v2_with_recovery_reason",
                    "expected_resumed_command_records": planned_commands,
                }
            },
        },
    )
    recovery_preflight_path = run_root / "RECOVERY_PREFLIGHT.json"
    recovery_preflight = json.loads(recovery_preflight_path.read_text())
    recovery_preflight["recovery_command_plan"] = {
        "path": str(plan_path.resolve()),
        "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "authorized_lanes": ["GPU0"],
    }
    write_json(recovery_preflight_path, recovery_preflight)
    args = SimpleNamespace(
        master_root=str(root),
        run_id=run_id,
        recovered_lane=["GPU0"],
        recovery_plan=str(plan_path),
        test_final_reports=final_reports,
        test_final_audits=final_audits,
        test_final_checkpoints=final_checkpoints,
    )
    return args, original_master_bytes


class OrchestrationTests(unittest.TestCase):
    def test_heartbeat_ignores_all_lanes_sentinel_until_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            preflight(root, run_id)
            run_root = root / "runs" / run_id
            write_json(run_root / "MASTER_ALL_LANES_TERMINAL", {"run_id": run_id})
            # The old two-field writable marker must never terminate monitoring.
            write_json(
                run_root / "MASTER_FINALIZATION_COMPLETE",
                {"run_id": run_id, "status": "MASTER_FINALIZATION_COMPLETE"},
            )
            write_json(
                root / "MASTER_FINALIZATION_COMPLETE",
                {"run_id": run_id, "status": "MASTER_FINALIZATION_COMPLETE"},
            )
            with self.assertRaises(RuntimeError):
                heartbeat.validate_finalization_sentinel(
                    root,
                    run_root,
                    run_id,
                    {"run_id": run_id, "monitor_status": "OBSERVING"},
                    30,
                )

    def test_exact_preflight_gate_rejects_stale_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = str(uuid.uuid4())
            preflight(root, current)
            observed, run_root = supervisor.validate_master_preflight(root, current)
            self.assertEqual(observed["run_id"], current)
            self.assertEqual(run_root, (root / "runs" / current).resolve())
            with self.assertRaisesRegex(RuntimeError, "stale"):
                supervisor.validate_master_preflight(root, str(uuid.uuid4()))

    def test_terminal_status_is_strict_and_recovery_needs_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            run_id = str(uuid.uuid4())
            lane = "GPU0"
            success = {
                "run_id": run_id,
                "lane": lane,
                "status": "SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER",
            }
            write_json(run_root / "lane_gpu0.science_complete.json", success)
            self.assertEqual(
                supervisor.normalize_terminal(run_root, run_id, lane, 0)["status"],
                "SUCCESS",
            )
            success["status"] = "RECOVERABLE_FAILURE_RESUMED"
            write_json(run_root / "lane_gpu0.science_complete.json", success)
            error_path = run_root / "lane_gpu0.error.json"
            write_json(
                error_path,
                {"run_id": run_id, "lane": lane, "status": "HARD_FAILURE", "exit_code": 9},
            )
            self.assertEqual(
                supervisor.normalize_terminal(run_root, run_id, lane, 0)["status"],
                "HARD_FAILURE",
            )
            success["recovery_evidence"] = {
                "prior_failure_marker_sha256": hashlib.sha256(error_path.read_bytes()).hexdigest(),
                "resume_checkpoint_sha256": "b" * 64,
                "resumed_command_records": ["record-1"],
                "strict_checkpoint_reopen_passed": True,
            }
            write_json(run_root / "lane_gpu0.science_complete.json", success)
            self.assertEqual(
                supervisor.normalize_terminal(run_root, run_id, lane, 0)["status"],
                "RECOVERABLE_FAILURE_RESUMED",
            )

    def test_dummy_four_lane_supervisor_leaves_heartbeat_until_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "master"
            scripts = Path(directory) / "scripts"
            scripts.mkdir(parents=True)
            run_id = str(uuid.uuid4())
            preflight(root, run_id)
            heartbeat_source = '''#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--master-root"); p.add_argument("--run-id"); p.add_argument("--interval-seconds"); a=p.parse_args()
s=Path(a.master_root)/"runs"/a.run_id/"MASTER_FINALIZATION_COMPLETE"
while True:
    if s.is_file() and json.loads(s.read_text()).get("run_id") == a.run_id:
        break
    time.sleep(0.02)
'''
            (scripts / "parallel_2d2_heartbeat.py").write_text(heartbeat_source)
            (scripts / "parallel_2d2_lane_common.sh").write_text("# dummy dependency\n")
            for index in range(4):
                lane = f"GPU{index}"
                source = f'''#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
from pathlib import Path
r=Path(os.environ["MASTER_ROOT"])/"runs"/os.environ["MASTER_RUN_ID"]
(r/"lane_gpu{index}.science_complete.json").write_text(json.dumps({{"run_id": os.environ["MASTER_RUN_ID"], "lane": "{lane}", "status": "SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER"}})+"\\n")
PY
'''
                (scripts / f"parallel_2d2_lane{index}.sh").write_text(source)
            args = SimpleNamespace(
                master_root=str(root),
                run_id=run_id,
                scripts_dir=str(scripts),
                heartbeat_interval_seconds=30,
                poll_seconds=0.01,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                self.assertEqual(supervisor.launch(args), 0)
            run_root = root / "runs" / run_id
            terminal = json.loads((run_root / "MASTER_ALL_LANES_TERMINAL").read_text())
            self.assertTrue(terminal["all_four_lane_shells_exited"])
            self.assertTrue(terminal["heartbeat_left_running_for_finalization"])
            heartbeat_pid = terminal["heartbeat_pid"]
            os.kill(heartbeat_pid, 0)
            finalize_args = SimpleNamespace(
                master_root=str(root),
                run_id=run_id,
                git_evidence="dummy-git",
                report_evidence="dummy-report",
                backup_evidence="dummy-backup",
            )
            with mock.patch.object(
                supervisor,
                "validate_finalization_boundary",
                return_value=passing_finalization_boundary(),
            ), mock.patch.object(
                supervisor,
                "validate_final_lightweight_process_gate",
                return_value={
                    "nvidia_compute_processes": [],
                    "scientific_cpu_processes": [],
                    "recorded_lane_processes": {"passed": True},
                    "all_gpus_compute_idle": True,
                    "passed": True,
                },
            ), mock.patch.object(
                supervisor,
                "capture_final_prepublication_fingerprint",
                return_value=passing_prepublication_fingerprint(),
            ):
                self.assertEqual(supervisor.mark_finalization_complete(finalize_args), 0)
            deadline = time.time() + 3
            while time.time() < deadline:
                waited, _ = os.waitpid(heartbeat_pid, os.WNOHANG)
                if waited == heartbeat_pid:
                    break
                time.sleep(0.02)
            else:
                os.kill(heartbeat_pid, 15)
                os.waitpid(heartbeat_pid, 0)
                self.fail("heartbeat did not exit after finalization sentinel")

    def test_recovery_reconciliation_preserves_failure_and_enables_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            commands = ["python first.py --exact", "python second.py --finalize"]
            args, original_master_bytes = recovery_fixture(root, run_id, commands)
            finalize_args = SimpleNamespace(
                master_root=str(root),
                run_id=run_id,
                git_evidence="pushed commit manifest",
                report_evidence="master report manifest",
                backup_evidence="local SHA manifest",
            )
            with self.assertRaisesRegex(RuntimeError, "not SUCCESS"):
                supervisor.mark_finalization_complete(finalize_args)

            original_process_is_alive = supervisor.process_is_alive
            original_group_is_alive = supervisor.process_group_is_alive
            original_reports = supervisor.FINAL_REPORTS
            original_audits = supervisor.FINAL_AUDITS
            original_checkpoints = supervisor.FINAL_CHECKPOINTS
            original_nvidia_compute_processes = supervisor.nvidia_compute_processes
            original_validate_final_checkpoint = supervisor.validate_final_checkpoint
            original_finalization_boundary = supervisor.validate_finalization_boundary
            original_final_process_gate = supervisor.validate_final_lightweight_process_gate
            original_final_fingerprint = supervisor.capture_final_prepublication_fingerprint
            supervisor.process_is_alive = lambda pid: pid == os.getpid()
            supervisor.process_group_is_alive = lambda pgid: pgid == os.getpgrp()
            supervisor.FINAL_REPORTS = args.test_final_reports
            supervisor.FINAL_AUDITS = args.test_final_audits
            supervisor.FINAL_CHECKPOINTS = args.test_final_checkpoints
            supervisor.nvidia_compute_processes = lambda: []
            supervisor.validate_final_checkpoint = lambda experiment, path, *unused: {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed": True,
            }
            supervisor.validate_finalization_boundary = (
                lambda *unused: passing_finalization_boundary()
            )
            supervisor.validate_final_lightweight_process_gate = lambda *unused: {
                "nvidia_compute_processes": [],
                "scientific_cpu_processes": [],
                "recorded_lane_processes": {"passed": True},
                "all_gpus_compute_idle": True,
                "passed": True,
            }
            supervisor.capture_final_prepublication_fingerprint = (
                lambda *unused: passing_prepublication_fingerprint()
            )
            try:
                self.assertEqual(supervisor.reconcile_recovery(args), 0)
                # Exact retry is safe even if a prior publication stopped after
                # the immutable reconciliation artifact was created.
                self.assertEqual(supervisor.reconcile_recovery(args), 0)
                run_root = (root / "runs" / run_id).resolve()
                preserved = run_root / "MASTER_TERMINAL_STATUS.original_supervisor.json"
                self.assertEqual(preserved.read_bytes(), original_master_bytes)
                self.assertEqual(json.loads(preserved.read_text())["status"], "HARD_FAILURE")
                canonical = json.loads((run_root / "MASTER_TERMINAL_STATUS.json").read_text())
                self.assertEqual(canonical["status"], "SUCCESS")
                self.assertTrue(canonical["recovery_reconciled"])
                self.assertEqual(canonical["recovered_lanes"], ["GPU0"])
                self.assertEqual(
                    canonical["lanes"]["GPU0"]["status"],
                    "RECOVERABLE_FAILURE_RESUMED",
                )
                standalone_path = run_root / "lane_gpu1.terminal.json"
                standalone_bytes = standalone_path.read_bytes()
                inconsistent = json.loads(standalone_bytes)
                inconsistent["returncode"] = 77
                write_json(standalone_path, inconsistent)
                with self.assertRaisesRegex(RuntimeError, "embedded and standalone"):
                    supervisor.mark_finalization_complete(finalize_args)
                standalone_path.write_bytes(standalone_bytes)
                supervisor.validate_final_lightweight_process_gate = lambda *unused: (_ for _ in ()).throw(
                    RuntimeError("scientific process appeared immediately before finalization publication")
                )
                with self.assertRaisesRegex(RuntimeError, "appeared immediately"):
                    supervisor.mark_finalization_complete(finalize_args)
                self.assertFalse((root / "MASTER_FINALIZATION_COMPLETE").exists())
                self.assertFalse((run_root / "MASTER_FINALIZATION_COMPLETE").exists())
                supervisor.validate_final_lightweight_process_gate = lambda *unused: {
                    "nvidia_compute_processes": [],
                    "scientific_cpu_processes": [],
                    "recorded_lane_processes": {"passed": True},
                    "all_gpus_compute_idle": True,
                    "passed": True,
                }
                original_exclusive_write = supervisor.durable_bytes_exclusive

                def fail_run_sentinel_once(path, content):
                    if path == run_root / "MASTER_FINALIZATION_COMPLETE":
                        raise RuntimeError("injected run-sentinel publication failure")
                    return original_exclusive_write(path, content)

                supervisor.durable_bytes_exclusive = fail_run_sentinel_once
                try:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        supervisor.mark_finalization_complete(finalize_args)
                finally:
                    supervisor.durable_bytes_exclusive = original_exclusive_write
                self.assertTrue((root / "MASTER_FINALIZATION_COMPLETE").exists())
                self.assertFalse((run_root / "MASTER_FINALIZATION_COMPLETE").exists())
                terminal_before_rejected_reconcile = (
                    run_root / "MASTER_TERMINAL_STATUS.json"
                ).read_bytes()
                with self.assertRaisesRegex(
                    RuntimeError,
                    "current-run master finalization publication has begun",
                ):
                    supervisor.reconcile_recovery(args)
                self.assertEqual(
                    (run_root / "MASTER_TERMINAL_STATUS.json").read_bytes(),
                    terminal_before_rejected_reconcile,
                )
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    supervisor.mark_finalization_complete(finalize_args)
                # A human operator may explicitly remove the fail-closed partial
                # publication in this isolated test fixture before a fresh retry.
                (root / "MASTER_FINALIZATION_COMPLETE").unlink()
                self.assertEqual(supervisor.mark_finalization_complete(finalize_args), 0)
                finalization = json.loads(
                    (run_root / "MASTER_FINALIZATION_COMPLETE").read_text()
                )
                self.assertTrue(finalization["recovery_reconciled"])
                self.assertEqual(finalization["recovered_lanes"], ["GPU0"])
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    supervisor.mark_finalization_complete(finalize_args)
            finally:
                supervisor.process_is_alive = original_process_is_alive
                supervisor.process_group_is_alive = original_group_is_alive
                supervisor.FINAL_REPORTS = original_reports
                supervisor.FINAL_AUDITS = original_audits
                supervisor.FINAL_CHECKPOINTS = original_checkpoints
                supervisor.nvidia_compute_processes = original_nvidia_compute_processes
                supervisor.validate_final_checkpoint = original_validate_final_checkpoint
                supervisor.validate_finalization_boundary = original_finalization_boundary
                supervisor.validate_final_lightweight_process_gate = original_final_process_gate
                supervisor.capture_final_prepublication_fingerprint = original_final_fingerprint

    def test_reconciliation_rejects_unplanned_commands_before_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            args, _ = recovery_fixture(root, run_id, ["python unexpected.py"])
            original_process_is_alive = supervisor.process_is_alive
            original_group_is_alive = supervisor.process_group_is_alive
            supervisor.process_is_alive = lambda pid: pid == os.getpid()
            supervisor.process_group_is_alive = lambda pgid: pgid == os.getpgrp()
            try:
                with self.assertRaisesRegex(RuntimeError, "independent expected command sequence"):
                    supervisor.reconcile_recovery(args)
            finally:
                supervisor.process_is_alive = original_process_is_alive
                supervisor.process_group_is_alive = original_group_is_alive
            run_root = root / "runs" / run_id
            self.assertFalse(
                (run_root / "MASTER_TERMINAL_STATUS.original_supervisor.json").exists()
            )
            self.assertFalse((run_root / "MASTER_RECOVERY_RECONCILIATION.json").exists())

    def test_reconciliation_requires_final_science_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            commands = ["python first.py --exact", "python second.py --finalize"]
            args, _ = recovery_fixture(root, run_id, commands)
            missing_report = Path(next(iter(args.test_final_reports.values())))
            missing_report.unlink()
            originals = (
                supervisor.process_is_alive,
                supervisor.process_group_is_alive,
                supervisor.FINAL_REPORTS,
                supervisor.FINAL_AUDITS,
                supervisor.FINAL_CHECKPOINTS,
                supervisor.nvidia_compute_processes,
                supervisor.validate_final_checkpoint,
            )
            supervisor.process_is_alive = lambda pid: pid == os.getpid()
            supervisor.process_group_is_alive = lambda pgid: pgid == os.getpgrp()
            supervisor.FINAL_REPORTS = args.test_final_reports
            supervisor.FINAL_AUDITS = args.test_final_audits
            supervisor.FINAL_CHECKPOINTS = args.test_final_checkpoints
            supervisor.nvidia_compute_processes = lambda: []
            supervisor.validate_final_checkpoint = lambda experiment, path, *unused: {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed": True,
            }
            try:
                with self.assertRaisesRegex(RuntimeError, "final report is unavailable"):
                    supervisor.reconcile_recovery(args)
                missing_report.write_text("restored exact final report\n")
                bad_audit = Path(next(iter(args.test_final_audits.values())))
                write_json(bad_audit, {"passed": False})
                with self.assertRaisesRegex(RuntimeError, "FINAL_AUDIT is not passing"):
                    supervisor.reconcile_recovery(args)
            finally:
                (
                    supervisor.process_is_alive,
                    supervisor.process_group_is_alive,
                    supervisor.FINAL_REPORTS,
                    supervisor.FINAL_AUDITS,
                    supervisor.FINAL_CHECKPOINTS,
                    supervisor.nvidia_compute_processes,
                    supervisor.validate_final_checkpoint,
                ) = originals
            run_root = root / "runs" / run_id
            self.assertFalse(
                (run_root / "MASTER_TERMINAL_STATUS.original_supervisor.json").exists()
            )

    def test_final_git_evidence_requires_live_origin_tracked_results_and_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master_root = root / "master"
            run_id = str(uuid.uuid4())
            run_root = master_root / "runs" / run_id
            worktree = master_root / "worktrees/master"
            bare = root / "origin.git"
            run_root.mkdir(parents=True)
            subprocess.check_call(
                ["git", "init", "--bare", str(bare)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["git", "init", "-b", "final-test", str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for key, value in (("user.name", "Test"), ("user.email", "test@example.com")):
                subprocess.check_call(["git", "config", key, value], cwd=worktree)
            (worktree / "MASTER_FINAL_REPORT.md").write_text("master report\n")
            result = worktree / "results/final"
            result.mkdir(parents=True)
            (result / "FINAL_AUDIT.json").write_text('{"passed": true}\n')
            write_json(
                result / "result_summary.json",
                {"git": {"implementation_commit": "PENDING"}},
            )
            write_json(
                result / "preflight_audit.json",
                {
                    "implementation_git_commit": "PENDING",
                    "implementation_fingerprint": {"script.py": "a" * 64},
                },
            )
            subprocess.check_call(["git", "add", "."], cwd=worktree)
            subprocess.check_call(
                ["git", "commit", "-m", "final results"],
                cwd=worktree,
                stdout=subprocess.DEVNULL,
            )
            canonical_origin = bare.resolve().as_uri()
            subprocess.check_call(
                ["git", "remote", "add", "origin", canonical_origin], cwd=worktree
            )
            subprocess.check_call(
                ["git", "push", "-u", "origin", "final-test"],
                cwd=worktree,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
            ).strip()
            for path in (result / "result_summary.json", result / "preflight_audit.json"):
                payload = json.loads(path.read_text())
                if path.name == "result_summary.json":
                    payload["git"]["implementation_commit"] = head
                else:
                    payload["implementation_git_commit"] = head
                write_json(path, payload)
            subprocess.check_call(["git", "add", "."], cwd=worktree)
            subprocess.check_call(
                ["git", "commit", "-m", "seal embedded implementation provenance"],
                cwd=worktree,
                stdout=subprocess.DEVNULL,
            )
            # The embedded commit is the exact implementation ancestor; the
            # second commit changes only authorized result evidence.
            sealed_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
            ).strip()
            subprocess.check_call(
                [
                    "git", "tag", "-a", "test-implementation-boundary",
                    "-m", "signed-boundary fixture", head,
                ],
                cwd=worktree,
            )
            tag_object = subprocess.check_output(
                ["git", "rev-parse", "refs/tags/test-implementation-boundary"],
                cwd=worktree,
                text=True,
            ).strip()
            subprocess.check_call(
                ["git", "push", "origin", "final-test"], cwd=worktree,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["git", "push", "origin", "refs/tags/test-implementation-boundary"],
                cwd=worktree, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            contract = {
                "2D2F": {
                    "worktree": "master",
                    "branch": "final-test",
                    "minimum_commit": head,
                    "implementation_tag": "test-implementation-boundary",
                    "implementation_commit": None,
                    "tracked_paths": ["results/final"],
                }
            }
            manifest = {
                **finalization_manifest_base(
                    "parallel_2d2_final_git_evidence_v1", run_id
                ),
                "repositories": {
                    "2D2F": {
                        "worktree": str(worktree.resolve()),
                        "branch": "final-test",
                        "origin_url": canonical_origin,
                        "implementation_tag": "test-implementation-boundary",
                        "implementation_tag_object": tag_object,
                        "implementation_commit": head,
                        "commit": sealed_head,
                        "origin_commit": sealed_head,
                        "tracked_paths": ["results/final"],
                    }
                },
            }
            manifest_path = run_root / supervisor.FINALIZATION_EVIDENCE_FILES["git"]
            write_json(manifest_path, manifest)
            manifest_path.chmod(0o444)
            signed_tag_audit = {
                "tag": "test-implementation-boundary",
                "tag_object": tag_object,
                "implementation_commit": head,
                "signer_principal": supervisor.FINAL_BACKUP_SIGNER_PRINCIPAL,
                "signer_fingerprint": supervisor.FINAL_BACKUP_SIGNER_FINGERPRINT,
                "signature_verified": True,
                "passed": True,
            }
            with mock.patch.object(
                supervisor, "FINAL_GIT_REPOSITORIES", contract
            ), mock.patch.object(
                supervisor, "CANONICAL_ORIGIN_URL", canonical_origin
            ), mock.patch.object(
                supervisor,
                "_verify_signed_implementation_tag",
                return_value=signed_tag_audit,
            ):
                audit = supervisor.validate_final_git_evidence(
                    master_root, run_root, run_id, str(manifest_path)
                )
                self.assertTrue(audit["passed"])
                (worktree / "untracked.txt").write_text("must block\n")
                with self.assertRaisesRegex(RuntimeError, "Git finalization evidence failed"):
                    supervisor.validate_final_git_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                (worktree / "untracked.txt").unlink()
                wrong = root / "wrong-origin.git"
                subprocess.check_call(
                    ["git", "init", "--bare", str(wrong)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.check_call(
                    ["git", "remote", "set-url", "origin", wrong.resolve().as_uri()],
                    cwd=worktree,
                )
                with self.assertRaisesRegex(RuntimeError, "Git finalization evidence failed"):
                    supervisor.validate_final_git_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                subprocess.check_call(
                    ["git", "remote", "set-url", "origin", canonical_origin], cwd=worktree
                )
                subprocess.check_call(
                    [
                        "git", "tag", "-a", "moved-boundary",
                        "-m", "unauthorized moved tag", sealed_head,
                    ],
                    cwd=worktree,
                )
                subprocess.check_call(
                    [
                        "git", "push", "--force", "origin",
                        "refs/tags/moved-boundary:refs/tags/test-implementation-boundary",
                    ],
                    cwd=worktree,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "remote_implementation_tag_matches"
                ):
                    supervisor.validate_final_git_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                subprocess.check_call(
                    [
                        "git", "push", "--force", "origin",
                        "refs/tags/test-implementation-boundary",
                    ],
                    cwd=worktree,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                bad = worktree / "scripts/unauthorized.py"
                bad.parent.mkdir()
                bad.write_text("raise SystemExit('unauthorized')\n")
                subprocess.check_call(["git", "add", str(bad)], cwd=worktree)
                subprocess.check_call(
                    ["git", "commit", "-m", "unauthorized code after implementation"],
                    cwd=worktree, stdout=subprocess.DEVNULL,
                )
                subprocess.check_call(
                    ["git", "push", "origin", "final-test"], cwd=worktree,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                new_head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
                ).strip()
                manifest["repositories"]["2D2F"]["commit"] = new_head
                manifest["repositories"]["2D2F"]["origin_commit"] = new_head
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "post_implementation_changes_authorized"):
                    supervisor.validate_final_git_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )

    def test_implementation_boundary_rejects_lightweight_and_unsigned_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory) / "repo"
            subprocess.check_call(
                ["git", "init", "-b", "main", str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for key, value in (("user.name", "Test"), ("user.email", "test@example.com")):
                subprocess.check_call(["git", "config", key, value], cwd=worktree)
            (worktree / "file.txt").write_text("implementation\n")
            subprocess.check_call(["git", "add", "file.txt"], cwd=worktree)
            subprocess.check_call(
                ["git", "commit", "-m", "implementation"],
                cwd=worktree,
                stdout=subprocess.DEVNULL,
            )
            subprocess.check_call(["git", "tag", "lightweight"], cwd=worktree)
            with self.assertRaisesRegex(RuntimeError, "not an annotated tag"):
                supervisor._verify_signed_implementation_tag(
                    worktree, "lightweight"
                )
            subprocess.check_call(
                ["git", "tag", "-a", "unsigned", "-m", "unsigned fixture"],
                cwd=worktree,
            )
            real_run = subprocess.run

            def selective_verifier(args, **kwargs):
                if "verify-tag" in args:
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        stdout=(
                            "Good git signature for rahul-local-backup with ED25519 key "
                            + supervisor.FINAL_BACKUP_SIGNER_FINGERPRINT
                        ),
                    )
                return real_run(args, **kwargs)

            with mock.patch.object(
                supervisor.subprocess, "run", side_effect=selective_verifier
            ) as verifier:
                synthetic = supervisor._verify_signed_implementation_tag(
                    worktree, "unsigned"
                )
            self.assertTrue(synthetic["signature_verified"])
            verify_args, verify_kwargs = next(
                call for call in verifier.call_args_list if "verify-tag" in call.args[0]
            )
            self.assertEqual(verify_args[0][0], supervisor.GIT_EXECUTABLE)
            self.assertEqual(verify_kwargs["env"], supervisor._clean_git_environment())
            self.assertIn("gpg.format=ssh", verify_args[0])
            self.assertIn(
                f"gpg.ssh.program={supervisor.SSH_KEYGEN_EXECUTABLE}",
                verify_args[0],
            )
            with self.assertRaisesRegex(RuntimeError, "signature is absent or untrusted"):
                supervisor._verify_signed_implementation_tag(worktree, "unsigned")

            fake_verifier = Path(directory) / "fake-ssh-keygen"
            fake_verifier.write_text(
                "#!/bin/sh\n"
                "echo 'Good git signature for rahul-local-backup with ED25519 key "
                + supervisor.FINAL_BACKUP_SIGNER_FINGERPRINT
                + "'\n"
                "exit 0\n"
            )
            fake_verifier.chmod(0o755)
            subprocess.check_call(
                ["git", "config", "--local", "gpg.ssh.program", str(fake_verifier)],
                cwd=worktree,
            )
            with self.assertRaisesRegex(
                RuntimeError, "repository-local GPG/SSH verifier configuration"
            ):
                supervisor._verify_signed_implementation_tag(worktree, "unsigned")

    def test_final_report_evidence_rehashes_master_reports_and_passing_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            run_root = master_root / "runs" / run_id
            run_root.mkdir(parents=True)
            coordinator = master_root / "MASTER_FINAL_REPORT.md"
            tracked = master_root / "worktrees/master/MASTER_FINAL_REPORT.md"
            report = master_root / "worktrees/lane/results/final/FINAL_REPORT.md"
            audit_path = report.with_name("FINAL_AUDIT.json")
            summary_path = report.with_name("result_summary.json")
            classification = "TEST CLASSIFICATION IS STRUCTURALLY BOUND"
            claims = synthetic_master_claims()
            master_text = synthetic_master_report(
                claims, f"Structured outcome: {classification}"
            )
            for path in (coordinator, tracked):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(master_text)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(f"sealed final report: {classification}\n")
            write_json(
                summary_path,
                {"experiment": "X", "primary_classification": classification},
            )
            write_json(audit_path, {"passed": True, "classification": classification})
            manifest = {
                **finalization_manifest_base(
                    "parallel_2d2_final_report_evidence_v1", run_id
                ),
                "master_report": file_evidence(coordinator),
                "tracked_master_report": file_evidence(tracked),
                "experiment_reports": {"X": file_evidence(report)},
                "result_summaries": {"X": file_evidence(summary_path)},
                "final_audits": {"X": file_evidence(audit_path)},
                "scientific_claims": claims,
            }
            manifest_path = run_root / supervisor.FINALIZATION_EVIDENCE_FILES["report"]
            write_json(manifest_path, manifest)
            manifest_path.chmod(0o444)
            with mock.patch.object(
                supervisor, "FINAL_REPORTS", {"X": str(report)}
            ), mock.patch.object(
                supervisor, "FINAL_AUDITS", {"X": str(audit_path)}
            ), mock.patch.object(
                supervisor, "derive_master_scientific_claims", return_value=claims
            ):
                result = supervisor.validate_final_report_evidence(
                    master_root, run_root, run_id, str(manifest_path)
                )
                self.assertTrue(result["passed"])
                forged_claims = json.loads(json.dumps(claims))
                forged_claims["matrix"]["2D2F"]["true_incremental"]["gain"] = 9.25
                forged_text = synthetic_master_report(
                    forged_claims, f"Structured outcome: {classification}"
                )
                for path in (coordinator, tracked):
                    path.write_text(forged_text)
                manifest["master_report"] = file_evidence(coordinator)
                manifest["tracked_master_report"] = file_evidence(tracked)
                manifest["scientific_claims"] = forged_claims
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "claims differ"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                false_answer = master_text.replace(
                    "ANSWER_CODE: YES_2D2F_IMPROVES_TRUE_B3_GAIN",
                    "ANSWER_CODE: NO_IMPROVEMENT",
                )
                for path in (coordinator, tracked):
                    path.write_text(false_answer)
                manifest["master_report"] = file_evidence(coordinator)
                manifest["tracked_master_report"] = file_evidence(tracked)
                manifest["scientific_claims"] = claims
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "M4 ANSWER_CODE differs"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                exact_table = supervisor._master_matrix_markdown(claims)
                forged_table = exact_table.replace("| 1000005 |", "| 9999999 |", 1)
                self.assertNotEqual(forged_table, exact_table)
                forged_matrix_report = master_text.replace(exact_table, forged_table)
                for path in (coordinator, tracked):
                    path.write_text(forged_matrix_report)
                manifest["master_report"] = file_evidence(coordinator)
                manifest["tracked_master_report"] = file_evidence(tracked)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "comparison matrix differs"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                for path in (coordinator, tracked):
                    path.write_text(master_text)
                manifest["master_report"] = file_evidence(coordinator)
                manifest["tracked_master_report"] = file_evidence(tracked)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                report.write_text("changed after evidence\n")
                with self.assertRaisesRegex(RuntimeError, "file identity differs"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                report.write_text(f"sealed final report: {classification}\n")
                manifest["experiment_reports"]["X"] = file_evidence(report)
                write_json(audit_path, {"passed": True, "classification": "FORGED"})
                manifest["final_audits"]["X"] = file_evidence(audit_path)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "classification differs"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )

                write_json(audit_path, {"passed": True, "classification": classification})
                broken_master = master_text.replace(
                    "M1. Did large frozen 2D2E confirmation reproduce positive B3 gain?",
                    "M1. TBD",
                )
                coordinator.write_text(broken_master)
                tracked.write_text(broken_master)
                manifest["master_report"] = file_evidence(coordinator)
                manifest["tracked_master_report"] = file_evidence(tracked)
                manifest["final_audits"]["X"] = file_evidence(audit_path)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "not substantive"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )

    def test_master_scientific_decisions_must_follow_sealed_numbers(self) -> None:
        answers = {
            name: {"answer_code": code, "evidence": {}}
            for name, code in supervisor.MASTER_SCIENTIFIC_ANSWER_CODES.items()
        }
        answers["M15"]["evidence"] = {
            "experiment_count": 1,
            "adaptation": "none",
            "candidates": ["2D2F", "2D2G"],
            "method": "larger_frozen_true_incremental_head_to_head",
        }

        def row(gain, gap, ce, state=None, rings=None):
            value = {
                "true_incremental": {
                    "gain": gain,
                    "sequence_gap": gap,
                    "real_validation_ce": ce,
                }
            }
            if state is not None:
                value["inference_state_bytes_b1"] = state
                value["recurrent_ring_count"] = rings
            return value

        matrix = {
            "2D2B": row(0.0036, 0.0025, 3.07),
            "2D2D": row(0.000005, -0.000066, 3.065),
            "2D2E-C1": row(0.000058, 0.000034, 3.06),
            "2D2E": row(0.000057, 0.000040, 3.06198, 33_289_728, 3),
            "2D2F": row(0.000130, 0.000090, 3.06196, 31_718_400, 2),
            "2D2G": row(0.000107, 0.000080, 3.05832, 34_765_824, 2),
            "2D2H": row(-0.000156, -0.000090, 3.07064, 33_096_192, 1),
            "2D2I": row(0.000032, 0.000020, 3.06261, 32_108_544, 4),
        }
        matrix["2D2E-C1"]["bootstrap_95_percent"] = {
            "off_minus_real": {"lower": -0.000019, "upper": 0.000134},
            "shuffled_minus_real": {"lower": -0.000040, "upper": 0.000108},
        }
        audit = supervisor._validate_master_scientific_decisions(matrix, answers)
        self.assertTrue(audit["passed"])
        matrix["2D2F"]["true_incremental"]["gain"] = -0.5
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            supervisor._validate_master_scientific_decisions(matrix, answers)

    def test_final_report_evidence_accepts_exact_2d2h_audit_without_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            run_root = master_root / "runs" / run_id
            run_root.mkdir(parents=True)
            coordinator = master_root / "MASTER_FINAL_REPORT.md"
            tracked = master_root / "worktrees/master/MASTER_FINAL_REPORT.md"
            report = master_root / "worktrees/2d2h/results/final/FINAL_REPORT.md"
            audit_path = report.with_name("FINAL_AUDIT.json")
            summary_path = report.with_name("result_summary.json")
            classification = "B2 W32 SECOND RECURRENT LINK IS HARMFUL"
            claims = synthetic_master_claims()
            master_text = synthetic_master_report(
                claims, f"Structured 2D2H outcome: {classification}"
            )
            for path in (coordinator, tracked):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(master_text)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(f"sealed final report: {classification}\n")
            write_json(
                summary_path,
                {"experiment": "2D2H", "primary_classification": classification},
            )
            real_schema_audit = {
                "artifact_inventory": {"passed": True},
                "checks": {"scientific integrity": True},
                "experiment": "2D2H",
                "passed": True,
            }
            write_json(audit_path, real_schema_audit)
            manifest = {
                **finalization_manifest_base(
                    "parallel_2d2_final_report_evidence_v1", run_id
                ),
                "master_report": file_evidence(coordinator),
                "tracked_master_report": file_evidence(tracked),
                "experiment_reports": {"2D2H": file_evidence(report)},
                "result_summaries": {"2D2H": file_evidence(summary_path)},
                "final_audits": {"2D2H": file_evidence(audit_path)},
                "scientific_claims": claims,
            }
            manifest_path = run_root / supervisor.FINALIZATION_EVIDENCE_FILES["report"]
            write_json(manifest_path, manifest)
            manifest_path.chmod(0o444)
            with mock.patch.object(
                supervisor, "FINAL_REPORTS", {"2D2H": str(report)}
            ), mock.patch.object(
                supervisor, "FINAL_AUDITS", {"2D2H": str(audit_path)}
            ), mock.patch.object(
                supervisor, "derive_master_scientific_claims", return_value=claims
            ):
                result = supervisor.validate_final_report_evidence(
                    master_root, run_root, run_id, str(manifest_path)
                )
                binding = result["structured_outcomes"]["2D2H"][
                    "audit_outcome_binding"
                ]
                self.assertFalse(binding["audit_classification_field_present"])
                self.assertTrue(binding["audit_experiment_field_verified"])

                forged = dict(real_schema_audit)
                forged["experiment"] = "2D2G"
                write_json(audit_path, forged)
                manifest["final_audits"]["2D2H"] = file_evidence(audit_path)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "published schema"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )

                forged = dict(real_schema_audit)
                forged["classification"] = classification
                write_json(audit_path, forged)
                manifest["final_audits"]["2D2H"] = file_evidence(audit_path)
                manifest_path.chmod(0o644)
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "published schema"):
                    supervisor.validate_final_report_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )

    def test_local_backup_evidence_covers_and_rehashes_every_critical_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master_root = root / "master"
            run_id = str(uuid.uuid4())
            run_root = master_root / "runs" / run_id
            run_root.mkdir(parents=True)
            coordinator = master_root / "MASTER_FINAL_REPORT.md"
            tracked = master_root / "worktrees/master/MASTER_FINAL_REPORT.md"
            report = master_root / "worktrees/lane/results/final/FINAL_REPORT.md"
            audit_path = report.with_name("FINAL_AUDIT.json")
            checkpoint = root / "persistent/final.pt"
            for path, content in (
                (coordinator, b"master\n"),
                (tracked, b"master\n"),
                (report, b"report\n"),
                (audit_path, b'{"passed": true}\n'),
                (checkpoint, b"checkpoint"),
                (checkpoint.with_suffix(".pt.sha256"), b"sidecar\n"),
                (checkpoint.with_suffix(".pt.verification.json"), b'{"passed": true}\n'),
                (run_root / supervisor.FINALIZATION_EVIDENCE_FILES["git"], b"git evidence\n"),
                (run_root / supervisor.FINALIZATION_EVIDENCE_FILES["report"], b"report evidence\n"),
                (run_root / "MASTER_RECOVERY_RECONCILIATION.json", b"reconciliation\n"),
                (master_root / "AUTO_STOP_PREFLIGHT.json", b"stop preflight\n"),
                (run_root / "AUTO_STOP_PREFLIGHT.json", b"stop preflight\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            backup_root = root / f"local_backup_{run_id}"
            backup_root.mkdir()
            with mock.patch.object(supervisor, "FINAL_REPORTS", {"X": str(report)}), mock.patch.object(
                supervisor, "FINAL_AUDITS", {"X": str(audit_path)}
            ), mock.patch.object(supervisor, "FINAL_CHECKPOINTS", {"X": str(checkpoint)}):
                sources = supervisor.required_final_backup_sources(master_root, run_root)
                rows = []
                for index, source in enumerate(sorted(sources)):
                    backup = backup_root / f"{index:04d}_{source.name}"
                    shutil.copyfile(source, backup)
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    rows.append(
                        {
                            "source_path": str(source),
                            "backup_path": str(backup),
                            "bytes": source.stat().st_size,
                            "sha256": digest,
                            "backup_bytes": backup.stat().st_size,
                            "backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
                        }
                    )
                manifest = {
                    **finalization_manifest_base(
                        "parallel_2d2_final_local_backup_evidence_v1", run_id
                    ),
                    "backup_root": str(backup_root),
                    "verification_host": {
                        "hostname": "test-mac",
                        "platform": "Darwin",
                        "verified_utc": "2026-08-27T00:00:00+00:00",
                    },
                    "all_backup_files_opened_and_hashed": True,
                    "inventory_sha256": supervisor._backup_inventory_sha256(rows),
                    "authenticated_pod_query": {
                        "run_id": run_id,
                        "command": f"runpodctl pod get {supervisor.POD_ID} -o json",
                        "authenticated": True,
                        "queried_utc": "2026-08-27T00:00:00+00:00",
                        "preflight_path": str((run_root / "AUTO_STOP_PREFLIGHT.json").resolve()),
                        "preflight_sha256": hashlib.sha256(b"stop preflight\n").hexdigest(),
                        "response": {},
                    },
                    "files": rows,
                }
                manifest_path = run_root / supervisor.FINALIZATION_EVIDENCE_FILES["backup"]
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                signature_path = run_root / supervisor.FINALIZATION_EVIDENCE_FILES[
                    "backup_signature"
                ]
                signature_path.write_text("test detached signature\n")
                signature_path.chmod(0o444)

                def accepted_signature(path, unused_run_root):
                    return {
                        "signed_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "passed": True,
                    }

                with mock.patch.object(
                    supervisor, "verify_backup_manifest_signature",
                    side_effect=accepted_signature,
                ):
                    result = supervisor.validate_final_backup_evidence(
                        master_root, run_root, run_id, str(manifest_path)
                    )
                    self.assertTrue(result["passed"])
                    self.assertTrue(result["all_backup_files_accessible_to_coordinator"])
                    Path(rows[0]["backup_path"]).write_text("tampered\n")
                    with self.assertRaisesRegex(RuntimeError, "accessible local backup differs"):
                        supervisor.validate_final_backup_evidence(
                            master_root, run_root, run_id, str(manifest_path)
                        )
                    shutil.copyfile(
                        Path(rows[0]["source_path"]), Path(rows[0]["backup_path"])
                    )
                    malformed = dict(manifest)
                    malformed["files"] = [None]
                    malformed["inventory_sha256"] = supervisor._backup_inventory_sha256(
                        malformed["files"]
                    )
                    manifest_path.chmod(0o644)
                    write_json(manifest_path, malformed)
                    manifest_path.chmod(0o444)
                    with self.assertRaisesRegex(RuntimeError, "row types are invalid"):
                        supervisor.validate_final_backup_evidence(
                            master_root, run_root, run_id, str(manifest_path)
                        )

    def test_backup_signature_uses_pinned_key_principal_namespace_and_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest = run_root / supervisor.FINALIZATION_EVIDENCE_FILES["backup"]
            signature = run_root / supervisor.FINALIZATION_EVIDENCE_FILES[
                "backup_signature"
            ]
            content = b'{"signed":"exact bytes"}\n'
            manifest.write_bytes(content)
            signature.write_text("detached signature fixture\n")
            signature.chmod(0o444)
            observed = {}

            def successful_verify(args, **kwargs):
                observed["args"] = args
                observed["input"] = kwargs["input"]
                observed["environment"] = kwargs["env"]
                allowed_index = args.index("-f") + 1
                observed["allowed"] = Path(args[allowed_index]).read_text()
                return subprocess.CompletedProcess(args, 0, stdout=b"Good signature\n")

            with mock.patch.object(supervisor.subprocess, "run", side_effect=successful_verify):
                audit = supervisor.verify_backup_manifest_signature(manifest, run_root)
            self.assertTrue(audit["passed"])
            self.assertEqual(
                observed["args"][0], supervisor.SSH_KEYGEN_EXECUTABLE
            )
            self.assertEqual(
                observed["environment"], supervisor.SANITIZED_TOOL_ENVIRONMENT
            )
            self.assertEqual(observed["input"], content)
            self.assertEqual(
                observed["allowed"],
                f"{supervisor.FINAL_BACKUP_SIGNER_PRINCIPAL} "
                f"{supervisor.FINAL_BACKUP_SIGNER_PUBLIC_KEY}\n",
            )
            self.assertIn(supervisor.FINAL_BACKUP_SIGNER_NAMESPACE, observed["args"])
            self.assertNotIn("@", observed["allowed"])
            with mock.patch.object(
                supervisor.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 255, stdout=b"bad signature"),
            ):
                with self.assertRaisesRegex(RuntimeError, "signature verification failed"):
                    supervisor.verify_backup_manifest_signature(manifest, run_root)
            signature.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "absent or mutable"):
                supervisor.verify_backup_manifest_signature(manifest, run_root)

    def test_final_checkpoint_verification_binds_identity_budget_and_every_strict_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for experiment in ("2D2F", "2D2H", "2D2I"):
                with self.subTest(experiment=experiment):
                    path = root / experiment / "scientific_update_0191.pt"
                    original = write_standard_final_checkpoint(path, experiment)
                    output = root / experiment / "tracked-results"
                    report = output / "FINAL_REPORT.md"
                    report.parent.mkdir(parents=True)
                    report.write_text("report\n")
                    summary = {
                        "final_checkpoint": str(path.resolve()),
                        "final_checkpoint_sha256": original["sha256"],
                    }
                    manifest = {
                        "smoke": {}, "scientific": {"191": original, "96": {}},
                    }
                    write_json(output / "result_summary.json", summary)
                    write_json(output / "checkpoint_manifest.json", manifest)
                    verification_path = path.with_suffix(
                        path.suffix + ".verification.json"
                    )
                    contract = supervisor.FINAL_CHECKPOINT_CONTRACTS[experiment]
                    with mock.patch.dict(
                        supervisor.FINAL_REPORTS, {experiment: str(report)}
                    ):
                        audit = supervisor.validate_final_checkpoint(experiment, path)
                        self.assertTrue(audit["passed"])
                        self.assertTrue(
                            audit["schema_audit"]["checkpoint_manifest"][
                                "scientific_191_exact"
                            ]
                        )
                        attacks = {
                            "wrong checkpoint path": {"checkpoint": "/tmp/copied.pt"},
                            "wrong sha": {"sha256": "0" * 64},
                            "wrong bytes": {"bytes": path.stat().st_size + 1},
                            "wrong update": {contract["updates_key"]: 96},
                            "wrong targets": {
                                contract["targets_key"]: contract["targets"] - 1
                            },
                        }
                        for label, change in attacks.items():
                            forged = json.loads(json.dumps(original))
                            forged.update(change)
                            write_json(verification_path, forged)
                            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                                supervisor.validate_final_checkpoint(experiment, path)
                        forged = json.loads(json.dumps(original))
                        strict_key = next(
                            key for key in contract["strict_keys"] if key != "passed"
                        )
                        forged["strict_reopen"][strict_key] = False
                        write_json(verification_path, forged)
                        with self.assertRaisesRegex(RuntimeError, "strict check map"):
                            supervisor.validate_final_checkpoint(experiment, path)
                        forged = json.loads(json.dumps(original))
                        forged["strict_reopen"].pop(strict_key)
                        write_json(verification_path, forged)
                        with self.assertRaisesRegex(RuntimeError, "strict check map"):
                            supervisor.validate_final_checkpoint(experiment, path)

                        write_json(verification_path, original)
                        for field, value in (
                            ("final_checkpoint", "/workspace/copied.pt"),
                            ("final_checkpoint_sha256", "0" * 64),
                        ):
                            forged_summary = dict(summary)
                            forged_summary[field] = value
                            write_json(output / "result_summary.json", forged_summary)
                            with self.assertRaisesRegex(RuntimeError, "result summary"):
                                supervisor.validate_final_checkpoint(experiment, path)
                        write_json(output / "result_summary.json", summary)

                        manifest_attacks = {
                            "checkpoint": "/workspace/copied.pt",
                            "sha256": "0" * 64,
                            "bytes": path.stat().st_size + 1,
                            "next_global_batch_sha256": "c" * 64,
                            "next_global_batch_stream_sha256": "d" * 64,
                        }
                        for field, value in manifest_attacks.items():
                            forged_manifest = json.loads(json.dumps(manifest))
                            forged_manifest["scientific"]["191"][field] = value
                            write_json(output / "checkpoint_manifest.json", forged_manifest)
                            with self.assertRaisesRegex(
                                RuntimeError, "scientific-191 manifest"
                            ):
                                supervisor.validate_final_checkpoint(experiment, path)

    def test_2d2g_checkpoint_uses_exact_strict_map_and_persistent_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/experiment_2d2g"
            report = output / "FINAL_REPORT.md"
            report.parent.mkdir(parents=True)
            report.write_text("report\n")
            path = root / "workspace/exp2d2g_run/checkpoints/stage_b_scientific_update_0191.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"2d2g persistent scientific checkpoint")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
            path.with_suffix(path.suffix + ".sha256").write_text(
                f"{digest}  {path.name}\n"
            )
            contract = supervisor.FINAL_CHECKPOINT_CONTRACTS["2D2G"]
            strict = {key: True for key in contract["strict_keys"]}
            write_json(path.with_suffix(path.suffix + ".verification.json"), strict)
            local = "/tmp/parallel_2d2_ephemeral/2d2g/stage_b_scientific_update_0191.pt"
            sidecar_checks = {
                "sha_sidecar_present": True, "verification_sidecar_present": True,
                "sha_sidecar_matches": True, "expected_sha_matches": True,
                "verification_passed": True,
            }

            def sidecar_audit(checkpoint):
                return {
                    "checkpoint": checkpoint, "sha256": digest,
                    "sha_sidecar": checkpoint + ".sha256",
                    "verification_sidecar": checkpoint + ".verification.json",
                    "checks": sidecar_checks, "passed": True,
                }

            persisted = {
                "local": local, "persistent": str(path.resolve()),
                "local_sha256": digest, "persistent_sha256": digest, "bytes": size,
                "lock": supervisor.CHECKPOINT_PERSIST_LOCK,
                "local_sidecar_audit": sidecar_audit(local),
                "persistent_sidecar_audit": sidecar_audit(str(path.resolve())),
                "persistent_sha_verified_while_lock_held": True,
                "reused_existing_exact_checkpoint": False, "passed": True,
                "path_audit": {
                    "local_checkpoint": local,
                    "persistent_directory": str(path.parent.resolve()),
                    "lock_path": supervisor.CHECKPOINT_PERSIST_LOCK,
                    "checks": {
                        "local_checkpoint_is_ephemeral": True,
                        "persistent_directory_is_workspace": True,
                        "persistent_directory_not_ephemeral": True,
                        "shared_lock_exact": True,
                    },
                    "passed": True,
                },
            }
            local_entry = {
                "checkpoint": local, "sha256": digest, "bytes": size,
                "next_global_batch_sha256": contract["next_batch"],
                "next_global_batch_stream_sha256": contract["next_stream"],
                "strict_reopen": strict,
            }
            write_json(
                output / "checkpoint_manifest.json",
                {"stage_a": {}, "stage_b": {"191": local_entry, "191_persistent": persisted}},
            )
            write_json(output / "persistent_final_checkpoint.json", persisted)
            write_json(
                output / "result_summary.json",
                {"final_checkpoint": str(path.resolve()), "final_checkpoint_sha256": digest},
            )
            targets_per_update = contract["targets"] // 191
            metrics = [
                {
                    "stage": "b", "local_update": update,
                    "processed_stage_targets": update * targets_per_update,
                }
                for update in range(1, 192)
            ]
            metrics_bytes = "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in metrics
            )
            (output / "stage_b_training_metrics.jsonl").write_text(metrics_bytes)
            (output / "training_metrics.jsonl").write_text(metrics_bytes)
            write_json(
                output / "stage_b_data_match.json",
                {
                    "reference": "2D2E", "pending_stage_a": False, "passed": True,
                    "update_191": {
                        "observed_next_global_batch_sha256": contract["next_batch"],
                        "observed_next_global_batch_stream_sha256": contract["next_stream"],
                        "expected": [contract["next_batch"], contract["next_stream"]],
                        "exact": True,
                    },
                },
            )
            with mock.patch.dict(supervisor.FINAL_REPORTS, {"2D2G": str(report)}):
                self.assertTrue(
                    supervisor.validate_final_checkpoint("2D2G", path)["passed"]
                )
                missing = dict(strict)
                missing.pop("matched_cursor")
                write_json(path.with_suffix(path.suffix + ".verification.json"), missing)
                with self.assertRaisesRegex(RuntimeError, "strict check map"):
                    supervisor.validate_final_checkpoint("2D2G", path)
                write_json(path.with_suffix(path.suffix + ".verification.json"), strict)
                forged_manifest = json.loads(
                    (output / "checkpoint_manifest.json").read_text()
                )
                forged_manifest["stage_b"]["191"]["sha256"] = "0" * 64
                write_json(output / "checkpoint_manifest.json", forged_manifest)
                with self.assertRaisesRegex(RuntimeError, "does not bind"):
                    supervisor.validate_final_checkpoint("2D2G", path)
                write_json(
                    output / "checkpoint_manifest.json",
                    {"stage_a": {}, "stage_b": {"191": local_entry, "191_persistent": persisted}},
                )
                forged_manifest = {
                    "stage_a": {},
                    "stage_b": {
                        "191": dict(local_entry), "191_persistent": persisted,
                    },
                }
                forged_manifest["stage_b"]["191"]["checkpoint"] = (
                    "/tmp/parallel_2d2_ephemeral/2d2g/copied.pt"
                )
                write_json(output / "checkpoint_manifest.json", forged_manifest)
                with self.assertRaisesRegex(RuntimeError, "does not bind"):
                    supervisor.validate_final_checkpoint("2D2G", path)
                base_manifest = {
                    "stage_a": {},
                    "stage_b": {"191": local_entry, "191_persistent": persisted},
                }
                write_json(output / "checkpoint_manifest.json", base_manifest)
                for audit_name, field in (
                    ("local_sidecar_audit", "sha_sidecar"),
                    ("local_sidecar_audit", "verification_sidecar"),
                    ("persistent_sidecar_audit", "sha_sidecar"),
                    ("persistent_sidecar_audit", "verification_sidecar"),
                ):
                    forged_persisted = json.loads(json.dumps(persisted))
                    forged_persisted[audit_name][field] = "/tmp/unbound.sidecar"
                    forged_manifest = json.loads(json.dumps(base_manifest))
                    forged_manifest["stage_b"]["191_persistent"] = forged_persisted
                    write_json(output / "checkpoint_manifest.json", forged_manifest)
                    write_json(
                        output / "persistent_final_checkpoint.json",
                        forged_persisted,
                    )
                    with self.assertRaisesRegex(RuntimeError, "schema/identity differs"):
                        supervisor.validate_final_checkpoint("2D2G", path)
                write_json(output / "checkpoint_manifest.json", base_manifest)
                write_json(output / "persistent_final_checkpoint.json", persisted)
                metrics[-1]["processed_stage_targets"] -= 1
                forged_metrics = "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in metrics
                )
                (output / "stage_b_training_metrics.jsonl").write_text(forged_metrics)
                (output / "training_metrics.jsonl").write_text(forged_metrics)
                with self.assertRaisesRegex(RuntimeError, "exact targets"):
                    supervisor.validate_final_checkpoint("2D2G", path)

    def test_finalization_boundary_rejects_scientific_cpu_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                supervisor, "validate_final_science_artifacts", return_value={"passed": True}
            ), mock.patch.object(
                supervisor,
                "running_scientific_processes",
                return_value=[{"pid": 9, "command": "python scripts/experiment_2d2g.py"}],
            ):
                with self.assertRaisesRegex(RuntimeError, "scientific processes remain"):
                    supervisor.validate_finalization_boundary(
                        root, root / "runs/x", "x", "git", "report", "backup",
                        {"created_utc": "2026-08-27T00:00:00+00:00"}, {},
                    )

    def test_process_audit_parses_exact_argv_and_rechecks_canonical_lane_pgids(self) -> None:
        ps_output = (
            "111 111 python python /workspace/scripts/experiment_2d2g.py finalize\n"
            "222 222 echo echo experiment_2d2g.py.backup\n"
        )
        with mock.patch.object(
            supervisor.subprocess, "check_output", return_value=ps_output
        ) as process_query:
            rows = supervisor.running_scientific_processes()
        self.assertEqual([row["pid"] for row in rows], [111])
        process_args, process_kwargs = process_query.call_args
        self.assertEqual(process_args[0][0], supervisor.PS_EXECUTABLE)
        self.assertEqual(
            process_kwargs["env"], supervisor.SANITIZED_TOOL_ENVIRONMENT
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            run_id = str(uuid.uuid4())
            terminal = {"run_id": run_id, "lanes": {}}
            state = {"lanes": {}}
            for index, (lane, script) in enumerate(supervisor.LANES.items()):
                pid = 1000 + index
                state["lanes"][lane] = {
                    "pid": pid, "process_group_id": pid,
                    "script": f"/workspace/scripts/{script}",
                }
                terminal["lanes"][lane] = {"status": "SUCCESS"}
            terminal["lanes"]["GPU1"] = {
                "status": "RECOVERABLE_FAILURE_RESUMED",
                "effective_shell_pid": 2001,
                "effective_shell_process_group_id": 2001,
            }
            archived = {
                "lane_gpu0.status.attempt2.json": ("GPU0", 9223),
                "lane_gpu0.status.recovery_attempt_0002.json": ("GPU0", 13921),
                "lane_gpu1.error.recovery_attempt_0001.json": ("GPU1", 13924),
                "lane_gpu1.status.recovery_attempt_0003.json": ("GPU1", 16149),
                "lane_gpu2.error.attempt2.json": ("GPU2", 9227),
                "lane_gpu2.status.recovery_attempt_0002.json": ("GPU2", 9729),
            }
            for name, (lane, pid) in archived.items():
                write_json(
                    run_root / name,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "lane": lane,
                        "status": "HARD_FAILURE",
                        "exit_code": 1,
                        "shell_pid": pid,
                        "process_group_id": pid,
                    },
                )
            with mock.patch.object(
                supervisor, "process_is_alive", return_value=False
            ), mock.patch.object(
                supervisor, "process_group_is_alive", return_value=False
            ):
                audit = supervisor.validate_recorded_lane_processes(
                    run_root, terminal, state
                )
                self.assertTrue(audit["passed"])
                observed = {
                    row["shell_pid"] for row in audit["archived_attempt_processes"]
                }
                self.assertEqual(
                    observed, {9223, 13921, 13924, 16149, 9227, 9729}
                )

            def one_group_alive(pgid):
                return pgid == 16149

            with mock.patch.object(
                supervisor, "process_is_alive", return_value=False
            ), mock.patch.object(
                supervisor, "process_group_is_alive", side_effect=one_group_alive
            ):
                with self.assertRaisesRegex(RuntimeError, "archived recovery PID/PGID"):
                    supervisor.validate_recorded_lane_processes(
                        run_root, terminal, state
                    )

    def test_security_critical_queries_use_fixed_executables_and_sanitized_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory)
            with mock.patch.object(
                supervisor.subprocess, "check_output", return_value="value\n"
            ) as git_query:
                self.assertEqual(supervisor._git_output(worktree, "version"), "value")
            args, kwargs = git_query.call_args
            self.assertEqual(args[0][0], supervisor.GIT_EXECUTABLE)
            self.assertEqual(kwargs["env"], supervisor._clean_git_environment())
            self.assertEqual(
                set(kwargs["env"]),
                {
                    "PATH", "LANG", "LC_ALL", "GIT_CONFIG_NOSYSTEM",
                    "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT",
                },
            )
        with mock.patch.object(
            supervisor.subprocess, "check_output", return_value=""
        ) as gpu_query:
            self.assertEqual(supervisor.nvidia_compute_processes(), [])
        args, kwargs = gpu_query.call_args
        self.assertEqual(args[0][0], supervisor.NVIDIA_SMI_EXECUTABLE)
        self.assertEqual(kwargs["env"], supervisor.SANITIZED_TOOL_ENVIRONMENT)

    def test_final_revalidation_rejects_git_report_or_checkpoint_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_root = master_root / "runs/run"
            run_root.mkdir(parents=True)
            terminal = {"created_utc": supervisor.now_utc(), "lanes": {}}
            state = {"lanes": {}}
            for path in (
                run_root / "MASTER_TERMINAL_STATUS.json",
                run_root / "MASTER_ALL_LANES_TERMINAL",
                master_root / "MASTER_TERMINAL_STATUS.json",
                master_root / "MASTER_ALL_LANES_TERMINAL",
            ):
                write_json(path, terminal)
            write_json(run_root / "MASTER_SUPERVISOR.json", state)
            write_json(master_root / "MASTER_STATUS.json", state)
            captured = passing_finalization_boundary()
            with mock.patch.object(
                supervisor, "validate_finalization_boundary", return_value=captured
            ):
                self.assertTrue(
                    supervisor.revalidate_finalization_snapshot(
                        master_root, run_root, "run", "git", "report", "backup",
                        terminal, state, captured,
                    )["passed"]
                )
            changed = json.loads(json.dumps(captured))
            changed["git"]["manifest"]["sha256"] = "f" * 64
            with mock.patch.object(
                supervisor, "validate_finalization_boundary", return_value=changed
            ):
                with self.assertRaisesRegex(RuntimeError, "state changed"):
                    supervisor.revalidate_finalization_snapshot(
                        master_root, run_root, "run", "git", "report", "backup",
                        terminal, state, captured,
                    )

    def test_finalization_lock_is_nonblocking_and_held_through_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            args = SimpleNamespace(master_root=str(master_root))
            lock_path = master_root / "locks/finalize.lock"
            subprocess_probe = '''
import fcntl
import sys

with open(sys.argv[1], "a+") as handle:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
raise SystemExit(9)
'''

            def publication_probe(unused_args):
                self.assertEqual(Path(unused_args.master_root), master_root)
                self.assertEqual(
                    subprocess.run(
                        [sys.executable, "-c", subprocess_probe, str(lock_path)],
                        check=False,
                        capture_output=True,
                        text=True,
                    ).returncode,
                    0,
                )
                return 73

            with mock.patch.object(
                supervisor,
                "_mark_finalization_complete_locked",
                side_effect=publication_probe,
            ) as publication:
                self.assertEqual(supervisor.mark_finalization_complete(args), 73)
                publication.assert_called_once_with(args)

            # The wrapper releases only after the complete publication routine returns.
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            with lock_path.open("a+") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(
                    supervisor, "_mark_finalization_complete_locked"
                ) as publication:
                    with self.assertRaisesRegex(
                        RuntimeError, "another finalizer holds the exact finalization lock"
                    ):
                        supervisor.mark_finalization_complete(args)
                    publication.assert_not_called()
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)
            self.assertFalse((master_root / "MASTER_FINALIZATION_COMPLETE").exists())

    def test_authenticated_stop_identity_is_exact_but_never_invoked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_root = master_root / "runs/run"
            run_id = str(uuid.uuid4())
            queried_utc = supervisor.now_utc()
            response = {
                "desiredStatus": "RUNNING",
                "gpuCount": 4,
                "id": supervisor.POD_ID,
                "name": supervisor.POD_NAME,
                "networkVolumeId": supervisor.VOLUME_ID,
                "runtimeStatus": "running",
            }
            payload = {
                "schema": "parallel_2d2_runpod_stop_capability_v1",
                "pod_id": supervisor.POD_ID,
                "pod_name": supervisor.POD_NAME,
                "gpu_count": 4,
                "volume_id": supervisor.VOLUME_ID,
                "authenticated": True,
                "authenticated_list_probe": True,
                "authenticated_pod_identity_response": response,
                "checked_utc": queried_utc,
                "desired_status": "RUNNING",
                "exact_stop_command": f"runpodctl pod stop {supervisor.POD_ID} -o json",
                "exact_stop_target": supervisor.POD_ID,
                "mechanism": "Authenticated runpodctl pod get/stop using local macOS Keychain credential",
                "network_volume_preservation_required": True,
                "persistent_volume_delete_authorized": False,
                "pod_delete_authorized": False,
                "pod_delete_forbidden": True,
                "runtime_status": "running",
                "secret_recorded": False,
                "stop_credential_available": True,
                "passed": True,
            }
            write_json(master_root / "AUTO_STOP_PREFLIGHT.json", payload)
            write_json(run_root / "AUTO_STOP_PREFLIGHT.json", payload)
            preflight_path = run_root / "AUTO_STOP_PREFLIGHT.json"
            query = {
                "run_id": run_id,
                "command": f"runpodctl pod get {supervisor.POD_ID} -o json",
                "authenticated": True,
                "queried_utc": queried_utc,
                "preflight_path": str(preflight_path.resolve()),
                "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
                "response": response,
            }
            backup = {
                "authenticated_pod_query": query,
                "detached_signature": {"passed": True},
            }
            not_before = supervisor.parse_canonical_utc(
                queried_utc, "test not-before"
            )
            result = supervisor.validate_authenticated_stop_identity(
                master_root, run_root, run_id, backup, not_before
            )
            self.assertTrue(result["passed"])
            query["unexpected"] = True
            with self.assertRaisesRegex(RuntimeError, "query key set is not exact"):
                supervisor.validate_authenticated_stop_identity(
                    master_root, run_root, run_id, backup, not_before
                )
            query.pop("unexpected")
            query["queried_utc"] = "2000-01-01T00:00:00Z"
            with self.assertRaisesRegex(RuntimeError, "stale or out of order"):
                supervisor.validate_authenticated_stop_identity(
                    master_root, run_root, run_id, backup, not_before
                )
            query["queried_utc"] = queried_utc
            payload["volume_id"] = "wrong-volume"
            write_json(master_root / "AUTO_STOP_PREFLIGHT.json", payload)
            write_json(run_root / "AUTO_STOP_PREFLIGHT.json", payload)
            with self.assertRaisesRegex(RuntimeError, "stop identity failed"):
                supervisor.validate_authenticated_stop_identity(
                    master_root, run_root, run_id, backup, not_before
                )
            source = Path(supervisor.__file__).read_text()
            function = source[
                source.index("def mark_finalization_complete") : source.index(
                    "def build_parser"
                )
            ]
            self.assertNotIn("runpodctl", function)
            self.assertNotIn("pod stop", function)

    def test_finalization_rejects_success_claim_with_unreconciled_lane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "master"
            run_id = str(uuid.uuid4())
            recovery_fixture(root, run_id, ["python first.py --exact"])
            run_root = (root / "runs" / run_id).resolve()
            forged = json.loads((run_root / "MASTER_TERMINAL_STATUS.json").read_text())
            forged["status"] = "SUCCESS"
            forged["lanes"]["GPU0"].update(
                {"status": "RECOVERABLE_FAILURE_RESUMED", "returncode": 0}
            )
            for path in (
                run_root / "MASTER_TERMINAL_STATUS.json",
                run_root / "MASTER_ALL_LANES_TERMINAL",
                root / "MASTER_TERMINAL_STATUS.json",
                root / "MASTER_ALL_LANES_TERMINAL",
            ):
                write_json(path, forged)
            state = json.loads((run_root / "MASTER_SUPERVISOR.json").read_text())
            state.update(
                {
                    "status": "ALL_LANES_TERMINAL_PENDING_MASTER_FINALIZATION",
                    "terminal_record": str(run_root / "MASTER_TERMINAL_STATUS.json"),
                    "all_lanes_terminal_sentinel": str(
                        run_root / "MASTER_ALL_LANES_TERMINAL"
                    ),
                }
            )
            write_json(run_root / "MASTER_SUPERVISOR.json", state)
            write_json(root / "MASTER_STATUS.json", state)
            finalize_args = SimpleNamespace(
                master_root=str(root),
                run_id=run_id,
                git_evidence="git",
                report_evidence="report",
                backup_evidence="backup",
            )
            with self.assertRaisesRegex(RuntimeError, "not been explicitly reconciled"):
                supervisor.mark_finalization_complete(finalize_args)
            self.assertFalse((run_root / "MASTER_FINALIZATION_COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
