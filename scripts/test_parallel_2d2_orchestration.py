#!/usr/bin/env python3
"""Focused CPU-only tests for run scoping and supervisor terminal semantics."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import time
import unittest
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parallel_2d2_supervisor as supervisor
import parallel_2d2_heartbeat as heartbeat


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


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
        final.with_suffix(".pt.sha256").write_text(final_sha + "\n")
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
            original_snapshot = heartbeat.snapshot
            original_sleep = heartbeat.time.sleep
            sleep_calls = []

            def fake_snapshot(*_args, **_kwargs):
                return {"run_id": run_id, "monitor_status": "OBSERVING"}

            def fake_sleep(_seconds):
                sleep_calls.append(True)
                write_json(
                    run_root / "MASTER_FINALIZATION_COMPLETE",
                    {"run_id": run_id, "status": "MASTER_FINALIZATION_COMPLETE"},
                )

            heartbeat.snapshot = fake_snapshot
            heartbeat.time.sleep = fake_sleep
            try:
                heartbeat.run(
                    SimpleNamespace(
                        master_root=str(root),
                        run_id=run_id,
                        interval_seconds=30,
                        stalled_seconds=60,
                    )
                )
            finally:
                heartbeat.snapshot = original_snapshot
                heartbeat.time.sleep = original_sleep
            self.assertEqual(len(sleep_calls), 1)
            final = json.loads((run_root / "MASTER_HEARTBEAT.json").read_text())
            self.assertEqual(
                final["monitor_status"],
                "MASTER_FINALIZATION_COMPLETE_SENTINEL_OBSERVED",
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
            supervisor.process_is_alive = lambda pid: pid == os.getpid()
            supervisor.process_group_is_alive = lambda pgid: pgid == os.getpgrp()
            supervisor.FINAL_REPORTS = args.test_final_reports
            supervisor.FINAL_AUDITS = args.test_final_audits
            supervisor.FINAL_CHECKPOINTS = args.test_final_checkpoints
            supervisor.nvidia_compute_processes = lambda: []
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
            )
            supervisor.process_is_alive = lambda pid: pid == os.getpid()
            supervisor.process_group_is_alive = lambda pgid: pgid == os.getpgrp()
            supervisor.FINAL_REPORTS = args.test_final_reports
            supervisor.FINAL_AUDITS = args.test_final_audits
            supervisor.FINAL_CHECKPOINTS = args.test_final_checkpoints
            supervisor.nvidia_compute_processes = lambda: []
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
                ) = originals
            run_root = root / "runs" / run_id
            self.assertFalse(
                (run_root / "MASTER_TERMINAL_STATUS.original_supervisor.json").exists()
            )

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
