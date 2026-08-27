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


if __name__ == "__main__":
    unittest.main()
