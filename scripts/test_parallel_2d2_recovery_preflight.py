#!/usr/bin/env python3
"""CPU-only contract tests for the exact multi-lane recovery preflight."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import parallel_2d2_recovery_preflight as recovery


class RecoveryPreflightTests(unittest.TestCase):
    @staticmethod
    def write_original_terminal_fixture(master_root: Path, run_root: Path, heartbeat_pid=None) -> dict:
        lanes = {}
        for index in range(4):
            lane = f"GPU{index}"
            returncode = 0 if lane == "GPU3" else index + 7
            status = "SUCCESS" if lane == "GPU3" else "HARD_FAILURE"
            row = {
                "schema_version": 1,
                "run_id": run_root.name,
                "lane": lane,
                "returncode": returncode,
                "status": status,
                "rationale": ["fixture"],
            }
            lanes[lane] = row
            recovery.durable_json(run_root / f"lane_{lane.lower()}.terminal.json", row)
        payload = {
            "schema_version": 1,
            "run_id": run_root.name,
            "pod": recovery.EXPECTED_POD,
            "status": "HARD_FAILURE",
            "all_four_lane_shells_exited": True,
            "all_lanes_terminal": True,
            "heartbeat_pid": os.getpid() if heartbeat_pid is None else heartbeat_pid,
            "heartbeat_process_group_id": os.getpgrp(),
            "heartbeat_left_running_for_finalization": True,
            "lanes": lanes,
        }
        for path in (
            run_root / "MASTER_TERMINAL_STATUS.json",
            run_root / "MASTER_ALL_LANES_TERMINAL",
            master_root / "MASTER_TERMINAL_STATUS.json",
            master_root / "MASTER_ALL_LANES_TERMINAL",
        ):
            recovery.durable_json(path, payload)
        return payload

    def test_registered_lane_recovery_contracts_are_exact(self) -> None:
        self.assertEqual(set(recovery.LANES), {"GPU0", "GPU1", "GPU2"})
        gpu1 = recovery.LANES["GPU1"]
        self.assertEqual(
            gpu1["base_checkpoint"],
            "/tmp/parallel_2d2_ephemeral/2d2g/checkpoints/"
            "stage_a_scientific_update_0191.pt",
        )
        self.assertIsNone(gpu1["base_sha256"])
        self.assertEqual(
            gpu1["allowed_untracked_result_roots"],
            {"results/experiment_2d2g_b2_full_b3_w64"},
        )
        self.assertIn("recovery attempt 1 stopped before science", gpu1["recovery_reason"])
        gpu2 = recovery.LANES["GPU2"]
        self.assertEqual(
            gpu2["base_checkpoint"],
            "/workspace/exp2d2h_run/checkpoints/scientific_update_0191.pt",
        )
        self.assertIsNone(gpu2["base_sha256"])
        self.assertIn("default TF32", gpu2["recovery_reason"])
        gpu0 = recovery.LANES["GPU0"]
        self.assertIn("artifact publication failed", gpu0["recovery_reason"])
        self.assertIn("preflight-only 2D2F diagnostic", gpu0["recovery_reason"])
        self.assertEqual(len(gpu0["dependent_worktree_patches"]), 1)
        patch = gpu0["dependent_worktree_patches"][0]
        self.assertEqual(
            patch["branch"], "experiment-2d2f-no-b2-recurrence-b3-w64"
        )
        self.assertEqual(
            patch["allowed_changed_files"],
            {
                "scripts/experiment_2d2f.py",
                "tests/test_experiment_2d2f_core.py",
            },
        )
        self.assertEqual(
            patch["allowed_untracked_result_roots"],
            {"results/experiment_2d2f_no_b2_recurrence_b3_w64"},
        )

    def test_command_plan_is_independent_and_matches_bash_percent_q(self) -> None:
        root = Path("/workspace/parallel_2d2_master")
        run_root = root / "runs/00000000-0000-4000-8000-000000000001"
        schemas = {
            lane: "v2_with_recovery_reason" for lane in ("GPU0", "GPU1", "GPU2")
        }
        plan = recovery.recovery_command_plan(
            root,
            run_root,
            ["GPU0", "GPU1", "GPU2"],
            ["GPU0"],
            2,
            schemas,
        )
        rows = plan["recovered_lanes"]
        self.assertEqual(plan["recovery_attempt"], 2)
        self.assertTrue(
            all(row["recovery_evidence_schema"] == "v2_with_recovery_reason" for row in rows.values())
        )
        self.assertEqual(len(rows["GPU0"]["expected_resumed_command_records"]), 6)
        self.assertEqual(len(rows["GPU1"]["expected_resumed_command_records"]), 6)
        self.assertEqual(len(rows["GPU2"]["expected_resumed_command_records"]), 2)

        fresh_attempt3 = recovery.recovery_command_plan(
            root,
            run_root,
            ["GPU0", "GPU1", "GPU2"],
            [],
            3,
            schemas,
        )
        self.assertEqual(fresh_attempt3["recovery_attempt"], 3)
        self.assertEqual(
            set(fresh_attempt3["recovered_lanes"]), {"GPU0", "GPU1", "GPU2"}
        )
        self.assertTrue(
            all(
                row["recovery_evidence_schema"] == "v2_with_recovery_reason"
                for row in fresh_attempt3["recovered_lanes"].values()
            )
        )
        retained_attempt4 = recovery.recovery_command_plan(
            root,
            run_root,
            ["GPU2", "GPU0", "GPU1"],
            ["GPU0"],
            4,
            schemas,
        )
        self.assertEqual(retained_attempt4["recovery_attempt"], 4)
        self.assertEqual(
            list(retained_attempt4["recovered_lanes"]),
            ["GPU2", "GPU0", "GPU1"],
        )
        self.assertEqual(
            {
                lane: len(row["expected_resumed_command_records"])
                for lane, row in retained_attempt4["recovered_lanes"].items()
            },
            {"GPU2": 2, "GPU0": 6, "GPU1": 6},
        )

        gpu1 = rows["GPU1"]["expected_resumed_command_records"]
        self.assertIn(" preflight ", gpu1[0])
        self.assertIn("/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt", gpu1[0])
        self.assertIn(" smoke-b ", gpu1[1])
        self.assertIn("stage_a_scientific_update_0191.pt", gpu1[1])
        self.assertIn("/tmp/parallel_2d2_ephemeral/2d2g/smoke", gpu1[1])
        self.assertIn(" train-b ", gpu1[2])
        self.assertIn("--end-update 96", gpu1[2])
        self.assertIn("--resume", gpu1[3])
        self.assertIn(" persist-final ", gpu1[4])
        self.assertIn(" finalize ", gpu1[5])
        self.assertNotIn("train-a", "\n".join(gpu1))

        gpu2 = rows["GPU2"]["expected_resumed_command_records"]
        self.assertIn(" authorize-audit-correction ", gpu2[0])
        self.assertIn(" finalize ", gpu2[1])
        self.assertIn("--final-checkpoint", gpu2[0])
        self.assertIn("--final-checkpoint", gpu2[1])
        self.assertNotIn(" train ", "\n".join(gpu2))
        self.assertNotIn(" smoke ", "\n".join(gpu2))
        self.assertNotIn(" preflight ", "\n".join(gpu2))

        for lane in ("GPU0", "GPU1", "GPU2"):
            specs = recovery.expected_recovery_argv(root, run_root, lane)
            expected = rows[lane]["expected_resumed_command_records"]
            self.assertEqual(len(specs), len(expected))
            for argv, rendered in zip(specs, expected):
                script = (
                    'result=""; for word in "$@"; do printf -v quoted "%q" "$word"; '
                    'result+="${result:+ }${quoted}"; done; printf "%s" "$result"'
                )
                observed = subprocess.check_output(
                    ["bash", "-c", script, "--", *argv], text=True
                )
                self.assertEqual(rendered, observed)

    def test_tracked_cleanliness_and_exact_untracked_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=worktree,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Recovery Test"],
                cwd=worktree,
                check=True,
            )
            tracked = worktree / "tracked.txt"
            tracked.write_text("sealed\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=worktree, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=worktree, check=True
            )
            root = "results/experiment_exact"
            artifact = worktree / root / "result_summary.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}\n")

            accepted = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertTrue(accepted["passed"])
            self.assertEqual(accepted["untracked_files"], [f"{root}/result_summary.json"])

            outside = worktree / "results/experiment_exact_extra/forged.json"
            outside.parent.mkdir(parents=True)
            outside.write_text("{}\n")
            rejected = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertFalse(rejected["checks"]["untracked_only_in_exact_result_roots"])
            self.assertIn("results/experiment_exact_extra/forged.json", rejected["disallowed_untracked_files"])

            outside.unlink()
            tracked.write_text("modified\n")
            dirty = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertFalse(dirty["checks"]["tracked_worktree_clean"])
            self.assertTrue(dirty["checks"]["untracked_only_in_exact_result_roots"])

    def test_dependent_worktree_patch_requires_narrow_pushed_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin.git"
            worktree = root / "worktree"
            subprocess.run(["git", "init", "-q", "--bare", origin], check=True)
            subprocess.run(["git", "init", "-q", worktree], check=True)
            for key, value in (
                ("user.email", "test@example.invalid"),
                ("user.name", "Recovery Test"),
            ):
                subprocess.run(
                    ["git", "config", key, value], cwd=worktree, check=True
                )
            subprocess.run(
                ["git", "remote", "add", "origin", str(origin)],
                cwd=worktree,
                check=True,
            )
            (worktree / ".gitignore").write_text(
                ".pytest_cache/\n__pycache__/\n"
            )
            (worktree / "implementation.py").write_text("VALUE = 1\n")
            tests = worktree / "tests"
            tests.mkdir()
            (tests / "test_smoke.py").write_text(
                "def test_smoke():\n    assert True\n"
            )
            subprocess.run(["git", "add", "."], cwd=worktree, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "baseline"], cwd=worktree, check=True
            )
            branch = "experiment-recovery-fixture"
            subprocess.run(
                ["git", "branch", "-M", branch], cwd=worktree, check=True
            )
            subprocess.run(
                ["git", "push", "-qu", "origin", branch],
                cwd=worktree,
                check=True,
            )
            old = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
            ).strip()
            (worktree / "implementation.py").write_text("VALUE = 2\n")
            subprocess.run(
                ["git", "add", "implementation.py"], cwd=worktree, check=True
            )
            subprocess.run(
                ["git", "commit", "-qm", "narrow fix"], cwd=worktree, check=True
            )
            subprocess.run(
                ["git", "push", "-q", "origin", branch],
                cwd=worktree,
                check=True,
            )
            result = worktree / "results" / "experiment_exact" / "audit.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n")
            spec = {
                "branch": branch,
                "worktree": str(worktree),
                "allowed_changed_files": {"implementation.py"},
                "allowed_untracked_result_roots": {"results/experiment_exact"},
                "tests": ["tests/test_smoke.py"],
            }
            original_git = {"branches": {branch: {"local": old}}}
            original_run = subprocess.run

            def run_with_focused_test(command, *args, **kwargs):
                if command[1:3] == ["-m", "pytest"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout="1 passed", stderr=""
                    )
                return original_run(command, *args, **kwargs)

            with mock.patch.object(
                recovery.subprocess, "run", side_effect=run_with_focused_test
            ):
                audit = recovery.audit_patched_worktree(spec, original_git, 0)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["changed_files"], ["implementation.py"])

            (worktree / "implementation.py").write_text("VALUE = 3\n")
            with mock.patch.object(
                recovery.subprocess, "run", side_effect=run_with_focused_test
            ):
                dirty = recovery.audit_patched_worktree(spec, original_git, 0)
            self.assertFalse(dirty["checks"]["tracked_worktree_clean"])
            self.assertFalse(dirty["passed"])

    def test_focused_gpu1_tests_cannot_use_retained_gpu0(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="passed\n", stderr=""
        )
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            with mock.patch.object(
                recovery.subprocess, "run", return_value=completed
            ) as invoked:
                command, observed, visible = recovery.run_focused_tests(
                    Path("/workspace/parallel_2d2_master/worktrees/2d2g"),
                    ["tests/test_experiment_2d2g_driver.py"],
                    1,
                )
            self.assertIs(observed, completed)
            self.assertEqual(visible, "1")
            self.assertEqual(
                command[:4], [recovery.sys.executable, "-m", "pytest", "-q"]
            )
            self.assertEqual(
                invoked.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1"
            )
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        finally:
            if inherited is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = inherited

    def test_coordinator_worktree_is_reaudited_on_fresh_gpu1(self) -> None:
        sealed = {"passed": True, "current_commit": "pushed"}
        original_git = {"branches": {}}
        with mock.patch.object(
            recovery, "audit_patched_worktree", return_value=sealed
        ) as invoked:
            audit = recovery.audit_coordinator_worktree(original_git, 1)

        spec, observed_git, assigned_gpu = invoked.call_args.args
        master = recovery.LANES["GPU0"]
        self.assertIs(observed_git, original_git)
        self.assertEqual(assigned_gpu, 1)
        self.assertEqual(spec["branch"], master["branch"])
        self.assertEqual(spec["worktree"], master["worktree"])
        self.assertEqual(
            spec["allowed_changed_files"], master["allowed_changed_files"]
        )
        self.assertEqual(
            spec["allowed_untracked_result_roots"],
            master["allowed_untracked_result_roots"],
        )
        self.assertEqual(spec["tests"], recovery.COORDINATOR_FOCUSED_TESTS)
        self.assertEqual(audit["role"], "recovery_coordinator")
        self.assertEqual(audit["assigned_test_gpu_index"], 1)
        source = inspect.getsource(recovery.run)
        self.assertIn("coordinator_worktree_audit = audit_coordinator_worktree(", source)
        self.assertIn(
            '"recovery_coordinator_patch_narrow_pushed_clean_and_tested"',
            source,
        )
        self.assertIn('"coordinator_worktree_audit": coordinator_worktree_audit', source)

    def test_original_terminal_recovery_requires_exact_explicit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_root = master_root / "runs/22222222-2222-4222-8222-222222222222"
            run_root.mkdir(parents=True)
            self.write_original_terminal_fixture(master_root, run_root)

            unauthorized = recovery.audit_original_terminal_recovery(
                master_root, run_root, False
            )
            self.assertFalse(unauthorized["passed"])
            gate = recovery.audit_original_terminal_recovery(
                master_root, run_root, True
            )
            self.assertTrue(gate["passed"])
            self.assertTrue(
                recovery.terminal_is_sealed_for_lane(
                    gate, "GPU1", run_root / "lane_gpu1.terminal.json"
                )
            )
            self.assertEqual(
                gate["lanes"]["GPU3"]["status"], "SUCCESS"
            )
            self.assertEqual(gate["lanes"]["GPU3"]["returncode"], 0)

            changed_master = dict(self.write_original_terminal_fixture(master_root, run_root))
            changed_master["created_utc"] = "forged"
            recovery.durable_json(
                master_root / "MASTER_ALL_LANES_TERMINAL", changed_master
            )
            mismatched_master = recovery.audit_original_terminal_recovery(
                master_root, run_root, True
            )
            self.assertFalse(mismatched_master["passed"])
            self.assertFalse(
                mismatched_master["checks"]["all_master_terminal_bytes_exact"]
            )
            self.write_original_terminal_fixture(master_root, run_root)

            forged = json.loads((run_root / "lane_gpu1.terminal.json").read_text())
            forged["returncode"] = 0
            recovery.durable_json(run_root / "lane_gpu1.terminal.json", forged)
            mismatch = recovery.audit_original_terminal_recovery(
                master_root, run_root, True
            )
            self.assertFalse(mismatch["passed"])
            self.assertFalse(mismatch["lanes"]["GPU1"]["passed"])

    def test_original_terminal_gate_requires_live_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_root = master_root / "runs/33333333-3333-4333-8333-333333333333"
            run_root.mkdir(parents=True)
            self.write_original_terminal_fixture(
                master_root, run_root, heartbeat_pid=999_999_999
            )
            gate = recovery.audit_original_terminal_recovery(
                master_root, run_root, True
            )
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["checks"]["heartbeat_pid_alive"])

    def test_checkpoint_requires_exact_colocated_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage_a_scientific_update_0191.pt"
            checkpoint.write_bytes(b"exact stage A 191")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.with_suffix(".pt.sha256").write_text(
                f"{digest}  {checkpoint.name}\n"
            )
            checkpoint.with_suffix(".pt.verification.json").write_text(
                json.dumps({"passed": True}) + "\n"
            )
            self.assertTrue(recovery.audit_checkpoint_sidecars(checkpoint, None)["passed"])
            checkpoint.with_suffix(".pt.sha256").write_text(f"{digest}  wrong.pt\n")
            self.assertFalse(recovery.audit_checkpoint_sidecars(checkpoint, None)["passed"])

    def test_gpu0_failed_science_archive_moves_exact_c1_and_partial_f(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master_root = Path(directory) / "master"
            run_root = master_root / "runs/44444444-4444-4444-8444-444444444444"
            run_root.mkdir(parents=True)
            c1 = (
                master_root
                / "worktrees/master/results/experiment_2d2e_c1_large_true_self_confirmation"
            )
            partial_f = (
                master_root
                / "worktrees/2d2f/results/experiment_2d2f_no_b2_recurrence_b3_w64"
            )
            c1.mkdir(parents=True)
            partial_f.mkdir(parents=True)
            for name in recovery.GPU0_COMPLETED_C1_FILES:
                payload = (
                    {"passed": True, "classification": "DIRECTIONAL CONFIRMATION"}
                    if name == "FINAL_AUDIT.json"
                    else {"fixture": name}
                )
                (c1 / name).write_text(json.dumps(payload) + "\n")
            for name in recovery.GPU0_FAILED_F_PREFLIGHT_FILES:
                payload = (
                    {"passed": True}
                    if name == "semantic_diff_audit.json"
                    else {"fixture": name}
                )
                (partial_f / name).write_text(json.dumps(payload) + "\n")

            plan = {
                "schema_version": 1,
                "run_id": run_root.name,
                "recovery_attempt": 2,
                "recovered_lanes": {
                    "GPU0": {
                        "expected_resumed_command_records": ["c1", "f-preflight"]
                    }
                },
            }
            recovery.preserve_exact_json(
                recovery.versioned_plan_path(run_root, 2), plan
            )
            (run_root / "lane_gpu0.recovery_commands.jsonl").write_text(
                json.dumps("c1") + "\n"
            )
            failure = {
                "run_id": run_root.name,
                "lane": "GPU0",
                "status": "HARD_FAILURE",
                "phase": "2D2F_PREFLIGHT",
                "exit_code": 1,
                "command": "f-preflight",
            }
            recovery.durable_json(run_root / "lane_gpu0.error.json", failure)
            recovery.durable_json(run_root / "lane_gpu0.status.json", failure)
            (run_root / "lane_gpu0.log").write_text("failed preflight\n")
            (run_root / "lane_gpu0.recovery.console.log").write_text(
                "Traceback\nKeyError: 'local_attention_weights'\n"
            )
            prior_manifest = run_root / "prior.json"
            recovery.durable_json(prior_manifest, {"passed": True})
            prior = {
                "failed_recovery_attempt": 2,
                "manifest_path": str(prior_manifest),
                "manifest_sha256": recovery.file_sha256(prior_manifest),
            }

            archived = recovery.archive_gpu0_attempt2_science(
                master_root, run_root, 3, prior
            )
            self.assertTrue(archived["passed"])
            self.assertFalse(c1.exists())
            self.assertFalse(partial_f.exists())
            self.assertTrue(
                Path(archived["completed_c1"]["archive_path"]).is_dir()
            )
            self.assertTrue(
                Path(archived["partial_2d2f_preflight"]["archive_path"]).is_dir()
            )
            repeated = recovery.archive_gpu0_attempt2_science(
                master_root, run_root, 3, prior
            )
            self.assertEqual(
                repeated["manifest_sha256"], archived["manifest_sha256"]
            )

    def test_gpu1_attempt3_smoke_archive_is_exact_idempotent_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master_root = root / "master"
            run_root = master_root / "runs/55555555-5555-4555-8555-555555555555"
            run_root.mkdir(parents=True)
            output = (
                master_root
                / "worktrees/2d2g/results/experiment_2d2g_b2_full_b3_w64"
            )
            output.mkdir(parents=True)
            for name in recovery.GPU1_ATTEMPT3_REQUIRED_OUTPUT_FILES:
                path = output / name
                if name == "storage_cleanup_manifest.json":
                    payload = {
                        "scientific_source_removed": False,
                        "cleanup_actions": [],
                    }
                    path.write_text(json.dumps(payload) + "\n")
                elif path.suffix == ".jsonl":
                    path.write_text(json.dumps({"fixture": name}) + "\n")
                else:
                    path.write_text(json.dumps({"fixture": name}) + "\n")

            provenance_source = root / "pre_attempt3_provenance"
            provenance_source.mkdir()
            for name in recovery.GPU1_RETAINED_STAGE_A_REQUIRED_FILES:
                path = provenance_source / name
                path.write_text(
                    json.dumps({"fixture": name}) + "\n"
                    if path.suffix == ".json"
                    else json.dumps({"fixture": name}) + "\n"
                )

            expected = [
                "g-preflight",
                "g-smoke",
                "g-train-96",
                "g-train-191",
                "g-persist",
                "g-finalize",
            ]
            recovery.preserve_exact_json(
                recovery.versioned_plan_path(run_root, 3),
                {
                    "schema_version": 1,
                    "run_id": run_root.name,
                    "recovery_attempt": 3,
                    "recovered_lanes": {
                        "GPU1": {
                            "expected_resumed_command_records": expected
                        }
                    },
                },
            )
            (run_root / "lane_gpu1.recovery_commands.jsonl").write_text(
                json.dumps(expected[0]) + "\n"
            )
            failure = {
                "run_id": run_root.name,
                "lane": "GPU1",
                "status": "HARD_FAILURE",
                "phase": "2D2G_B_RECOVERY_SMOKE",
                "exit_code": 1,
                "command": expected[1],
            }
            recovery.durable_json(run_root / "lane_gpu1.error.json", failure)
            recovery.durable_json(run_root / "lane_gpu1.status.json", failure)
            (run_root / "lane_gpu1.log").write_text("failed smoke\n")
            (run_root / "lane_gpu1.recovery_attempt3.console.log").write_text(
                "Traceback: torch.equal compared cpu and cuda tensors\n"
            )
            prior_manifest = run_root / "attempt3_prior_evidence.json"
            recovery.durable_json(prior_manifest, {"passed": True})
            prior = {
                "failed_recovery_attempt": 3,
                "manifest_path": str(prior_manifest),
                "manifest_sha256": recovery.file_sha256(prior_manifest),
            }

            smoke_root = root / "smoke"
            smoke_root.mkdir()
            checkpoint = (
                smoke_root
                / "stage_b_disposable_smoke_update_0003_pid_12345.pt"
            )
            checkpoint.write_bytes(b"verified disposable smoke checkpoint")
            checkpoint_sha = recovery.file_sha256(checkpoint)
            checkpoint.with_suffix(".pt.sha256").write_text(
                f"{checkpoint_sha}  {checkpoint.name}\n"
            )
            recovery.durable_json(
                checkpoint.with_suffix(".pt.verification.json"),
                {"passed": True},
            )
            passing_provenance = {
                "passed": True,
                "checks": {"fixture": True},
            }
            patches = (
                mock.patch.object(
                    recovery,
                    "gpu1_retained_provenance_source",
                    return_value=provenance_source,
                ),
                mock.patch.object(
                    recovery,
                    "gpu1_disposable_smoke_root",
                    return_value=smoke_root,
                ),
                mock.patch.object(
                    recovery,
                    "validate_retained_gpu1_stage_a_provenance",
                    return_value=passing_provenance,
                ),
            )
            with patches[0], patches[1], patches[2]:
                archived = recovery.archive_gpu1_attempt3_smoke(
                    master_root, run_root, 4, prior
                )
                repeated = recovery.archive_gpu1_attempt3_smoke(
                    master_root, run_root, 4, prior
                )
                self.assertEqual(
                    repeated["manifest_sha256"], archived["manifest_sha256"]
                )
                self.assertTrue(archived["passed"])
                self.assertFalse(any(smoke_root.iterdir()))
                cleanup = json.loads(
                    (output / "storage_cleanup_manifest.json").read_text()
                )
                self.assertEqual(len(cleanup["cleanup_actions"]), 3)
                self.assertTrue(
                    all(row["removed"] for row in cleanup["cleanup_actions"])
                )
                snapshot = Path(archived["attempt3_output_snapshot"]["path"])
                self.assertNotEqual(snapshot, output)
                self.assertTrue((snapshot / "checkpoint_manifest.json").is_file())

                checkpoint.write_bytes(b"ambiguous replacement")
                with self.assertRaisesRegex(
                    RuntimeError, "refusing changed disposable"
                ):
                    recovery.archive_gpu1_attempt3_smoke(
                        master_root, run_root, 4, prior
                    )

    def test_gpu1_original_runtime_is_bound_to_master_log_and_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "55555555-5555-4555-8555-555555555555"
            provenance = root / "provenance"
            run_root.mkdir()
            provenance.mkdir()
            runtime_commands = [
                {
                    "command": "scripts/experiment_2d2g.py preflight --output-dir /workspace/out",
                    "kind": "preflight",
                },
                {
                    "command": "scripts/experiment_2d2g.py train-a --end-update 96",
                    "stage": "a",
                    "start_update": 1,
                    "end_update": 96,
                    "pid": 101,
                    "wall_seconds": 12.5,
                },
                {
                    "command": "scripts/experiment_2d2g.py train-a --resume /tmp/a96.pt --end-update 191",
                    "stage": "a",
                    "start_update": 97,
                    "end_update": 191,
                    "pid": 202,
                    "wall_seconds": 11.25,
                },
            ]
            runtime_path = provenance / "commands_and_runtime.json"
            recovery.durable_json(runtime_path, {"commands": runtime_commands})
            recovery.durable_json(
                provenance / "stage_a_restart_required_update_96.json",
                {"saved_process_id": 101},
            )
            recovery.durable_json(
                provenance / "stage_a_forced_restart_update_96.json",
                {
                    "checkpoint_process_id": 101,
                    "resumed_process_id": 202,
                    "fresh_process": True,
                    "passed": True,
                },
            )
            recovery.durable_json(
                provenance / "HEARTBEAT.json",
                {"stage": "a", "local_update": 191, "pid": 202},
            )
            master_rows = []
            for phase, runtime_row in zip(
                recovery.GPU1_ORIGINAL_STAGE_A_PHASES, runtime_commands
            ):
                master_rows.append(
                    "2026-08-25T00:00:00Z "
                    f"run_id={run_root.name} lane=GPU1 shell_pid=303 pgid=303 "
                    f"phase={phase} command=python {runtime_row['command']}"
                )
            (run_root / "MASTER_COMMANDS.log").write_text(
                "\n".join(master_rows) + "\n"
            )

            audit = recovery.validate_gpu1_original_runtime_lineage(
                provenance, run_root
            )
            self.assertTrue(audit["passed"], audit["checks"])
            self.assertEqual(
                [row["phase"] for row in audit["command_links"]],
                list(recovery.GPU1_ORIGINAL_STAGE_A_PHASES),
            )

            runtime_commands[1]["wall_seconds"] = -1.0
            recovery.durable_json(runtime_path, {"commands": runtime_commands})
            failed = recovery.validate_gpu1_original_runtime_lineage(
                provenance, run_root
            )
            self.assertFalse(failed["passed"])
            self.assertFalse(
                failed["checks"]["training_wall_seconds_finite_positive"]
            )

            runtime_commands[1]["wall_seconds"] = 12.5
            runtime_commands[2]["command"] += " --tampered"
            recovery.durable_json(runtime_path, {"commands": runtime_commands})
            failed = recovery.validate_gpu1_original_runtime_lineage(
                provenance, run_root
            )
            self.assertFalse(failed["passed"])
            self.assertFalse(
                failed["checks"]["runtime_commands_bind_exact_master_rows"]
            )

    def test_lane1_shell_is_stage_b_only_and_marks_complete(self) -> None:
        path = Path(__file__).with_name("parallel_2d2_lane1_stage_b_recovery.sh")
        text = path.read_text()
        self.assertIn("source \"$MASTER_ROOT/worktrees/master/scripts/parallel_2d2_lane_common.sh\"", text)
        self.assertIn("export MASTER_RECOVERY_MODE=1", text)
        self.assertEqual(text.count("log_command "), 6)
        self.assertNotIn("train-a", text)
        self.assertIn("2D2G_RECOVERY_PREFLIGHT", text)
        self.assertIn("--recovery-provenance-dir", text)
        self.assertIn("retained_science_provenance", text)
        self.assertLess(text.index(" preflight "), text.index("smoke-b"))
        self.assertLess(text.index("smoke-b"), text.index("--end-update 96"))
        self.assertLess(text.index("--end-update 96"), text.index("--end-update 191"))
        self.assertLess(text.index("persist-final"), text.index(" finalize "))
        self.assertTrue(text.rstrip().endswith("lane_mark_science_complete"))

    def test_lane2_shell_is_authorize_then_finalize_only(self) -> None:
        path = Path(__file__).with_name("parallel_2d2_lane2_finalize_recovery.sh")
        text = path.read_text()
        self.assertIn("export MASTER_RECOVERY_MODE=1", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", text)
        self.assertEqual(text.count("log_command "), 2)
        self.assertLess(
            text.index("experiment_2d2h.py authorize-audit-correction"),
            text.index("experiment_2d2h.py finalize"),
        )
        self.assertEqual(text.count('--final-checkpoint "$FINAL_CHECKPOINT"'), 2)
        self.assertNotIn("experiment_2d2h.py preflight", text)
        self.assertNotIn(" train ", text)
        self.assertNotIn(" smoke ", text)
        self.assertIn("RUN_ROOT=/workspace/exp2d2h_run", text)
        self.assertIn('FINAL_CHECKPOINT="$RUN_ROOT/checkpoints/scientific_update_0191.pt"', text)
        self.assertTrue(text.rstrip().endswith("lane_mark_science_complete"))

    def test_lane_common_accepts_only_sealed_original_terminal(self) -> None:
        path = Path(__file__).with_name("parallel_2d2_lane_common.sh")
        text = path.read_text()
        lane_init_start = text.index("lane_init() {")
        embedded_gate_start = text.index(
            '  "$LANE_PYTHON_BIN" - <<\'PY\'', lane_init_start
        )
        embedded_gate_end = text.index("\nPY\n", embedded_gate_start)
        embedded_gate = text[embedded_gate_start:embedded_gate_end]
        self.assertIn("import hashlib", embedded_gate)
        self.assertLess(
            embedded_gate.index("import hashlib"),
            embedded_gate.index("hashlib.sha256"),
        )
        self.assertIn('recovery.get("original_terminal_recovery_gate", {})', text)
        self.assertIn('sealed.get("sha256") != terminal_sha', text)
        self.assertIn('terminal.get("status") != "HARD_FAILURE"', text)
        self.assertIn("recovery refuses an existing science-complete marker", text)
        self.assertNotIn("success or terminal marker", text)

    def test_completed_lane_retention_uses_full_supervisor_evidence(self) -> None:
        source = inspect.getsource(recovery.run)
        self.assertIn("args.retain_completed_lane", source)
        self.assertIn("supervisor.validate_recovery_lane(", source)
        self.assertIn("supervisor.parse_master_command_log(", source)
        self.assertIn('"retained_completed_lanes": retained_completed_lanes', source)
        self.assertIn(
            '"retained_completed_recovery_lanes_exact"', source
        )
        self.assertIn("terminal_is_sealed_for_lane(", source)
        self.assertIn("idle = gpu_idle(", source)

    def test_recovery_plan_is_preserved_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RECOVERY_COMMAND_PLAN.json"
            plan = {"schema_version": 1, "run_id": "exact", "recovered_lanes": {}}
            recovery.preserve_exact_json(path, plan)
            original = path.read_bytes()
            recovery.preserve_exact_json(path, plan)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                recovery.preserve_exact_json(path, {**plan, "run_id": "changed"})
            self.assertEqual(path.read_bytes(), original)

    def test_attempt2_preserves_and_validates_attempt1_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "11111111-1111-4111-8111-111111111111"
            run_root.mkdir()
            plan_path = recovery.versioned_plan_path(run_root, 1)
            plan = {
                "schema_version": 1,
                "run_id": run_root.name,
                "recovered_lanes": {
                    lane: {
                        "expected_resumed_command_records": [f"python {lane}.py"],
                        "recovery_reason": "attempt one",
                        "recovery_evidence_schema": "v2_with_recovery_reason",
                    }
                    for lane in ("GPU0", "GPU1", "GPU2")
                },
            }
            recovery.preserve_exact_json(plan_path, plan)
            plan_sha = recovery.file_sha256(plan_path)
            preflight = {
                "schema_version": 1,
                "run_id": run_root.name,
                "passed": True,
                "authorized_lanes": ["GPU0", "GPU1", "GPU2"],
                "recovery_command_plan": {
                    "path": str(plan_path),
                    "sha256": plan_sha,
                    "authorized_lanes": ["GPU0", "GPU1", "GPU2"],
                },
            }
            recovery.durable_json(run_root / "RECOVERY_PREFLIGHT.json", preflight)
            original_preflight = (run_root / "RECOVERY_PREFLIGHT.json").read_bytes()
            for lane in ("GPU1", "GPU2"):
                lower = lane.lower()
                recovery.durable_json(
                    run_root / f"lane_{lower}.error.json",
                    {
                        "run_id": run_root.name,
                        "lane": lane,
                        "status": "HARD_FAILURE",
                        "exit_code": 17,
                    },
                )
                recovery.durable_json(
                    run_root / f"lane_{lower}.status.json",
                    {
                        "run_id": run_root.name,
                        "lane": lane,
                        "status": "HARD_FAILURE",
                        "exit_code": 17,
                    },
                )
                (run_root / f"lane_{lower}.recovery_commands.jsonl").write_text(
                    json.dumps(f"python {lane}.py") + "\n"
                )

            evidence = recovery.prepare_prior_attempt_evidence(
                run_root, ["GPU1", "GPU2"], 2
            )
            self.assertEqual(evidence["failed_recovery_attempt"], 1)
            self.assertEqual(evidence["prior_command_plan_sha256"], plan_sha)
            archived = recovery.versioned_preflight_path(run_root, 1)
            self.assertEqual(archived.read_bytes(), original_preflight)
            manifest = json.loads(Path(evidence["manifest_path"]).read_text())
            self.assertEqual(manifest["retried_lanes"], ["GPU1", "GPU2"])
            self.assertEqual(manifest["next_recovery_attempt"], 2)
            for lane in ("GPU1", "GPU2"):
                for row in manifest["lane_failure_evidence"][lane].values():
                    self.assertEqual(
                        recovery.file_sha256(Path(row["preserved_path"])),
                        row["sha256"],
                    )

            # Re-auditing identical bytes is idempotent, but changed failed-attempt
            # evidence can never replace the sealed archive.
            self.assertEqual(
                recovery.prepare_prior_attempt_evidence(
                    run_root, ["GPU1", "GPU2"], 2
                )["manifest_sha256"],
                evidence["manifest_sha256"],
            )

            lane = "GPU0"
            lower = lane.lower()
            for kind in ("error", "status"):
                recovery.durable_json(
                    run_root / f"lane_{lower}.{kind}.json",
                    {
                        "run_id": run_root.name,
                        "lane": lane,
                        "status": "HARD_FAILURE",
                        "exit_code": 19,
                    },
                )
            (run_root / f"lane_{lower}.recovery_commands.jsonl").write_text(
                json.dumps(f"python {lane}.py") + "\n"
            )
            expanded = recovery.prepare_prior_attempt_evidence(
                run_root, ["GPU0", "GPU1", "GPU2"], 2
            )
            expanded_manifest = json.loads(
                Path(expanded["manifest_path"]).read_text()
            )
            self.assertEqual(
                expanded_manifest["retried_lanes"], ["GPU0", "GPU1", "GPU2"]
            )
            self.assertEqual(len(expanded_manifest["components"]), 2)
            self.assertEqual(
                json.loads(Path(evidence["manifest_path"]).read_text())[
                    "retried_lanes"
                ],
                ["GPU1", "GPU2"],
            )

            (run_root / "lane_gpu1.status.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "prior recovery outcome"):
                recovery.prepare_prior_attempt_evidence(
                    run_root, ["GPU1", "GPU2"], 2
                )


if __name__ == "__main__":
    unittest.main()
