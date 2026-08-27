#!/usr/bin/env python3
"""CPU-only adversarial tests for the finalization Git trust boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parallel_2d2_supervisor as supervisor


class GitHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worktree = self.root / "repo"
        self.git("init", "-b", "main", str(self.worktree), cwd=self.root)
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.com")
        (self.worktree / "tracked.txt").write_text("first\n")
        self.git("add", "tracked.txt")
        self.git("commit", "-m", "first")
        self.first = self.git_output("rev-parse", "HEAD")
        (self.worktree / "tracked.txt").write_text("second\n")
        self.git("commit", "-am", "second")
        self.head = self.git_output("rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *args: str, cwd: Path | None = None) -> None:
        subprocess.check_call(
            [supervisor.GIT_EXECUTABLE, *args],
            cwd=cwd or self.worktree,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def git_output(self, *args: str) -> str:
        return subprocess.check_output(
            [supervisor.GIT_EXECUTABLE, *args],
            cwd=self.worktree,
            text=True,
        ).strip()

    def test_clean_raw_head_index_worktree_audit_passes(self) -> None:
        audit = supervisor._audit_git_repository(self.worktree, self.head)
        self.assertTrue(audit["head_index_worktree_exact"])
        self.assertTrue(audit["symlink_free"])
        self.assertEqual(audit["tracked_files"], 1)

    def test_local_fsmonitor_is_rejected_and_never_executed(self) -> None:
        marker = self.root / "fsmonitor-executed"
        hook = self.root / "fsmonitor"
        hook.write_text(
            "#!/bin/sh\n"
            f"touch {marker}\n"
            "printf 'new-token\\000'\n"
        )
        hook.chmod(0o755)
        self.git("config", "core.fsmonitor", str(hook))
        with self.assertRaisesRegex(RuntimeError, "unsafe repository-local Git controls"):
            supervisor._audit_git_repository(self.worktree, self.head)
        self.assertFalse(marker.exists())

    def test_assume_unchanged_cannot_hide_modified_tracked_bytes(self) -> None:
        self.git("update-index", "--assume-unchanged", "tracked.txt")
        (self.worktree / "tracked.txt").write_text("hidden assume-unchanged edit\n")
        with self.assertRaisesRegex(RuntimeError, "hidden state flags"):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_skip_worktree_cannot_hide_modified_tracked_bytes(self) -> None:
        self.git("update-index", "--skip-worktree", "tracked.txt")
        (self.worktree / "tracked.txt").write_text("hidden skip-worktree edit\n")
        with self.assertRaisesRegex(RuntimeError, "hidden state flags"):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_replace_ref_is_rejected_without_resolving_replacement(self) -> None:
        self.git("replace", self.head, self.first)
        # The safe invocation must still see the real HEAD object, then reject
        # the replacement namespace itself.
        self.assertEqual(
            supervisor._git_output(self.worktree, "rev-parse", "HEAD"), self.head
        )
        with self.assertRaisesRegex(RuntimeError, "replacements"):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_legacy_grafts_file_is_rejected(self) -> None:
        common = Path(self.git_output("rev-parse", "--git-common-dir"))
        if not common.is_absolute():
            common = self.worktree / common
        grafts = common / "info" / "grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(f"{self.head} {self.first}\n")
        with self.assertRaises(RuntimeError):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_raw_blob_audit_catches_plain_worktree_tampering(self) -> None:
        (self.worktree / "tracked.txt").write_text("ordinary hidden edit\n")
        with self.assertRaisesRegex(RuntimeError, "raw HEAD blob/mode"):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_tracked_symlink_is_never_followed(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("second\n")
        (self.worktree / "tracked.txt").unlink()
        (self.worktree / "tracked.txt").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "symlink-free tracked Git path"):
            supervisor._audit_git_repository(self.worktree, self.head)

    def test_shared_git_invocation_pins_security_controls(self) -> None:
        argv = supervisor._safe_git_argv("status")
        self.assertEqual(argv[0], supervisor.GIT_EXECUTABLE)
        self.assertIn("--no-replace-objects", argv)
        self.assertIn("core.fsmonitor=false", argv)
        self.assertIn("core.hooksPath=/dev/null", argv)
        self.assertIn("credential.helper=", argv)
        self.assertEqual(
            set(supervisor._clean_git_environment()),
            {
                "PATH", "LANG", "LC_ALL", "GIT_CONFIG_NOSYSTEM",
                "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT",
            },
        )

    def test_exact_execution_commit_and_blob_fingerprint_are_required(self) -> None:
        implementation = self.worktree / "impl.py"
        implementation.write_text("print('implementation')\n")
        self.git("add", "impl.py")
        self.git("commit", "-m", "execution implementation")
        execution = self.git_output("rev-parse", "HEAD")
        digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
        files = {"impl.py": digest}
        aggregate = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        fingerprint = {
            "algorithm": "sha256",
            "files": files,
            "aggregate_sha256": aggregate,
        }
        result = self.worktree / "results/final"
        result.mkdir(parents=True)
        (result / "result_summary.json").write_text("{}\n")
        (result / "preflight_audit.json").write_text(
            json.dumps(
                {
                    "implementation_git_commit": execution,
                    "environment": {"git_commit": execution},
                    "implementation_fingerprint": fingerprint,
                }
            )
            + "\n"
        )
        self.git("add", "results")
        self.git("commit", "-m", "result evidence")
        boundary = self.git_output("rev-parse", "HEAD")
        contract = {
            "tracked_paths": ["results/final"],
            "allowed_execution_commits": (execution,),
            "execution_commit_source": "preflight",
            "execution_environment_commit_required": True,
            "implementation_fingerprint_format": "canonical_file_map_sha256",
            "implementation_fingerprint_aggregate": aggregate,
            "implementation_fingerprint_files": ("impl.py",),
        }
        with mock.patch.dict(
            supervisor.FINAL_GIT_REPOSITORIES, {"2D2F": contract}, clear=True
        ):
            self.assertTrue(
                supervisor._validate_embedded_implementation(
                    "2D2F", self.worktree, boundary
                )["passed"]
            )
            forged = json.loads((result / "preflight_audit.json").read_text())
            forged["implementation_git_commit"] = self.first
            forged["environment"]["git_commit"] = self.first
            old_blob = supervisor._git_blob_at_commit(
                self.worktree, self.first, "tracked.txt"
            )
            old_digest = hashlib.sha256(old_blob).hexdigest()
            forged["implementation_fingerprint"] = {
                "algorithm": "sha256",
                "files": {"tracked.txt": old_digest},
                "aggregate_sha256": hashlib.sha256(
                    json.dumps(
                        {"tracked.txt": old_digest},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            (result / "preflight_audit.json").write_text(json.dumps(forged) + "\n")
            with self.assertRaisesRegex(RuntimeError, "explicitly allowed commit"):
                supervisor._validate_embedded_implementation(
                    "2D2F", self.worktree, boundary
                )


if __name__ == "__main__":
    unittest.main()
