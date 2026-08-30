#!/usr/bin/env python3
"""Fail-closed, exact-target RunPod guard for Experiment 2D5C.

The guard is intentionally local to the independent Mac control environment.
It obtains the existing account credential from macOS Keychain, passes it only
through a runpodctl child-process environment, and emits only allowlisted JSON.

Production stop authorization is two-part:

1. ``preflight`` creates a private, expiring authorization artifact bound to
   the current exact pod instance and retained network volume.
2. ``stop`` requires that artifact plus a matching terminal trigger artifact.

``watchdog`` can either wait for that exact trigger path or supervise a whole
experiment command.  A supervised command creates the terminal trigger and
attempts the guarded stop only after successful scientific completion.  A
non-zero child exit is preserved for diagnosis while the allocated pod remains
running; an explicit terminal trigger can still request a stop after the user
has reviewed a recoverable failure.  The watchdog must wrap the *entire*
experiment/finalization workflow, not one training segment or an ordinary SSH
connectivity probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EXPERIMENT = "2D5C"
POD_ID = "rvgztsr0azrwyo"
POD_NAME = "happy_apricot_stork"
GPU_COUNT = 1
NETWORK_VOLUME_ID = "yhzyb27fb5"
NETWORK_VOLUME_NAME = "unlikely_lime_flamingo"
NETWORK_VOLUME_SIZE_GB = 150
NETWORK_VOLUME_DATACENTER = "US-MD-1"
VOLUME_MOUNT_PATH = "/workspace"

RUNPODCTL_PATH = Path("/opt/homebrew/bin/runpodctl")
SECURITY_PATH = Path("/usr/bin/security")
KEYCHAIN_SERVICE = "runpod-codex-pod-stopper"
KEYCHAIN_ACCOUNT = "rahul"

ACTION = "stop_exact_pod_after_terminal_2d5c"
EXACT_STOP_ARGV = ("runpodctl", "pod", "stop", POD_ID, "-o", "json")
EXACT_STOP_COMMAND = " ".join(EXACT_STOP_ARGV)

AUTHORIZATION_SCHEMA = "experiment_2d5c_runpod_guard_authorization_v1"
TRIGGER_SCHEMA = "experiment_2d5c_runpod_guard_trigger_v1"
REPORT_SCHEMA = "experiment_2d5c_runpod_guard_report_v1"
DEFAULT_AUTHORIZATION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_STOP_TIMEOUT_SECONDS = 240.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_WATCH_TIMEOUT_SECONDS = DEFAULT_AUTHORIZATION_SECONDS
MAX_ARTIFACT_BYTES = 32 * 1024
MAX_CLOCK_SKEW_SECONDS = 300


AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "experiment",
        "action",
        "pod_id",
        "pod_name",
        "gpu_count",
        "network_volume_id",
        "network_volume_name",
        "network_volume_size_gb",
        "network_volume_datacenter",
        "volume_mount_path",
        "pod_created_at",
        "pod_running_last_status_change",
        "identity_sha256",
        "exact_stop_command",
        "credential_source",
        "issued_at_utc",
        "expires_at_utc",
        "authorization_nonce",
    }
)

TRIGGER_KEYS = frozenset(
    {
        "schema",
        "experiment",
        "action",
        "pod_id",
        "pod_name",
        "gpu_count",
        "network_volume_id",
        "authorization_sha256",
        "authorization_nonce",
        "terminal_outcome",
        "exit_code",
        "source",
        "created_at_utc",
    }
)


class GuardError(RuntimeError):
    """A safe, user-displayable fail-closed error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise GuardError("invalid_artifact", f"{label} is not a UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise GuardError("invalid_artifact", f"{label} is not canonical UTC") from error
    return parsed


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_absolute_artifact_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise GuardError("unsafe_path", f"{label} must be an absolute path")
    if path.name in {"", ".", ".."}:
        raise GuardError("unsafe_path", f"{label} has no exact filename")
    return path


def write_private_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path = _require_absolute_artifact_path(path, "artifact path")
    parent = path.parent
    if not parent.is_dir():
        raise GuardError("unsafe_path", "artifact parent directory does not exist")
    data = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise GuardError("artifact_exists", "refusing to overwrite artifact") from error
    except OSError as error:
        raise GuardError("artifact_write_failed", "cannot create private artifact") from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise GuardError("artifact_write_failed", "artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    if mode != 0o600:
        raise GuardError("unsafe_permissions", "artifact permissions are not 0600")
    try:
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # The artifact itself is already fsynced. Some filesystems reject a
        # directory fsync; this does not weaken the no-overwrite contract.
        pass
    return sha256_bytes(data)


def read_private_canonical_json(path: Path, expected_keys: frozenset[str]) -> tuple[dict, bytes]:
    path = _require_absolute_artifact_path(path, "artifact path")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise GuardError("artifact_missing", "required artifact is unavailable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise GuardError("unsafe_artifact", "artifact is not a regular non-symlink file")
    if before.st_uid != os.getuid():
        raise GuardError("unsafe_artifact", "artifact is not owned by the current user")
    if stat.S_IMODE(before.st_mode) & 0o077:
        raise GuardError("unsafe_permissions", "artifact must not be group/world accessible")
    if before.st_size <= 0 or before.st_size > MAX_ARTIFACT_BYTES:
        raise GuardError("invalid_artifact", "artifact size is outside the safe bound")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise GuardError("artifact_changed", "artifact identity changed while opening")
            chunks = []
            remaining = MAX_ARTIFACT_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(descriptor)
    except GuardError:
        raise
    except OSError as error:
        raise GuardError("artifact_read_failed", "cannot read artifact") from error
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuardError("invalid_artifact", "artifact is not canonical JSON") from error
    if not isinstance(payload, dict) or set(payload) != set(expected_keys):
        raise GuardError("invalid_artifact", "artifact key set is not exact")
    if canonical_json_bytes(payload) != data:
        raise GuardError("invalid_artifact", "artifact serialization is not canonical")
    return payload, data


class KeychainCredentialProvider:
    """Read the existing credential without writing it to process-global state."""

    def __init__(
        self,
        security_path: Path = SECURITY_PATH,
        service: str = KEYCHAIN_SERVICE,
        account: str = KEYCHAIN_ACCOUNT,
    ) -> None:
        self.security_path = security_path
        self.service = service
        self.account = account

    def read(self) -> bytes:
        try:
            completed = subprocess.run(
                [
                    str(self.security_path),
                    "find-generic-password",
                    "-s",
                    self.service,
                    "-a",
                    self.account,
                    "-w",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GuardError(
                "credential_unavailable", "Keychain credential lookup failed"
            ) from error
        credential = completed.stdout.rstrip(b"\r\n")
        if completed.returncode != 0 or not credential or len(credential) > 1024:
            raise GuardError("credential_unavailable", "Keychain credential is unavailable")
        return credential


class RunPodClient:
    """Minimal allowlisted RunPod client; all responses are parsed in memory."""

    def __init__(
        self,
        runpodctl_path: Path = RUNPODCTL_PATH,
        credential_provider: Any | None = None,
    ) -> None:
        self.runpodctl_path = runpodctl_path
        self.credential_provider = credential_provider or KeychainCredentialProvider()

    def _call(self, arguments: Sequence[str], operation: str) -> Any:
        credential = self.credential_provider.read()
        if hasattr(os, "environb"):
            child_environment: dict[Any, Any] = dict(os.environb)
            child_environment[b"RUNPOD_API_KEY"] = credential
        else:  # pragma: no cover - macOS and Linux expose environb
            child_environment = dict(os.environ)
            child_environment["RUNPOD_API_KEY"] = credential.decode("utf-8")
        argv = [str(self.runpodctl_path), *arguments]
        try:
            completed = subprocess.run(
                argv,
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=45,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GuardError("runpodctl_failed", f"{operation} could not be executed") from error
        finally:
            child_environment.pop(b"RUNPOD_API_KEY", None)
            child_environment.pop("RUNPOD_API_KEY", None)
            credential = b""
        raw = completed.stdout if completed.stdout.strip() else completed.stderr
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GuardError(
                "runpodctl_invalid_json",
                f"{operation} returned no parseable JSON (rc={completed.returncode})",
            ) from error
        if completed.returncode != 0:
            code = payload.get("code") if isinstance(payload, dict) else None
            safe_suffix = f" ({code})" if isinstance(code, str) and len(code) <= 64 else ""
            raise GuardError("runpodctl_failed", f"{operation} failed{safe_suffix}")
        return payload

    def pod_get(self) -> dict:
        payload = self._call(
            ["pod", "get", POD_ID, "--include-network-volume", "-o", "json"],
            "exact pod status query",
        )
        if not isinstance(payload, dict):
            raise GuardError("runpodctl_shape", "exact pod query did not return an object")
        return payload

    def pod_list_all(self) -> list[dict]:
        payload = self._call(
            ["pod", "list", "--all", "-o", "json"], "account pod list query"
        )
        rows = (
            payload
            if isinstance(payload, list)
            else payload.get("pods")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise GuardError("runpodctl_shape", "pod list did not return a list of objects")
        return rows

    def network_volume_get(self) -> dict:
        payload = self._call(
            ["network-volume", "get", NETWORK_VOLUME_ID, "-o", "json"],
            "retained network-volume query",
        )
        if not isinstance(payload, dict):
            raise GuardError("runpodctl_shape", "network-volume query did not return an object")
        return payload

    def stop_exact_pod(self) -> dict:
        payload = self._call(
            ["pod", "stop", POD_ID, "-o", "json"], "exact pod stop"
        )
        if not isinstance(payload, dict):
            raise GuardError("runpodctl_shape", "exact pod stop did not return an object")
        return payload


def safe_pod_projection(payload: Mapping[str, Any]) -> dict:
    allowed = (
        "id",
        "name",
        "desiredStatus",
        "runtimeStatus",
        "runtimeStatusReason",
        "lastStatusChange",
        "createdAt",
        "gpuCount",
        "networkVolumeId",
        "volumeInGb",
        "volumeMountPath",
    )
    return {key: payload.get(key) for key in allowed if key in payload}


def safe_volume_projection(payload: Mapping[str, Any]) -> dict:
    allowed = ("id", "name", "size", "dataCenterId")
    return {key: payload.get(key) for key in allowed if key in payload}


def validate_exact_pod(payload: Mapping[str, Any]) -> None:
    checks = {
        "id": payload.get("id") == POD_ID,
        "name": payload.get("name") == POD_NAME,
        "gpu_count": payload.get("gpuCount") == GPU_COUNT,
        "network_volume_id": payload.get("networkVolumeId") == NETWORK_VOLUME_ID,
        "volume_mount_path": payload.get("volumeMountPath") == VOLUME_MOUNT_PATH,
        "no_local_volume_substitute": payload.get("volumeInGb") in (0, None),
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise GuardError("pod_identity_mismatch", "exact pod identity failed: " + ",".join(failed))


def validate_exact_volume(payload: Mapping[str, Any]) -> None:
    checks = {
        "id": payload.get("id") == NETWORK_VOLUME_ID,
        "name": payload.get("name") == NETWORK_VOLUME_NAME,
        "size": payload.get("size") == NETWORK_VOLUME_SIZE_GB,
        "datacenter": payload.get("dataCenterId") == NETWORK_VOLUME_DATACENTER,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise GuardError(
            "volume_identity_mismatch", "retained volume identity failed: " + ",".join(failed)
        )


def identity_payload(pod: Mapping[str, Any], volume: Mapping[str, Any]) -> dict:
    return {"pod": safe_pod_projection(pod), "network_volume": safe_volume_projection(volume)}


def identity_sha256(pod: Mapping[str, Any], volume: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(identity_payload(pod, volume)))


@dataclass
class Experiment2D5CRunPodGuard:
    client: Any
    now: Callable[[], datetime] = utc_now
    sleeper: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def _live_identity(self, *, require_running: bool) -> tuple[dict, list[dict], dict]:
        pod = self.client.pod_get()
        rows = self.client.pod_list_all()
        volume = self.client.network_volume_get()
        validate_exact_pod(pod)
        validate_exact_volume(volume)
        matches = [row for row in rows if row.get("id") == POD_ID]
        if len(matches) != 1 or matches[0].get("name") != POD_NAME:
            raise GuardError(
                "pod_selection_ambiguous",
                "account pod list does not contain one exact target",
            )
        if require_running and not (
            pod.get("desiredStatus") == "RUNNING" and pod.get("runtimeStatus") == "running"
        ):
            raise GuardError("pod_not_running", "exact pod is not in verified running state")
        return pod, rows, volume

    def preflight(self, authorization_path: Path, valid_for_seconds: int) -> dict:
        if valid_for_seconds < 60 or valid_for_seconds > 7 * 24 * 60 * 60:
            raise GuardError("invalid_ttl", "authorization lifetime is outside 60s..7d")
        pod, rows, volume = self._live_identity(require_running=True)
        issued = self.now()
        authorization = {
            "schema": AUTHORIZATION_SCHEMA,
            "experiment": EXPERIMENT,
            "action": ACTION,
            "pod_id": POD_ID,
            "pod_name": POD_NAME,
            "gpu_count": GPU_COUNT,
            "network_volume_id": NETWORK_VOLUME_ID,
            "network_volume_name": NETWORK_VOLUME_NAME,
            "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
            "network_volume_datacenter": NETWORK_VOLUME_DATACENTER,
            "volume_mount_path": VOLUME_MOUNT_PATH,
            "pod_created_at": pod.get("createdAt"),
            "pod_running_last_status_change": pod.get("lastStatusChange"),
            "identity_sha256": identity_sha256(pod, volume),
            "exact_stop_command": EXACT_STOP_COMMAND,
            "credential_source": {
                "kind": "macOS Keychain generic password",
                "service": KEYCHAIN_SERVICE,
                "account": KEYCHAIN_ACCOUNT,
            },
            "issued_at_utc": canonical_utc(issued),
            "expires_at_utc": canonical_utc(issued + timedelta(seconds=valid_for_seconds)),
            "authorization_nonce": secrets.token_hex(32),
        }
        digest = write_private_json_exclusive(authorization_path, authorization)
        return {
            "schema": REPORT_SCHEMA,
            "mode": "preflight",
            "passed": True,
            "authenticated": True,
            "exact_id_match_count": sum(row.get("id") == POD_ID for row in rows),
            "pod": safe_pod_projection(pod),
            "network_volume": safe_volume_projection(volume),
            "authorization_artifact": str(authorization_path),
            "authorization_sha256": digest,
            "authorization_expires_at_utc": authorization["expires_at_utc"],
            "exact_stop_command": EXACT_STOP_COMMAND,
            "secret_recorded": False,
        }

    def validate_authorization(self, path: Path) -> tuple[dict, bytes]:
        authorization, raw = read_private_canonical_json(path, AUTHORIZATION_KEYS)
        expected = {
            "schema": AUTHORIZATION_SCHEMA,
            "experiment": EXPERIMENT,
            "action": ACTION,
            "pod_id": POD_ID,
            "pod_name": POD_NAME,
            "gpu_count": GPU_COUNT,
            "network_volume_id": NETWORK_VOLUME_ID,
            "network_volume_name": NETWORK_VOLUME_NAME,
            "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
            "network_volume_datacenter": NETWORK_VOLUME_DATACENTER,
            "volume_mount_path": VOLUME_MOUNT_PATH,
            "exact_stop_command": EXACT_STOP_COMMAND,
            "credential_source": {
                "kind": "macOS Keychain generic password",
                "service": KEYCHAIN_SERVICE,
                "account": KEYCHAIN_ACCOUNT,
            },
        }
        for key, value in expected.items():
            if authorization.get(key) != value:
                raise GuardError("authorization_mismatch", f"authorization {key} is not exact")
        nonce = authorization.get("authorization_nonce")
        if not isinstance(nonce, str) or len(nonce) != 64:
            raise GuardError("invalid_artifact", "authorization nonce is invalid")
        try:
            int(nonce, 16)
        except ValueError as error:
            raise GuardError("invalid_artifact", "authorization nonce is invalid") from error
        identity_digest = authorization.get("identity_sha256")
        if not isinstance(identity_digest, str) or len(identity_digest) != 64:
            raise GuardError("invalid_artifact", "authorization identity digest is invalid")
        try:
            int(identity_digest, 16)
        except ValueError as error:
            raise GuardError(
                "invalid_artifact", "authorization identity digest is invalid"
            ) from error
        issued = parse_utc(authorization.get("issued_at_utc"), "authorization issued_at_utc")
        expires = parse_utc(authorization.get("expires_at_utc"), "authorization expires_at_utc")
        now = self.now()
        if issued > now + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
            raise GuardError("authorization_not_yet_valid", "authorization is from the future")
        if expires <= issued or now >= expires:
            raise GuardError("authorization_expired", "authorization has expired")
        return authorization, raw

    def create_trigger(
        self,
        authorization_path: Path,
        trigger_path: Path,
        terminal_outcome: str,
        exit_code: int,
        source: str,
    ) -> dict:
        authorization, raw = self.validate_authorization(authorization_path)
        if terminal_outcome not in {"success", "failure"}:
            raise GuardError("invalid_trigger", "terminal outcome must be success or failure")
        if not isinstance(exit_code, int) or exit_code < 0 or exit_code > 255:
            raise GuardError("invalid_trigger", "exit code is outside 0..255")
        if (terminal_outcome == "success") != (exit_code == 0):
            raise GuardError("invalid_trigger", "terminal outcome and exit code disagree")
        if source not in {"explicit_terminal", "supervised_child"}:
            raise GuardError("invalid_trigger", "trigger source is not allowed")
        trigger = {
            "schema": TRIGGER_SCHEMA,
            "experiment": EXPERIMENT,
            "action": ACTION,
            "pod_id": POD_ID,
            "pod_name": POD_NAME,
            "gpu_count": GPU_COUNT,
            "network_volume_id": NETWORK_VOLUME_ID,
            "authorization_sha256": sha256_bytes(raw),
            "authorization_nonce": authorization["authorization_nonce"],
            "terminal_outcome": terminal_outcome,
            "exit_code": exit_code,
            "source": source,
            "created_at_utc": canonical_utc(self.now()),
        }
        digest = write_private_json_exclusive(trigger_path, trigger)
        return {
            "schema": REPORT_SCHEMA,
            "mode": "trigger",
            "passed": True,
            "terminal_outcome": terminal_outcome,
            "exit_code": exit_code,
            "trigger_artifact": str(trigger_path),
            "trigger_sha256": digest,
        }

    def validate_trigger(
        self, authorization: Mapping[str, Any], authorization_raw: bytes, trigger_path: Path
    ) -> tuple[dict, bytes]:
        trigger, raw = read_private_canonical_json(trigger_path, TRIGGER_KEYS)
        expected = {
            "schema": TRIGGER_SCHEMA,
            "experiment": EXPERIMENT,
            "action": ACTION,
            "pod_id": POD_ID,
            "pod_name": POD_NAME,
            "gpu_count": GPU_COUNT,
            "network_volume_id": NETWORK_VOLUME_ID,
            "authorization_sha256": sha256_bytes(authorization_raw),
            "authorization_nonce": authorization["authorization_nonce"],
        }
        for key, value in expected.items():
            if trigger.get(key) != value:
                raise GuardError("trigger_mismatch", f"trigger {key} is not exact")
        outcome = trigger.get("terminal_outcome")
        exit_code = trigger.get("exit_code")
        if outcome not in {"success", "failure"} or not isinstance(exit_code, int):
            raise GuardError("invalid_trigger", "trigger outcome is invalid")
        if (outcome == "success") != (exit_code == 0):
            raise GuardError("invalid_trigger", "trigger outcome and exit code disagree")
        if trigger.get("source") not in {"explicit_terminal", "supervised_child"}:
            raise GuardError("invalid_trigger", "trigger source is invalid")
        created = parse_utc(trigger.get("created_at_utc"), "trigger created_at_utc")
        issued = parse_utc(authorization.get("issued_at_utc"), "authorization issued_at_utc")
        expires = parse_utc(authorization.get("expires_at_utc"), "authorization expires_at_utc")
        if created < issued or created > self.now() + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
            raise GuardError("invalid_trigger", "trigger timestamp is out of order")
        if created >= expires:
            raise GuardError("invalid_trigger", "trigger was created after authorization expiry")
        return trigger, raw

    def stop(
        self,
        authorization_path: Path,
        trigger_path: Path,
        *,
        timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> dict:
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise GuardError("invalid_timeout", "stop timeout and poll interval must be positive")
        authorization, authorization_raw = self.validate_authorization(authorization_path)
        trigger, trigger_raw = self.validate_trigger(
            authorization, authorization_raw, trigger_path
        )
        pod = self.client.pod_get()
        volume_before = self.client.network_volume_get()
        validate_exact_pod(pod)
        validate_exact_volume(volume_before)

        already_stopped = (
            pod.get("desiredStatus") == "EXITED"
            and pod.get("runtimeStatus") == "stopped"
        )
        stop_in_progress = (
            pod.get("desiredStatus") == "EXITED" and not already_stopped
        )
        if not already_stopped and not stop_in_progress and not (
            pod.get("desiredStatus") == "RUNNING" and pod.get("runtimeStatus") == "running"
        ):
            raise GuardError("unsafe_pod_state", "exact pod is neither verified running nor exited")
        if not already_stopped and not stop_in_progress and (
            pod.get("createdAt") != authorization.get("pod_created_at")
            or pod.get("lastStatusChange")
            != authorization.get("pod_running_last_status_change")
            or identity_sha256(pod, volume_before) != authorization.get("identity_sha256")
        ):
            raise GuardError(
                "pod_instance_changed",
                "live pod instance no longer matches authorization",
            )

        stop_invoked = False
        if not already_stopped and not stop_in_progress:
            stop_response = self.client.stop_exact_pod()
            stop_invoked = True
            if stop_response.get("id") not in (None, POD_ID):
                raise GuardError("stop_response_mismatch", "stop response identified another pod")

        deadline = self.monotonic() + timeout_seconds
        final_pod = pod if already_stopped else None
        while self.monotonic() < deadline:
            if final_pod is not None:
                break
            candidate = self.client.pod_get()
            validate_exact_pod(candidate)
            if (
                candidate.get("desiredStatus") == "EXITED"
                and candidate.get("runtimeStatus") == "stopped"
            ):
                final_pod = candidate
                break
            self.sleeper(poll_interval_seconds)
        if final_pod is None:
            raise GuardError(
                "stop_timeout",
                "exact pod did not reach EXITED/stopped before timeout",
            )

        final_volume = self.client.network_volume_get()
        validate_exact_volume(final_volume)
        if final_pod.get("networkVolumeId") != NETWORK_VOLUME_ID:
            raise GuardError(
                "volume_detached",
                "stopped pod no longer identifies the retained volume",
            )
        return {
            "schema": REPORT_SCHEMA,
            "mode": "stop",
            "passed": True,
            "stop_invoked": stop_invoked,
            "status": (
                "stopped_and_volume_retained_verified"
                if stop_invoked else "already_stopped_verified"
            ),
            "pod": safe_pod_projection(final_pod),
            "network_volume": safe_volume_projection(final_volume),
            "authorization_sha256": sha256_bytes(authorization_raw),
            "trigger_sha256": sha256_bytes(trigger_raw),
            "terminal_outcome": trigger["terminal_outcome"],
            "exact_stop_command": EXACT_STOP_COMMAND,
            "secret_recorded": False,
        }

    def wait_for_trigger(
        self,
        authorization_path: Path,
        trigger_path: Path,
        *,
        wait_timeout_seconds: float,
        poll_interval_seconds: float,
        stop_timeout_seconds: float,
    ) -> dict:
        self.validate_authorization(authorization_path)
        _require_absolute_artifact_path(trigger_path, "trigger path")
        if wait_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise GuardError("invalid_timeout", "watch timeout and poll interval must be positive")
        deadline = self.monotonic() + wait_timeout_seconds
        while self.monotonic() < deadline:
            if trigger_path.exists():
                return self.stop(
                    authorization_path,
                    trigger_path,
                    timeout_seconds=stop_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
            self.sleeper(poll_interval_seconds)
        raise GuardError("watch_timeout", "terminal trigger did not appear before watchdog timeout")

    def supervise_and_stop(
        self,
        authorization_path: Path,
        trigger_path: Path,
        child_command: Sequence[str],
        *,
        stop_timeout_seconds: float,
        poll_interval_seconds: float,
        child_runner: Callable[..., Any] | None = None,
    ) -> tuple[int, dict]:
        authorization, authorization_raw = self.validate_authorization(
            authorization_path
        )
        _require_absolute_artifact_path(trigger_path, "trigger path")
        if not child_command:
            raise GuardError("missing_child", "supervised child command is empty")
        if not trigger_path.parent.is_dir():
            raise GuardError("unsafe_path", "trigger parent directory does not exist")
        if trigger_path.exists():
            raise GuardError("artifact_exists", "supervised trigger path already exists")
        child_environment = dict(os.environ)
        child_environment.pop("RUNPOD_API_KEY", None)
        runner = child_runner or subprocess.run
        interrupted_signal = None

        class SupervisionSignal(Exception):
            def __init__(self, signum: int):
                self.signum = signum

        def signal_handler(signum: int, _frame: Any) -> None:
            raise SupervisionSignal(signum)

        previous_handlers = {}
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signal_handler)
        try:
            completed = runner(list(child_command), env=child_environment, check=False)
            exit_code = int(completed.returncode)
        except SupervisionSignal as error:
            interrupted_signal = error.signum
            exit_code = min(255, 128 + error.signum)
        except OSError:
            exit_code = 127
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        if exit_code < 0:
            exit_code = min(255, 128 + abs(exit_code))
        else:
            exit_code = min(exit_code, 255)
        outcome = "success" if exit_code == 0 else "failure"
        if outcome == "failure":
            # A failed scientific gate blocks the workflow but is normally
            # recoverable.  Preserve the scarce allocation for diagnosis; no
            # stop-capable trigger is created without an explicit reviewed
            # terminal action.
            pod = self.client.pod_get()
            volume = self.client.network_volume_get()
            validate_exact_pod(pod)
            validate_exact_volume(volume)
            if identity_sha256(pod, volume) != authorization["identity_sha256"]:
                raise GuardError(
                    "pod_instance_changed",
                    "live pod instance no longer matches authorization",
                )
            report = {
                "schema": REPORT_SCHEMA,
                "mode": "watchdog_supervise_failure_retained",
                "passed": True,
                "terminal_outcome": "failure",
                "child_exit_code": exit_code,
                "pod_stop_attempted": False,
                "trigger_artifact_created": False,
                "retained_for_recoverable_diagnosis": True,
                "pod": safe_pod_projection(pod),
                "network_volume": safe_volume_projection(volume),
                "authorization_sha256": sha256_bytes(authorization_raw),
                "exact_stop_command": EXACT_STOP_COMMAND,
                "secret_recorded": False,
            }
            if interrupted_signal is not None:
                report["supervision_signal"] = interrupted_signal
            return exit_code, report
        self.create_trigger(
            authorization_path,
            trigger_path,
            outcome,
            exit_code,
            "supervised_child",
        )
        report = self.stop(
            authorization_path,
            trigger_path,
            timeout_seconds=stop_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        report = {**report, "mode": "watchdog_supervise", "child_exit_code": exit_code}
        if interrupted_signal is not None:
            report["supervision_signal"] = interrupted_signal
        return exit_code, report


def _emit(payload: Mapping[str, Any], stream: Any = sys.stdout) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _record_report(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    """Write one clean machine-readable report apart from inherited child logs."""
    path = getattr(args, "report_artifact", None)
    if path is not None:
        write_private_json_exclusive(path, payload)


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-target RunPod guard for Experiment 2D5C"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="read-only authenticated identity check and authorization"
    )
    preflight.add_argument("--authorization-artifact", type=_path, required=True)
    preflight.add_argument(
        "--valid-for-seconds", type=int, default=DEFAULT_AUTHORIZATION_SECONDS
    )

    trigger = subparsers.add_parser(
        "trigger", help="create the exact terminal trigger; performs no RunPod mutation"
    )
    trigger.add_argument("--authorization-artifact", type=_path, required=True)
    trigger.add_argument("--trigger-file", type=_path, required=True)
    trigger.add_argument("--outcome", choices=("success", "failure"), required=True)
    trigger.add_argument("--exit-code", type=int, required=True)

    stop = subparsers.add_parser(
        "stop", help="trigger-gated stop of the one exact authorized pod"
    )
    stop.add_argument("--authorization-artifact", type=_path, required=True)
    stop.add_argument("--trigger-file", type=_path, required=True)
    stop.add_argument("--report-artifact", type=_path)
    stop.add_argument("--stop-timeout-seconds", type=float, default=DEFAULT_STOP_TIMEOUT_SECONDS)
    stop.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)

    watchdog = subparsers.add_parser(
        "watchdog", help="wait for an exact trigger or supervise a whole child workflow"
    )
    watchdog.add_argument("--authorization-artifact", type=_path, required=True)
    watchdog.add_argument("--trigger-file", type=_path, required=True)
    watchdog.add_argument("--report-artifact", type=_path)
    watchdog.add_argument(
        "--watch-timeout-seconds", type=float, default=DEFAULT_WATCH_TIMEOUT_SECONDS
    )
    watchdog.add_argument(
        "--stop-timeout-seconds", type=float, default=DEFAULT_STOP_TIMEOUT_SECONDS
    )
    watchdog.add_argument(
        "--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS
    )
    watchdog.add_argument(
        "child_command",
        nargs=argparse.REMAINDER,
        help="optional whole workflow command after --; absence means wait for trigger",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    guard = Experiment2D5CRunPodGuard(RunPodClient())
    try:
        if args.mode == "preflight":
            report = guard.preflight(args.authorization_artifact, args.valid_for_seconds)
            _emit(report)
            return 0
        if args.mode == "trigger":
            report = guard.create_trigger(
                args.authorization_artifact,
                args.trigger_file,
                args.outcome,
                args.exit_code,
                "explicit_terminal",
            )
            _emit(report)
            return 0
        if args.mode == "stop":
            report = guard.stop(
                args.authorization_artifact,
                args.trigger_file,
                timeout_seconds=args.stop_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )
            _record_report(args, report)
            _emit(report)
            return 0
        if args.mode == "watchdog":
            child_command = list(args.child_command)
            if child_command and child_command[0] == "--":
                child_command = child_command[1:]
            if child_command:
                exit_code, report = guard.supervise_and_stop(
                    args.authorization_artifact,
                    args.trigger_file,
                    child_command,
                    stop_timeout_seconds=args.stop_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                )
                _record_report(args, report)
                _emit(report)
                return exit_code
            report = guard.wait_for_trigger(
                args.authorization_artifact,
                args.trigger_file,
                wait_timeout_seconds=args.watch_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                stop_timeout_seconds=args.stop_timeout_seconds,
            )
            report = {**report, "mode": "watchdog_trigger"}
            _record_report(args, report)
            _emit(report)
            return 0
        raise GuardError("invalid_mode", "guard mode is unsupported")
    except GuardError as error:
        failure = {
            "schema": REPORT_SCHEMA,
            "mode": getattr(args, "mode", "unknown"),
            "passed": False,
            "error_code": error.code,
            "message": error.message,
            "secret_recorded": False,
        }
        try:
            _record_report(args, failure)
        except GuardError:
            pass
        _emit(failure, sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
