import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c_runpod_guard as runpod_guard  # noqa: E402


RUNNING_POD = {
    "id": runpod_guard.POD_ID,
    "name": runpod_guard.POD_NAME,
    "desiredStatus": "RUNNING",
    "runtimeStatus": "running",
    "gpuCount": 1,
    "networkVolumeId": runpod_guard.NETWORK_VOLUME_ID,
    "volumeInGb": 0,
    "volumeMountPath": "/workspace",
    "createdAt": "2026-08-20T10:00:00Z",
    "lastStatusChange": "Resumed by user: Sun Aug 30 2026 14:18:54 GMT+0000",
    "env": {"MUST_NOT_APPEAR": "secret-value"},
}

STOPPED_POD = {
    **RUNNING_POD,
    "desiredStatus": "EXITED",
    "runtimeStatus": "stopped",
    "lastStatusChange": "Stopped by user: Sun Aug 30 2026 18:00:00 GMT+0000",
}

VOLUME = {
    "id": runpod_guard.NETWORK_VOLUME_ID,
    "name": runpod_guard.NETWORK_VOLUME_NAME,
    "size": runpod_guard.NETWORK_VOLUME_SIZE_GB,
    "dataCenterId": runpod_guard.NETWORK_VOLUME_DATACENTER,
}


class FakeClient:
    def __init__(self):
        self.stopped = False
        self.stop_arguments = []
        self.calls = []
        self.pod_override = None
        self.volume_override = None

    def pod_get(self):
        self.calls.append("pod_get")
        if self.pod_override is not None:
            return dict(self.pod_override)
        return dict(STOPPED_POD if self.stopped else RUNNING_POD)

    def pod_list_all(self):
        self.calls.append("pod_list_all")
        return [
            {"id": "some-other-pod", "name": "other"},
            {
                "id": runpod_guard.POD_ID,
                "name": runpod_guard.POD_NAME,
                "gpuCount": 1,
                "desiredStatus": "RUNNING",
            },
        ]

    def network_volume_get(self):
        self.calls.append("network_volume_get")
        return dict(self.volume_override or VOLUME)

    def stop_exact_pod(self):
        self.calls.append("stop_exact_pod")
        self.stop_arguments.append(
            ("pod", "stop", runpod_guard.POD_ID, "-o", "json")
        )
        self.stopped = True
        return {
            "id": runpod_guard.POD_ID,
            "desiredStatus": "EXITED",
            "env": {"MUST_NOT_APPEAR": "secret-value"},
        }


def fixed_now():
    return datetime(2026, 8, 30, 17, 0, 0, tzinfo=timezone.utc)


def make_guard(client=None):
    return runpod_guard.Experiment2D5CRunPodGuard(
        client or FakeClient(),
        now=fixed_now,
        sleeper=lambda _: None,
        monotonic=lambda: 1.0,
    )


def authorize_and_trigger(tmp_path, guard, outcome="success", exit_code=0):
    authorization = tmp_path / "exp2d5c-stop-authorization.json"
    trigger = tmp_path / "exp2d5c-terminal-trigger.json"
    guard.preflight(authorization, valid_for_seconds=3600)
    guard.create_trigger(
        authorization,
        trigger,
        outcome,
        exit_code,
        "explicit_terminal",
    )
    return authorization, trigger


def test_preflight_is_read_only_private_and_allowlisted(tmp_path):
    client = FakeClient()
    guard = make_guard(client)
    authorization = tmp_path / "exp2d5c-stop-authorization.json"

    report = guard.preflight(authorization, valid_for_seconds=3600)

    assert report["passed"] is True
    assert report["authenticated"] is True
    assert report["exact_id_match_count"] == 1
    assert report["pod"]["id"] == runpod_guard.POD_ID
    assert "env" not in report["pod"]
    assert "secret-value" not in json.dumps(report)
    assert report["network_volume"]["id"] == runpod_guard.NETWORK_VOLUME_ID
    assert stat.S_IMODE(authorization.stat().st_mode) == 0o600
    assert client.stop_arguments == []
    assert client.calls == ["pod_get", "pod_list_all", "network_volume_get"]

    payload = json.loads(authorization.read_text())
    assert set(payload) == set(runpod_guard.AUTHORIZATION_KEYS)
    assert payload["exact_stop_command"] == (
        "runpodctl pod stop h6of430yxncf6h -o json"
    )
    assert payload["pod_running_last_status_change"] == RUNNING_POD["lastStatusChange"]


def test_stop_requires_bound_trigger_uses_only_exact_stop_and_verifies_volume(tmp_path):
    client = FakeClient()
    guard = make_guard(client)
    authorization, trigger = authorize_and_trigger(tmp_path, guard)

    report = guard.stop(
        authorization,
        trigger,
        timeout_seconds=5,
        poll_interval_seconds=0.01,
    )

    assert report["passed"] is True
    assert report["stop_invoked"] is True
    assert report["status"] == "stopped_and_volume_retained_verified"
    assert report["pod"]["desiredStatus"] == "EXITED"
    assert report["network_volume"] == VOLUME
    assert report["exact_stop_command"] == runpod_guard.EXACT_STOP_COMMAND
    assert client.stop_arguments == [
        ("pod", "stop", "h6of430yxncf6h", "-o", "json")
    ]
    assert "secret-value" not in json.dumps(report)


def test_stop_without_trigger_or_with_identity_change_fails_before_mutation(tmp_path):
    client = FakeClient()
    guard = make_guard(client)
    authorization = tmp_path / "exp2d5c-stop-authorization.json"
    missing_trigger = tmp_path / "missing-trigger.json"
    guard.preflight(authorization, valid_for_seconds=3600)

    with pytest.raises(runpod_guard.GuardError, match="required artifact"):
        guard.stop(authorization, missing_trigger)
    assert client.stop_arguments == []

    trigger = tmp_path / "exp2d5c-terminal-trigger.json"
    guard.create_trigger(
        authorization, trigger, "failure", 7, "explicit_terminal"
    )
    client.pod_override = {**RUNNING_POD, "networkVolumeId": "wrong-volume"}
    with pytest.raises(runpod_guard.GuardError, match="exact pod identity"):
        guard.stop(authorization, trigger)
    assert client.stop_arguments == []


def test_trigger_is_bound_to_exact_authorization_digest(tmp_path):
    client = FakeClient()
    guard = make_guard(client)
    authorization, trigger = authorize_and_trigger(tmp_path, guard)
    payload = json.loads(trigger.read_text())
    payload["authorization_sha256"] = "0" * 64
    trigger.write_bytes(runpod_guard.canonical_json_bytes(payload))
    os.chmod(trigger, 0o600)

    with pytest.raises(runpod_guard.GuardError, match="authorization_sha256"):
        guard.stop(authorization, trigger)
    assert client.stop_arguments == []


@pytest.mark.parametrize("child_exit", [0, 9])
def test_supervised_success_and_failure_both_create_trigger_and_stop(
    tmp_path, child_exit
):
    client = FakeClient()
    guard = make_guard(client)
    authorization = tmp_path / "exp2d5c-stop-authorization.json"
    trigger = tmp_path / f"exp2d5c-terminal-{child_exit}.json"
    guard.preflight(authorization, valid_for_seconds=3600)

    def child_runner(argv, **kwargs):
        assert argv == ["whole-experiment-command", "--sealed"]
        assert "RUNPOD_API_KEY" not in kwargs["env"]
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(argv, child_exit)

    returned_exit, report = guard.supervise_and_stop(
        authorization,
        trigger,
        ["whole-experiment-command", "--sealed"],
        stop_timeout_seconds=5,
        poll_interval_seconds=0.01,
        child_runner=child_runner,
    )

    assert returned_exit == child_exit
    assert report["child_exit_code"] == child_exit
    assert report["terminal_outcome"] == (
        "success" if child_exit == 0 else "failure"
    )
    assert client.stop_arguments == [
        ("pod", "stop", "h6of430yxncf6h", "-o", "json")
    ]
    trigger_payload = json.loads(trigger.read_text())
    assert trigger_payload["source"] == "supervised_child"
    assert trigger_payload["exit_code"] == child_exit


def test_supervised_mode_rejects_nonexact_trigger_path_before_child_runs(tmp_path):
    client = FakeClient()
    guard = make_guard(client)
    authorization = tmp_path / "exp2d5c-stop-authorization.json"
    guard.preflight(authorization, valid_for_seconds=3600)
    child_ran = False

    def child_runner(argv, **kwargs):
        nonlocal child_ran
        child_ran = True
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(runpod_guard.GuardError, match="absolute path"):
        guard.supervise_and_stop(
            authorization,
            Path("relative-trigger.json"),
            ["whole-experiment-command"],
            stop_timeout_seconds=5,
            poll_interval_seconds=0.01,
            child_runner=child_runner,
        )
    assert child_ran is False
    assert client.stop_arguments == []


def test_runpod_client_passes_secret_only_in_child_env_and_exact_stop_argv(monkeypatch):
    calls = []

    class StaticCredential:
        def read(self):
            return b"test-key-never-print"

    def fake_run(argv, **kwargs):
        captured = dict(kwargs)
        captured["env"] = dict(kwargs["env"])
        calls.append((argv, captured))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "id": runpod_guard.POD_ID,
                    "desiredStatus": "EXITED",
                    "env": {"sensitive": "not-emitted"},
                }
            ).encode(),
            stderr=b"",
        )

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(runpod_guard.subprocess, "run", fake_run)
    client = runpod_guard.RunPodClient(
        Path("/opt/homebrew/bin/runpodctl"), StaticCredential()
    )

    response = client.stop_exact_pod()

    assert response["id"] == runpod_guard.POD_ID
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        "/opt/homebrew/bin/runpodctl",
        "pod",
        "stop",
        "h6of430yxncf6h",
        "-o",
        "json",
    ]
    environment = kwargs["env"]
    assert environment.get(b"RUNPOD_API_KEY", environment.get("RUNPOD_API_KEY")) in {
        b"test-key-never-print",
        "test-key-never-print",
    }
    assert "RUNPOD_API_KEY" not in os.environ


def test_safe_projections_never_include_raw_environment():
    assert "env" not in runpod_guard.safe_pod_projection(RUNNING_POD)
    assert "secret-value" not in json.dumps(
        runpod_guard.safe_pod_projection(RUNNING_POD)
    )
