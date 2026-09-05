"""Exact-pod independent guard for the user-started bounded preparation probe."""
import argparse
import sys
import time
from pathlib import Path
from .common import REPO, atomic_json, read_json
sys.path.insert(0, str(REPO/'scripts'))
from experiment_2d5c_runpod_guard import RunPodClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    expected = read_json(out/'ASSIGNED_POD_INITIAL_STATUS.json')
    assert expected['id'] == 'nagj1hv18p3z2c' and expected['gpuCount'] == 4
    assert expected['networkVolumeId'] == 'yhzyb27fb5'
    client = RunPodClient()
    atomic_json(out/'INITIAL_GUARD.json', dict(armed=True, deadline=args.deadline,
                exact_pod=expected['id'], independent_process=True))
    while time.time() < args.deadline and not (out/'STOP_INITIAL_PROBE').exists():
        time.sleep(5)
    retries = []
    while True:
        try:
            p = client._call(['pod','get',expected['id'],'-o','json'], 'verify exact assigned pod')
            assert all(p[k] == expected[k] for k in ('id','name','createdAt','networkVolumeId','gpuCount'))
            state = {k: p.get(k) for k in ('id','desiredStatus','runtimeStatus','networkVolumeId','lastStatusChange')}
            if state['desiredStatus'] == 'EXITED' and state['runtimeStatus'] == 'stopped':
                atomic_json(out/'INITIAL_STOP_VERIFICATION.json', dict(passed=True, pod=state,
                            verified_at=time.time(), retries=retries, volume_retained=True))
                return
            client._call(['pod','stop',expected['id'],'-o','json'], 'stop assigned 2D11 preparation pod')
            retries.append(dict(time=time.time(), stop_requested=True))
        except Exception as e:
            retries.append(dict(time=time.time(), error=type(e).__name__))
        atomic_json(out/'INITIAL_STOP_RETRIES.json', dict(alert='POD STOP NOT YET VERIFIED', retries=retries))
        time.sleep(10)


if __name__ == '__main__':
    main()
