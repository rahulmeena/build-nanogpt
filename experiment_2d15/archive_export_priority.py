"""Give required scientific checkpoint exports priority over historical copies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def run(archive, job):
    pid=json.loads((job/'LAUNCH.json').read_text())['pid']
    assert os.getpgid(pid)==pid
    minimum=27_250_000_000
    paused=None
    while True:
        rows=subprocess.check_output(['ps','-axo','pid=,ppid=,stat=,command='],text=True).splitlines()
        processes={int(r.split(None,3)[0]):r.split(None,3) for r in rows}
        if pid not in processes or processes[pid][2].startswith('Z'):
            (job/'PRIORITY_GUARD_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),archival_process_finished=True))+'\n')
            return
        if (archive/'STOP_VERIFICATION.json').exists():
            os.killpg(pid,signal.SIGTERM)
            os.killpg(pid,signal.SIGCONT)
            return
        controller=json.loads((archive/'LOCAL_CONTROLLER_LAUNCH.json').read_text())['pid']
        transfer=any(int(r[1])==controller and 'rsync -a --partial' in r[3] for r in processes.values())
        temporary=any('downloading' in p.name for arm in ['L_nf4','R_nf4'] for p in (archive/arm/'checkpoints').iterdir())
        heartbeat=json.loads((archive/'CONTROLLER_HEARTBEAT.json').read_text())['time']
        free=shutil.disk_usage(archive).free
        reasons=[]
        if transfer or temporary:reasons.append('required_checkpoint_export')
        if controller not in processes or time.time()-heartbeat>120:reasons.append('await_fresh_controller_cycle')
        if free<minimum:reasons.append('preserve_export_allowance_and_Mac_reserve')
        should_pause=bool(reasons)
        if should_pause!=paused:
            os.killpg(pid,signal.SIGSTOP if should_pause else signal.SIGCONT)
            event=dict(time=time.time(),event='PAUSE' if should_pause else 'RESUME',pid=pid,reasons=reasons,mac_free_bytes=free,pod_stop_requested=False)
            with (job/'PRIORITY_EVENTS.jsonl').open('a') as out:out.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
            paused=should_pause
        time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    args=p.parse_args()
    try:run(args.archive,args.job)
    except Exception as e:
        (args.job/'PRIORITY_GUARD_ERROR.json').write_text(json.dumps(dict(time=time.time(),type=type(e).__name__,message=str(e),pod_stop_requested=False))+'\n')
        raise
