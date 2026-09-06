"""Persistent remote supervisor and independent deadline watchdog."""
import argparse
from dataclasses import dataclass,field
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import *
from .provider import Provider


@dataclass
class StateMachine:
    """Pure policy used by the live supervisor and synthetic end-to-end rehearsals."""
    hard_deadline: float
    state: str='STARTUP'
    outcome: str|None=None
    failure_at: float|None=None
    export_started: float|None=None
    trace: list=field(default_factory=list)
    export_timeout: float=900

    def event(self,event,now,**data):
        self.trace.append(dict(event=event,time=now,state=self.state))
        if now>=self.hard_deadline:
            self.state='STOP_POD';self.outcome=self.outcome or 'BUDGET_STOPPED';return self.state
        if event=='preflight_started':self.state='CUDA_PREFLIGHT'
        elif event=='preflight_passed':self.state='RESTORE_FRESH_STATE'
        elif event=='training_started':self.state='TRAIN'
        elif event=='evaluation_started':self.state='SCHEDULED_EVALUATION'
        elif event=='continue':self.state='TRAIN'
        elif event=='futility':
            self.outcome='EARLY_STOPPED_FOR_FUTILITY';self.state='FINAL_EXPORT_VERIFICATION';self.export_started=now
        elif event=='complete':
            self.outcome='COMPLETED';self.state='FINAL_EXPORT_VERIFICATION';self.export_started=now
        elif event in ('rank_failed','stage_deadline','budget_rejected','preflight_failed'):
            self.outcome='BUDGET_STOPPED' if event=='budget_rejected' else 'INVALID_INCOMPLETE'
            self.state='FINAL_EXPORT_VERIFICATION';self.failure_at=now;self.export_started=now
        elif event=='soft_deadline':
            self.state='SAFE_CHECKPOINT';self.outcome='BUDGET_STOPPED'
        elif event=='exports_verified':self.state='STOP_POD'
        elif event=='export_poll' and self.export_started is not None and now-self.export_started>=self.export_timeout:
            self.state='STOP_POD'
        elif event=='stop_verified':self.state='LOCAL_ANALYSIS'
        return self.state


def stage_command(args,module,extra=()):
    argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
          '-m','--','experiment_2d11.'+module,'--run',args.run,'--binding',args.binding,
          '--data',args.data,'--initial',args.initial,'--hella',args.hella]
    argv+=['--baseline',args.baseline]
    return argv+list(extra)


def supervise(args, *, poll_seconds=5, export_seconds=900, terminate_seconds=300, preflight_seconds=1200):
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True);binding=read_json(args.binding)
    machine=StateMachine(binding['hard_deadline'],export_timeout=export_seconds);atomic_json(run/'SUPERVISOR_STARTED.json',dict(pid=os.getpid(),time=time.time()))
    def state(event,**extra):
        machine.event(event,time.time(),**extra)
        atomic_json(run/'SUPERVISOR_STATE.json',dict(state=machine.state,outcome=machine.outcome,
                    timestamp=time.time(),trace=machine.trace,**extra))

    def execute(module,stage_seconds,extra=()):
        log=(run/(module+'-'+str(int(time.time()))+'.log')).open('ab')
        child=subprocess.Popen(stage_command(args,module,extra),stdout=log,stderr=log,start_new_session=True)
        started=time.time();first_progress=False;deadline=min(started+stage_seconds,binding['hard_deadline']-300)
        while child.poll() is None:
            now=time.time()
            atomic_json(run/'SUPERVISOR_HEARTBEAT.json',dict(time=now,child_pid=child.pid,stage=module,deadline=deadline))
            reason=None
            if module=='train' and now-started>600 and not first_progress:reason='recovery_startup_deadline'
            if now>=deadline:reason='stage_deadline'
            for rank in range(4):
                path=run/f'heartbeat-rank{rank}.json'
                if path.exists():
                    h=read_json(path)
                    if h['last_successful_write']>=started:first_progress=True
                    if h['last_successful_write']>=started and (now>h['stage_deadline'] or now-h['last_successful_write']>300):reason='stage_deadline'
            if reason:
                (run/'STOP_REQUEST').touch();os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=terminate_seconds)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                return 124
            time.sleep(poll_seconds)
        return child.returncode

    try:
        resume=getattr(args,'resume',None);attempt=getattr(args,'attempt','attempt01')
        state('preflight_started')
        if resume:
            manifest=read_json(str(resume)+'.manifest.json')
            assert manifest['audit']['passed'] and sha256(resume)==manifest['sha256']
            assert read_json(run/'PREFLIGHT.json')['passed']
            atomic_json(run/'PREFLIGHT_REUSED.json',dict(time=time.time(),checkpoint=str(resume),
                        checkpoint_sha256=manifest['sha256'],preflight_sha256=sha256(run/'PREFLIGHT.json')))
            rc=0
        else:rc=execute('preflight',preflight_seconds)
        if rc!=0:
            state('preflight_failed',exit_code=rc)
        else:
            pf=read_json(run/'PREFLIGHT.json')
            if not pf['passed']:
                state('budget_rejected')
            else:
                state('preflight_passed');state('training_started')
                rc=execute('train',binding['hard_deadline']-time.time()-300,['--attempt',attempt]+(['--resume',str(resume)] if resume else []))
                # One bounded recovery: only a fully verified complete checkpoint, no interactive debugging.
                if rc!=0 and time.time()<binding['hard_deadline']-2400:
                    candidates=sorted((run/'checkpoints').glob('u*.pt.manifest.json'))
                    if candidates:
                        latest=candidates[-1];checkpoint=str(latest).removesuffix('.manifest.json')
                        if read_json(latest)['audit']['passed']:
                            atomic_json(run/'RECOVERY.json',dict(started=time.time(),checkpoint=checkpoint,max_startup_seconds=600))
                            (run/'STOP_REQUEST').unlink(missing_ok=True)
                            rc=execute('train',binding['hard_deadline']-time.time()-300,
                                       ['--attempt',attempt+'_recovery','--resume',checkpoint])
                if rc:
                    state('rank_failed',exit_code=rc)
                else:
                    outcome=read_json(run/'SCIENTIFIC_OUTCOME.json')['outcome']
                    state('complete' if outcome=='COMPLETED' else 'futility' if outcome=='EARLY_STOPPED_FOR_FUTILITY' else 'budget_rejected')
        atomic_json(run/'GPU_WORK_FINISHED.json',dict(outcome=machine.outcome,time=time.time()))
        while machine.state!='STOP_POD':
            atomic_json(run/'SUPERVISOR_HEARTBEAT.json',dict(time=time.time(),stage='export',deadline=binding['hard_deadline']))
            if (run/'FINAL_EXPORTS_VERIFIED.json').exists():state('exports_verified')
            else:state('export_poll')
            if machine.state!='STOP_POD':time.sleep(poll_seconds)
    except BaseException as e:
        atomic_json(run/'SUPERVISOR_FAILURE.json',dict(error_type=type(e).__name__,time=time.time()))
    finally:
        # Independent watchdog owns provider credentials and performs the stop.
        atomic_json(run/'STOP_POD_NOW.json',dict(time=time.time(),outcome=machine.outcome))


def watchdog(args):
    binding=read_json(args.binding);run=Path(args.run);provider=Provider(binding)
    os.environ['EXP2D11_REMOTE_GUARD']='1'
    armed=time.time();failure=None
    atomic_json(run/'REMOTE_WATCHDOG_ARMED.json',dict(time=armed,pid=os.getpid(),hard_deadline=binding['hard_deadline'],pod_id=binding['pod_id']))
    while True:
        now=time.time()
        if now>=binding['hard_deadline'] or (run/'STOP_POD_NOW.json').exists():break
        h=run/'SUPERVISOR_HEARTBEAT.json'
        last=read_json(h).get('time',armed) if h.exists() else armed
        startup_allowance=max(armed,binding['billing_start']+600) if not h.exists() else last
        if now-startup_allowance>300:
            (run/'STOP_REQUEST').touch()
            failure=failure or now
        elif failure is not None:
            failure=None
        if failure is not None and now-failure>330:
            atomic_json(run/'WATCHDOG_FAILURE.json',dict(reason='stalled supervisor',time=now));break
        if now>=binding['hard_deadline']-1800:(run/'STOP_REQUEST').touch()
        time.sleep(5)
    provider.stop_verified(run/'REMOTE_STOP_VERIFICATION.json')


def install_watchdog(args):
    # Read the existing credential through encrypted SSH stdin. Keep it only in child memory.
    credential=sys.stdin.buffer.read().strip()
    assert credential and len(credential)<1024
    env=dict(os.environ);env['RUNPOD_API_KEY']=credential.decode();env['EXP2D11_REMOTE_GUARD']='1'
    log=(Path(args.run)/'watchdog.log').open('ab')
    child=subprocess.Popen([sys.executable,'-m','experiment_2d11.supervisor','watchdog','--run',args.run,
                            '--binding',args.binding],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    credential=b'';env.pop('RUNPOD_API_KEY',None)
    print(json.dumps(dict(watchdog_pid=child.pid)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','watchdog','install-watchdog'])
    p.add_argument('--run',required=True);p.add_argument('--binding',required=True)
    for name in ['data','initial','hella','baseline']:p.add_argument('--'+name)
    p.add_argument('--resume');p.add_argument('--attempt',default='attempt01')
    args=p.parse_args();{'run':supervise,'watchdog':watchdog,'install-watchdog':install_watchdog}[args.mode](args)
