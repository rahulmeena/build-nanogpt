"""Bounded operational recovery of the unchanged u6000 scientific trajectory."""
import os,shlex,shutil,subprocess,sys,tarfile,time,json
from pathlib import Path
from .common import *
from .controller import connection,monitor
from .provider import Provider
from .budget import project_remaining

ARCHIVE=Path('/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11')
HISTORY='network_failure_u6000'
CONTROLS=['STOP_VERIFICATION.json','BILLING_ACCOUNTING.json','CONTROLLER_FAILURE.json',
 'INDEPENDENT_LOCAL_STOP_VERIFICATION.json','INDEPENDENT_LOCAL_STOP_REASON.json',
 'GPU_WORK_FINISHED.json','SCIENTIFIC_OUTCOME.json','FINAL_EXPORTS_VERIFIED.json',
 'STOP_POD_NOW.json','STOP_REQUEST','REMOTE_STOP_VERIFICATION.json','WATCHDOG_FAILURE.json',
 'SUPERVISOR_FAILURE.json','SUPERVISOR_HEARTBEAT.json','SUPERVISOR_STATE.json',
 'SUPERVISOR_STARTED.json','REMOTE_WATCHDOG_ARMED.json','LOCAL_CONTROLLER_HEARTBEAT.json']


def preserve(run):
    """Preserve raw pre-recovery records; canonical history ends at restored boundary."""
    run=Path(run);history=run/'recovery_history'/HISTORY;history.mkdir(parents=True,exist_ok=True)
    for name in CONTROLS+[p.name for p in run.glob('heartbeat-rank*.json')]:
        p=run/name
        if p.exists():
            target=history/name
            assert not target.exists(), 'recovery history must not be overwritten'
            p.rename(target)
    raw=run/'metrics-attempt01.jsonl'
    if raw.exists():
        target=history/raw.name;assert not target.exists()
        shutil.copy2(raw,target)
        rows=[json.loads(x) for x in raw.read_text().splitlines()]
        lost=[r for r in rows if r['completed_updates']>6000]
        atomic_bytes(raw,b''.join(canonical(r)+b'\n' for r in rows if r['completed_updates']<=6000))
        atomic_json(history/'ROLLED_BACK_TAIL.json',dict(checkpoint_updates=6000,
            last_recorded_update=max(r['completed_updates'] for r in rows),discarded_updates=len(lost),
            discarded_training_seconds=sum(r['training_seconds'] for r in lost),all_billing_retained=True))


def recover():
    local=ARCHIVE/'run';bp=ARCHIVE/'binding.json';b=read_json(bp);provider=Provider(b)
    assert provider.stopped(provider.status())
    account=read_json(local/'BILLING_ACCOUNTING.json');assert read_json(local/'STOP_VERIFICATION.json')['passed']
    prior=account['cumulative_billed_seconds']
    pf=read_json(local/'PREFLIGHT.json');assert pf['passed']
    projection=project_remaining(pf['measurements'],prior+600,6000,
                [u for u in CE_SCHEDULE if u<=6000],[u for u in HELLA_SCHEDULE if u<=6000])
    assert projection['fits'],'recovery with ten-minute startup does not fit'
    bundle=ARCHIVE/'network_recovery_patch.tar.gz'
    files=['experiment_2d11/supervisor.py','experiment_2d11/controller.py','experiment_2d11/recover_network.py']
    with tarfile.open(bundle,'w:gz') as tar:
        for name in files:tar.add(REPO/name,arcname=name)
    preserve(local)
    b.update(prior_billed_seconds=prior,billing_start=time.time(),operational_recovery_patch_sha256=sha256(bundle))
    b['hard_deadline']=b['billing_start']+b['cumulative_ceiling_seconds']-prior
    atomic_json(bp,b);atomic_json(local/'BINDING.json',b)
    atomic_json(local/'RECOVERY_LAUNCH.json',dict(time=time.time(),checkpoint_updates=6000,
                operational_patch_sha256=sha256(bundle),scientific_identity_unchanged=True,projection=projection))
    def heartbeat(state):atomic_json(local/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state=state))
    heartbeat('RECOVERY_STARTUP')
    log=(local/'local_guard.log').open('ab')
    guard=subprocess.Popen([sys.executable,'-m','experiment_2d11.controller','guard','--binding',str(bp),'--archive',str(local)],stdout=log,stderr=log,start_new_session=True)
    context=read_json(ARCHIVE/'CURRENT_RUN_CONTEXT.json');context.update(controller_pid=os.getpid(),controller_log=str(ARCHIVE/'network_recovery_controller.log'),status='Starting bounded checkpoint recovery',prior_billed_seconds=prior,hard_deadline=b['hard_deadline'],time=time.time())
    atomic_json(ARCHIVE/'CURRENT_RUN_CONTEXT.json',context)
    try:
        provider.start();print('POD_START_REQUESTED',flush=True)
        remote=None;deadline=b['billing_start']+600
        while time.time()<deadline:
            heartbeat('WAIT_SSH')
            try:
                remote=connection(provider,b);remote.run(['true'],timeout=20);break
            except Exception:time.sleep(10)
        else:raise TimeoutError('recovery SSH startup exceeded ten minutes')
        root=b['remote_root'];run=root+'/run';code=root+'/code';inputs=b['remote_inputs_root']
        remote.upload(bundle,root+'/network_recovery_patch.tar.gz',timeout=60)
        digest=remote.run(['sha256sum',root+'/network_recovery_patch.tar.gz']).decode().split()[0]
        assert digest==sha256(bundle)
        remote.run(['tar','--no-same-owner','-xzf',root+'/network_recovery_patch.tar.gz','-C',code])
        # Archive all stale stop/heartbeat records before arming either remote process.
        remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-c','from experiment_2d11.recover_network import preserve;import sys;preserve(sys.argv[1])',run])])
        remote.put_json(root+'/binding.json',b)
        sys.path.insert(0,str(REPO/'scripts'))
        from experiment_2d5c_runpod_guard import KeychainCredentialProvider
        credential=KeychainCredentialProvider().read()
        remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-m','experiment_2d11.supervisor','install-watchdog','--run',run,'--binding',root+'/binding.json'])],input=credential)
        credential=b'';heartbeat('VERIFY_RECOVERY')
        verify='''import sys,torch
from pathlib import Path
from experiment_2d11.common import *
from experiment_2d11.budget import project_remaining
r=Path(sys.argv[1]);b=read_json(sys.argv[2]);p=r/'checkpoints/u06000.pt';m=read_json(str(p)+'.manifest.json')
assert sha256(p)==m['sha256'] and p.stat().st_size==m['bytes'] and m['audit']['passed']
c=torch.load(p,map_location='cpu',weights_only=False);assert c['completed_updates']==6000
ledger=c['evaluation_ledger'];pf=read_json(r/'PREFLIGHT.json')
projection=project_remaining(pf['measurements'],time.time()-b['billing_start']+b['prior_billed_seconds'],6000,[int(k.split(':')[1]) for k in ledger if k.startswith('ce:')],[int(k.split(':')[1]) for k in ledger if k.startswith('hella:')])
assert projection['fits']
atomic_json(r/'RECOVERY_CHECKPOINT_VERIFICATION.json',dict(passed=True,sha256=m['sha256'],completed_updates=6000,identities=c['identities'],projection=projection,time=time.time()))
print('CHECKPOINT_VERIFIED_AND_BUDGET_FITS')'''
        print(remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-c',verify,run,root+'/binding.json'])],timeout=120).decode(),flush=True)
        validate=['python3','-m','experiment_2d11.remote_inputs','--data',b['data_root'],'--inputs',inputs,'--output',run+'/RECOVERY_INPUT_VERIFICATION.json']
        remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(validate)],timeout=180)
        assert time.time()<deadline,'recovery preparation exceeded ten minutes'
        heartbeat('START_TRAIN')
        argv=['python3','-m','experiment_2d11.supervisor','run','--run',run,'--binding',root+'/binding.json','--data',b['data_root'],'--initial',inputs+'/initial.pt','--hella',inputs+'/hellaswag.json','--baseline',inputs+'/baseline.pt','--resume',run+'/checkpoints/u06000.pt','--attempt','attempt02']
        start='import subprocess,sys;f=open(sys.argv[1],"ab");p=subprocess.Popen(sys.argv[2:],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True);print(p.pid)'
        pid=int(remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-c',start,run+'/supervisor.log',*argv])]))
        context.update(status='Supervisor launched to resume checkpoint u6000; awaiting verified progress',remote_supervisor_pid=pid,time=time.time());atomic_json(ARCHIVE/'CURRENT_RUN_CONTEXT.json',context)
        print('RESUME_SUPERVISOR_STARTED',pid,flush=True)
        monitor(remote,provider,b,local,run)
    except BaseException as e:
        atomic_json(local/'CONTROLLER_FAILURE.json',dict(time=time.time(),error_type=type(e).__name__,message=str(e)[:250]))
        raise
    finally:
        stop=provider.stop_verified(local/'STOP_VERIFICATION.json')
        atomic_json(local/'BILLING_ACCOUNTING.json',dict(start=b['billing_start'],verified_stop=stop['verified_at'],this_attempt_seconds=stop['verified_at']-b['billing_start'],prior_billed_seconds=prior,cumulative_billed_seconds=stop['verified_at']-b['billing_start']+prior))

if __name__=='__main__':recover()
