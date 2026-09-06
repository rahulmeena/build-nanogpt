"""Independent local deadline guard, incremental exports, stop before analysis."""
import argparse,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from .common import *
from . import provider
from experiment_2d11.controller import Remote as SealedRemote


class Remote(SealedRemote):
    def run(self,argv,timeout=60,input=None):
        p=subprocess.run(['ssh',*self.common,'-p',str(self.port),'root@'+self.host,shlex.join(argv)],
            input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        if p.returncode:
            raise RuntimeError('remote command failed, rc='+str(p.returncode)+'; stderr='+p.stderr.decode(errors='replace')[-3000:])
        return p.stdout


def guard(binding_path,archive):
    # Armed before startup; binding is written before provider start is requested.
    atomic_json(archive/'GUARD_ARMED.json',dict(time=time.time(),pid=os.getpid()))
    while not binding_path.exists():time.sleep(.5)
    binding=read_json(binding_path)
    while True:
        if (archive/'STOP_VERIFICATION.json').exists() and read_json(archive/'STOP_VERIFICATION.json').get('passed'):return
        hb=read_json(archive/'CONTROLLER_HEARTBEAT.json') if (archive/'CONTROLLER_HEARTBEAT.json').exists() else None
        reason=provider.stop_reason(time.time(),binding,hb)
        if reason:
            atomic_json(archive/'GUARD_TRIGGER.json',dict(time=time.time(),reason=reason))
            provider.stop_verified(archive/'GUARD_STOP_VERIFICATION.json');return
        time.sleep(3)


def connect():
    info=provider.call(['ssh','info',provider.POD,'-o','json']);info=info.get('ssh',info)
    if isinstance(info,dict):command=info.get('command') or info.get('sshCommand') or info.get('ssh_command') or info.get('ssh')
    else:command=info
    if isinstance(command,str):
        w=shlex.split(command);host=next(z.split('@',1)[1] for z in w if z.startswith('root@'));port=w[w.index('-p')+1]
    else:host=info.get('host') or info.get('ip');port=info['port']
    return Remote(host,port,'/Users/rahul/.ssh/id_ed25519')


def remote_manifest(remote,root):
    code='''import json,hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]);out={}
for p in sorted(r.rglob('*.json')):
 if p.name=='PROGRESS.json':continue
 out[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out))'''
    return json.loads(remote.run(['python3','-c',code,root],timeout=45))


def export(remote,root,archive):
    manifest=remote_manifest(remote,root)
    for name,digest in manifest.items():
        local=archive/'raw'/name
        if local.exists() and sha256(local)==digest:continue
        temporary=local.with_suffix('.transfer');remote.download(root+'/'+name,temporary,timeout=45)
        assert sha256(temporary)==digest
        temporary.replace(local)
    atomic_json(archive/'ONGOING_EXPORT.json',dict(time=time.time(),files=manifest))
    return manifest


def launch(archive,bundle,user_start=False):
    archive.mkdir(parents=True,exist_ok=True)
    assert read_json(PACKAGE/'results/LOCAL_READY.json')['passed']
    p=provider.status();assert provider.stopped(p) and p['costPerHr']<=1.59
    atomic_json(archive/'PROVIDER_PREFLIGHT.json',dict(time=time.time(),provider=p,stop_capability_verified=read_json(PACKAGE/'results/INITIAL_STOP.json')['passed']))
    initial=read_json(PACKAGE/'results/INITIAL_STOP.json')
    first_start=datetime(2026,9,6,12,6,55,tzinfo=timezone.utc).timestamp()
    initial_seconds=initial['verified_at']-first_start
    histories=[read_json(p) for p in sorted((archive/'attempts').glob('*/RUNTIME_ACCOUNTING.json'))]
    prior=initial_seconds+sum(h['evaluation_attempt_billed_seconds'] for h in histories)
    prior_intervals=[h['intervals'][-1] for h in histories]
    binding_path=archive/'binding.json'
    assert not binding_path.exists(),'do not overwrite prior attempt budget'
    with (archive/'guard.log').open('ab') as log:
        subprocess.Popen([sys.executable,'-m','experiment_2d12.controller','guard','--archive',str(archive)],stdout=log,stderr=log,start_new_session=True)
    for _ in range(40):
        if (archive/'GUARD_ARMED.json').exists():break
        time.sleep(.1)
    assert (archive/'GUARD_ARMED.json').exists()
    if user_start:
        atomic_json(archive/'READY_FOR_USER_START.json',dict(time=time.time(),pod_id=provider.POD,guard_armed=True))
        last_stopped=time.time()
        while True:
            current=provider.status()
            if not provider.stopped(current):break
            last_stopped=time.time();time.sleep(3)
        assert current['costPerHr']<=p['costPerHr']
        start=last_stopped  # Conservative start bound, includes provider polling uncertainty.
    else:start=time.time()
    rate=float(p['costPerHr']);seconds=min(3*3600-prior,(10-prior*rate/3600)/rate*3600)
    binding=dict(pod_id=provider.POD,volume_id=provider.VOLUME,gpu_count=1,quoted_hourly_rate=rate,
        billing_start=start,prior_billed_seconds=prior,hard_deadline=start+seconds,scientific_deadline=start+seconds-600,
        max_gpu_hours=3,max_compute_dollars=10,initial_running_interval=[first_start,initial['verified_at']],previous_attempt_intervals=prior_intervals)
    atomic_json(binding_path,binding)
    def hb(stage,last=None,timeout=600):
        atomic_json(archive/'CONTROLLER_HEARTBEAT.json',dict(time=time.time(),stage=stage,last_progress=last or time.time(),progress_timeout=timeout))
    remote=None;root=f'/workspace/exp2d12/h10b_ablation_20260906_attempt_{len(histories)+1:02d}';raw=root+'/raw'
    try:
        hb('STARTUP')
        if not user_start:provider.call(['pod','start',provider.POD,'-o','json'])
        for _ in range(30):
            try:
                remote=connect();remote.run(['true'],timeout=20);break
            except Exception:
                hb('STARTUP',start,300);time.sleep(5)
        else:raise RuntimeError('SSH startup deadline')
        hb('STAGING');remote.run(['mkdir','-p',root+'/code',raw])
        remote.upload(bundle,root+'/bundle.tar.gz',timeout=60)
        assert remote.run(['sha256sum',root+'/bundle.tar.gz']).decode().split()[0]==sha256(bundle)
        remote.run(['tar','--no-same-owner','--no-same-permissions','-xzf',root+'/bundle.tar.gz','-C',root+'/code'],timeout=60)
        remote.put_json(root+'/binding.json',binding)
        # Resolve actual mounted files and content hashes before model execution.
        code='''import json,hashlib
from pathlib import Path
cp=Path('/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u19072.pt')
val=Path('/workspace/build-nanogpt/edu_fineweb10B/edufineweb_val_000000.npy')
assert cp.is_file() and val.is_file(), 'required existing persistent inputs missing'
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):d.update(b)
 return d.hexdigest()
print(json.dumps({'checkpoint_path':str(cp),'checkpoint_sha256':h(cp),'validation_path':str(val),'validation_sha256':h(val)}))'''
        inputs=json.loads(remote.run(['python3','-c',code],timeout=90));assert inputs['checkpoint_sha256']==CHECKPOINT_SHA and inputs['validation_sha256']==VAL_SHA
        atomic_json(archive/'REMOTE_INPUT_IDENTITIES.json',inputs)
        cmd=['python3','-m','experiment_2d12.evaluate','--output',raw,'--binding',root+'/binding.json','--checkpoint',inputs['checkpoint_path'],'--validation',inputs['validation_path']]
        launch_code='''import subprocess,sys,os
root=sys.argv[1]
f=open(root+'/runner.log','ab')
p=subprocess.Popen(sys.argv[2:],cwd=root+'/code',stdout=f,stderr=f,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
print(p.pid)'''
        pid=int(remote.run(['python3','-c',launch_code,root,*cmd]));atomic_json(archive/'REMOTE_LAUNCH.json',dict(time=time.time(),pid=pid,root=root,command=cmd))
        previous=None;progress_at=time.time();last_export=0
        while True:
            hb('MONITOR',progress_at,600)
            if time.time()>=binding['scientific_deadline']:raise TimeoutError('scientific deadline; preserve outputs and stop')
            try:prog=remote.json(raw+'/PROGRESS.json')
            except Exception:prog={}
            marker=(prog.get('stage'),prog.get('condition'),prog.get('batch_start'),prog.get('position'),prog.get('part'))
            if marker!=previous:previous=marker;progress_at=time.time()
            hb(prog.get('stage','CUDA_START'),progress_at,600 if prog.get('stage')=='CUDA_PREFLIGHT' else 240)
            if time.time()-last_export>25 or prog.get('stage')=='FINISHED':
                manifest=export(remote,raw,archive);last_export=time.time()
                if 'FAILURE.json' in manifest:raise RuntimeError('GPU evaluator failed; diagnostics exported')
                if 'GPU_WORK_FINISHED.json' in manifest:
                    # Independent second remote read/hash pass and local rehash, no bootstrap.
                    second=remote_manifest(remote,raw);assert second==manifest
                    assert all(sha256(archive/'raw'/n)==h for n,h in second.items())
                    from .artifacts import collect
                    collect(archive/'raw',read_json(FROZEN/'PANEL.json'))
                    atomic_json(archive/'EXPORT_VERIFICATION.json',dict(passed=True,time=time.time(),
                        local_archive=str(archive/'raw'),persistent_storage=raw,volume_id=provider.VOLUME,
                        independent_remote_hash_passes=2,files=second))
                    break
            time.sleep(5)
    except BaseException as e:
        atomic_json(archive/'CONTROLLER_FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        if remote is not None:
            try:export(remote,raw,archive)
            except Exception:pass
        raise
    finally:
        stop=provider.stop_verified(archive/'STOP_VERIFICATION.json')
        current=stop['verified_at']-start;cumulative=prior+current
        atomic_json(archive/'RUNTIME_ACCOUNTING.json',dict(initial_billed_seconds=initial_seconds,prior_attempt_billed_seconds=prior-initial_seconds,evaluation_attempt_billed_seconds=current,
            cumulative_billed_seconds=cumulative,cumulative_billed_gpu_hours=cumulative/3600,
            quoted_hourly_rate=rate,compute_cost_at_quoted_rate=cumulative/3600*rate,
            max_gpu_hours=3,max_compute_dollars=10,within_ceiling=cumulative<=10800 and cumulative/3600*rate<=10,
            provider_invoice_available=False,intervals=[binding['initial_running_interval'],*prior_intervals,[start,stop['verified_at']]],
            startup_retry_attempts=len(histories),scientific_retries=0,
            note='Conservative request-to-verified-stop wall time including startup/transfer/stop latency; quoted-rate cost, not provider invoice. Storage excluded.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['guard','launch']);p.add_argument('--archive',type=Path,required=True);p.add_argument('--bundle',type=Path);p.add_argument('--user-start',action='store_true')
    a=p.parse_args()
    if a.mode=='guard':guard(a.archive/'binding.json',a.archive)
    else:launch(a.archive,a.bundle,a.user_start)
