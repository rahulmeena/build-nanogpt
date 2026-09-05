"""Local launch/transfer controller. Stops the exact pod before local analysis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from .common import *
from .provider import Provider


class Remote:
    def __init__(self,host,port,key):
        ipaddress.ip_address(host);self.host=host;self.port=int(port);self.key=key
        self.common=['-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=12',
                     '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-i',key]
    def run(self,argv,timeout=60,input=None):
        p=subprocess.run(['ssh',*self.common,'-p',str(self.port),'root@'+self.host,shlex.join(argv)],
                         input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        if p.returncode:raise RuntimeError('remote command failed, rc='+str(p.returncode))
        return p.stdout
    def upload(self,local,remote,timeout=300):
        subprocess.run(['scp','-q',*self.common,'-P',str(self.port),str(local),'root@'+self.host+':'+remote],
                       check=True,timeout=timeout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    def download(self,remote,local,timeout=300):
        Path(local).parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp','-q',*self.common,'-P',str(self.port),'root@'+self.host+':'+remote,str(local)],
                       check=True,timeout=timeout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    def json(self,remote):return json.loads(self.run(['cat',remote]))
    def put_json(self,remote,value):
        program='import sys,os;from pathlib import Path;p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".transfer");t.write_bytes(sys.stdin.buffer.read());t.replace(p)'
        self.run(['python3','-c',program,remote],input=canonical(value)+b'\n')


def connection(provider,binding):
    info=provider._local_call(['ssh','info',binding['pod_id'],'-o','json'])
    if isinstance(info.get('ssh'),dict):info=info['ssh']
    command=info.get('command') or info.get('sshCommand') or info.get('ssh_command') or info.get('ssh')
    if not isinstance(command,str):
        host=info.get('host') or info.get('ip');port=info.get('port')
        if host and port:return Remote(host,port,binding['ssh_key'])
        raise RuntimeError('assigned pod SSH connection not ready')
    words=shlex.split(command)
    host=next(w.split('@',1)[1] for w in words if w.startswith('root@'))
    port=words[words.index('-p')+1]
    return Remote(host,port,binding['ssh_key'])


def inventory(remote,root):
    program='''from pathlib import Path
import json,sys
r=Path(sys.argv[1]);rows=[]
for p in r.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.jsonl','.md','.log','.txt') and p.stat().st_size<100*1024**2:
  s=p.stat();rows.append(dict(path=str(p.relative_to(r)),size=s.st_size,mtime=s.st_mtime))
print(json.dumps(rows))'''
    return json.loads(remote.run(['python3','-c',program,root]))


def export_checkpoint(remote,remote_path,local_path,manifest,run):
    started=time.time();local_path=Path(local_path)
    local_path.parent.mkdir(parents=True,exist_ok=True)
    if not local_path.exists():
        tmp=local_path.with_suffix('.transfer');remote.download(remote_path,tmp,timeout=900)
        if sha256(tmp)!=manifest['sha256']:raise ValueError('independent checkpoint SHA mismatch')
        os.replace(tmp,local_path)
    assert sha256(local_path)==manifest['sha256']
    assert local_path.stat().st_size==manifest['bytes']
    acknowledgment=dict(verified=True,sha256=manifest['sha256'],bytes=manifest['bytes'],
                        local_path=str(local_path),persistent_path=remote_path,time=time.time(),
                        transfer_verify_seconds=time.time()-started)
    atomic_json(str(local_path)+'.verification.json',acknowledgment)
    remote.put_json(run+'/export_acks/'+local_path.name+'.json',acknowledgment)
    return acknowledgment


def monitor(remote,provider,binding,local,run):
    known={};jobs={};acks={};pool=ThreadPoolExecutor(max_workers=1)
    local=Path(local);local.mkdir(parents=True,exist_ok=True)
    last_success=time.time()
    while True:
        atomic_json(local/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state='MONITOR'))
        try:
            for row in inventory(remote,run):
                name=row['path']
                if known.get(name)==(row['size'],row['mtime']):continue
                path=local/name;temp=path.with_suffix(path.suffix+'.download')
                remote.download(run+'/'+name,temp,timeout=120);temp.replace(path)
                known[name]=(row['size'],row['mtime'])
            last_success=time.time()
            remote.put_json(run+'/LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time()))
        except Exception as e:
            append_json(local/'controller_events.jsonl',dict(time=time.time(),event='poll_error',error_type=type(e).__name__))
            p=provider.status()
            if provider.stopped(p):
                atomic_json(local/'STOP_VERIFICATION.json',dict(passed=True,pod=p,verified_at=time.time()))
                return
            if time.time()-last_success>300:
                provider.stop_verified(local/'STOP_VERIFICATION.json');return
            time.sleep(10);continue
        for path in sorted((local/'checkpoints').glob('u*.pt.manifest.json')):
            manifest=read_json(path);name=manifest['file']
            if name not in jobs and name not in acks:
                jobs[name]=pool.submit(export_checkpoint,remote,manifest['persistent_path'],local/'checkpoints'/name,manifest,run)
        for name,job in list(jobs.items()):
            if job.done():
                acks[name]=job.result();del jobs[name]
                atomic_json(local/'CHECKPOINT_EXPORTS.json',acks)
        # Interim publication is local and never blocks the four-GPU queue.
        for title in ('500M','1B'):
            report=local/f'INTERIM_{title}.json'
            rendered=local/f'INTERIM_{title}.published'
            if report.exists() and not rendered.exists():
                from .analysis import interim_plot
                interim_plot(local)
                atomic_bytes(rendered,b'published locally\n')
                print('INTERIM_AVAILABLE '+str(report),flush=True)
        if (local/'PREFLIGHT.json').exists() and not (local/'preflight.published').exists():
            print('PREFLIGHT_AVAILABLE '+str(local/'PREFLIGHT.json'),flush=True)
            atomic_bytes(local/'preflight.published',b'published\n')
        if (local/'GPU_WORK_FINISHED.json').exists() and not jobs:
            outcome=read_json(local/'SCIENTIFIC_OUTCOME.json') if (local/'SCIENTIFIC_OUTCOME.json').exists() else None
            required=[]
            if outcome:
                u=outcome['completed_updates']
                required=[f'u{x:05d}.pt' for x in PRESERVE if x<=u]+[f'u{u:05d}.pt']
            if all(name in acks for name in required):
                verified_artifacts=verify_final_artifacts(remote,run,local)
                verification=dict(passed=True,checkpoints=acks,time=time.time(),outcome=outcome,
                                  independently_hashed_artifacts=verified_artifacts,
                                  small_artifact_inventory=known)
                atomic_json(local/'FINAL_EXPORTS_VERIFIED.json',verification)
                remote.put_json(run+'/FINAL_EXPORTS_VERIFIED.json',verification)
                provider.stop_verified(local/'STOP_VERIFICATION.json')
                return
        if time.time()>=binding['hard_deadline']:
            provider.stop_verified(local/'STOP_VERIFICATION.json');return
        time.sleep(15)



def verify_final_artifacts(remote,run,local):
    # Only immutable science/input records; live watchdog/log/heartbeat files are diagnostics.
    program="""import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[]
for p in r.rglob('*'):
 if not p.is_file():continue
 n=p.name;relative=str(p.relative_to(r))
 selected=(relative.startswith(('evaluations/','preflight/','checkpoints/')) and p.suffix=='.json') or n.startswith(('metrics-','INTERIM_')) or n in ('SCIENTIFIC_OUTCOME.json','PREFLIGHT.json','INPUT_VERIFICATION.json','TRANSFER_BENCHMARK.json')
 if selected:
  b=p.read_bytes();rows.append(dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))"""
    rows=json.loads(remote.run(['python3','-c',program,run],timeout=120))
    for row in rows:
        path=Path(local)/row['path']
        if not path.exists() or sha256(path)!=row['sha256']:
            remote.download(run+'/'+row['path'],path,timeout=120)
        assert path.stat().st_size==row['bytes'] and sha256(path)==row['sha256']
    atomic_json(Path(local)/'SMALL_ARTIFACT_HASH_VERIFICATION.json',dict(passed=True,files=rows,time=time.time()))
    return rows


def local_guard(binding_path,output):
    binding=read_json(binding_path);provider=Provider(binding);output=Path(output)
    while time.time()<binding['hard_deadline']:
        if (output/'STOP_VERIFICATION.json').exists() and read_json(output/'STOP_VERIFICATION.json')['passed']:return
        h=output/'LOCAL_CONTROLLER_HEARTBEAT.json'
        latest=read_json(h)['time'] if h.exists() else binding['billing_start']
        if time.time()-latest>600:break
        time.sleep(15)
    provider.stop_verified(output/'INDEPENDENT_LOCAL_STOP_VERIFICATION.json')


def launch(args):
    ready=read_json(args.ready);assert ready['status']=='READY'
    binding=read_json(args.binding);provider=Provider(binding)
    local=Path(args.archive)/'run';local.mkdir(parents=True,exist_ok=True)
    assert sha256(args.bundle)==ready['bundle_sha256']
    assert read_json(FROZEN/'storage_plan.json')['required_total_decimal_GB']<=binding['authorized_volume_gb']
    assert provider.status()['networkVolume']['size']==binding['authorized_volume_gb']
    binding['billing_start']=time.time()
    remaining=binding['cumulative_ceiling_seconds']-binding['prior_billed_seconds']
    binding['hard_deadline']=binding['billing_start']+remaining
    atomic_json(args.binding,binding);atomic_json(local/'BINDING.json',binding)
    atomic_json(local/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state='STARTUP'))
    log=(local/'local_guard.log').open('ab')
    subprocess.Popen([sys.executable,'-m','experiment_2d11.controller','guard','--binding',str(args.binding),
                      '--archive',str(local)],stdout=log,stderr=log,start_new_session=True)
    remote=None;connected=False;startup_deadline=binding['billing_start']+600
    try:
        provider.start()
        while time.time()<startup_deadline:
            atomic_json(local/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state='STARTUP'))
            try:
                remote=connection(provider,binding);remote.run(['true'],timeout=20);connected=True;break
            except Exception:time.sleep(10)
        if not connected:raise TimeoutError('SSH startup exceeded ten-minute budget')
        root=binding['remote_root'];run=root+'/run';code=root+'/code';inputs=root+'/inputs'
        remote.run(['mkdir','-p',run,code,inputs])
        remote.upload(args.bundle,root+'/bundle.tar.gz')
        remote.run(['tar','--no-same-owner','-xzf',root+'/bundle.tar.gz','-C',code])
        remote.put_json(root+'/binding.json',binding)
        sys.path.insert(0,str(REPO/'scripts'))
        from experiment_2d5c_runpod_guard import KeychainCredentialProvider
        credentials=KeychainCredentialProvider().read()
        command=['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-m','experiment_2d11.supervisor','install-watchdog',
                         '--run',run,'--binding',root+'/binding.json'])]
        remote.run(command,input=credentials);credentials=b''
        source_files=[(args.initial,'initial.pt'),(args.baseline,'baseline.pt'),(args.hella,'hellaswag.json')]
        uploads=[]
        for path,name in source_files:
            before=time.time();remote.upload(path,inputs+'/'+name,timeout=max(60,int(startup_deadline-time.time())))
            uploads.append(dict(file=name,bytes=Path(path).stat().st_size,seconds=time.time()-before))
            atomic_json(local/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state='UPLOAD'))
        assert time.time()<startup_deadline, 'ordinary startup preparation exceeded ten minutes'
        # Independent shutdown is already armed before the full validation/preflight process.
        validate=['python3','-m','experiment_2d11.remote_inputs','--data',binding['data_root'],'--inputs',inputs,'--output',run+'/INPUT_VERIFICATION.json']
        remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(validate)],timeout=max(1,min(180,int(startup_deadline-time.time()))))
        assert time.time()<startup_deadline, 'input verification exceeded startup budget'
        atomic_json(local/'UPLOAD_TIMING.json',uploads)
        # Benchmark actual download bandwidth while GPU preflight begins.
        remote.run(['python3','-c',"from pathlib import Path;import sys;p=Path(sys.argv[1]);f=p.open('rb');Path(sys.argv[2]).write_bytes(f.read(128<<20))",inputs+'/initial.pt',inputs+'/transfer_probe.bin'])
        argv=['python3','-m','experiment_2d11.supervisor','run','--run',run,'--binding',root+'/binding.json',
              '--data',binding['data_root'],'--initial',inputs+'/initial.pt','--hella',inputs+'/hellaswag.json','--baseline',inputs+'/baseline.pt']
        start_program='import subprocess,sys;f=open(sys.argv[1],"ab");p=subprocess.Popen(sys.argv[2:],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True);print(p.pid)'
        remote.run(['bash','-c','cd '+shlex.quote(code)+' && '+shlex.join(['python3','-c',start_program,run+'/supervisor.log',*argv])])
        s=time.time();probe=Path(args.archive)/'transfer_probe.bin';remote.download(inputs+'/transfer_probe.bin',probe,timeout=180)
        seconds=time.time()-s
        transfer=dict(bytes=probe.stat().st_size,seconds=seconds,bytes_per_second=probe.stat().st_size/seconds)
        remote.put_json(run+'/TRANSFER_BENCHMARK.json',transfer);atomic_json(local/'TRANSFER_BENCHMARK.json',transfer);probe.unlink()
        print('AUTONOMOUS_GPU_RUN_STARTED '+run,flush=True)
        monitor(remote,provider,binding,local,run)
    except BaseException as e:
        atomic_json(local/'CONTROLLER_FAILURE.json',dict(time=time.time(),error_type=type(e).__name__,message=str(e)[:250]))
        provider.stop_verified(local/'STOP_VERIFICATION.json')
        raise
    finally:
        if not (local/'STOP_VERIFICATION.json').exists() or not read_json(local/'STOP_VERIFICATION.json').get('passed'):provider.stop_verified(local/'STOP_VERIFICATION.json')
        stopped=read_json(local/'STOP_VERIFICATION.json')
        atomic_json(local/'BILLING_ACCOUNTING.json',dict(start=binding['billing_start'],verified_stop=stopped['verified_at'],
                    this_attempt_seconds=stopped['verified_at']-binding['billing_start'],
                    prior_billed_seconds=binding['prior_billed_seconds'],
                    cumulative_billed_seconds=stopped['verified_at']-binding['billing_start']+binding['prior_billed_seconds']))
    from .analysis import analyze
    analyze(local)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','guard'])
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    for name in ['ready','bundle','initial','baseline','hella']:p.add_argument('--'+name,type=Path)
    a=p.parse_args();local_guard(a.binding,a.archive) if a.mode=='guard' else launch(a)
