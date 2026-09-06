"""Incremental bounded transfers, raw backup verification and prompt shutdown."""
import argparse,concurrent.futures,json,os,shlex,subprocess,sys,tarfile,time,traceback,threading
from .common import *
from . import provider
from experiment_2d12.controller import Remote as BaseRemote

class Remote(BaseRemote):
    def transfer(self,source,dest,download=False):
        shell=shlex.join(['ssh',*self.common,'-p',str(self.port)])
        if download:
            Path(dest).parent.mkdir(parents=True,exist_ok=True);source='root@'+self.host+':'+source
        else:dest='root@'+self.host+':'+dest
        for attempt in range(3):
            try:
                p=subprocess.run(['rsync','-a','--partial','--append','--timeout=45','-e',shell,str(source),str(dest)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
                if p.returncode==0:return
            except subprocess.TimeoutExpired:continue
        raise RuntimeError('resumable transfer failed')

def connect():
    return Remote('154.54.102.37',16807,'/Users/rahul/.ssh/id_ed25519')

def inventory(r,root):
    program='''from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);out={}
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.name.endswith('.writing') or '.tmp-' in p.name or p.name=='PROGRESS.json':continue
 if p.suffix not in ('.json','.jsonl','.pt','.log'):continue
 if p.suffix in ('.jsonl','.log') and not (root/'GPU_WORK_FINISHED.json').exists() and not (root/'FAILURE.json').exists():continue
 if p.suffix=='.pt' and p.parent.name!='checkpoints':continue
 if p.suffix=='.pt' and not Path(str(p)+'.manifest.json').exists():continue
 if p.suffix=='.pt':
  h=json.loads(Path(str(p)+'.manifest.json').read_text())['sha256']
 else:h=hashlib.sha256(p.read_bytes()).hexdigest()
 out[str(p.relative_to(root))]={'sha256':h,'bytes':p.stat().st_size}
print(json.dumps(out))'''
    return json.loads(r.run(['python3','-c',program,root],timeout=45))

def export_one(r,root,archive,name,meta):
    dest=archive/'raw'/name
    if dest.exists() and sha256(dest)==meta['sha256']:return meta
    tmp=dest.with_name(dest.name+'.transfer');r.transfer(root+'/'+name,tmp,True)
    assert tmp.stat().st_size==meta['bytes'] and sha256(tmp)==meta['sha256'],name
    tmp.replace(dest);return meta

def export_all(r,root,archive):
    manifest=inventory(r,root)
    for n,v in manifest.items():export_one(r,root,archive,n,v)
    atomic_json(archive/'ONGOING_EXPORT.json',dict(time=time.time(),files=manifest));return manifest

def validate_attachment(r,archive,root,run):
    prior=read_json(archive/'REMOTE_LAUNCH.json')
    assert prior['root']==root
    assert prior['command'][:3]==['python3','-m','experiment_2d13.runner']
    assert prior['command'][prior['command'].index('--run')+1]==run
    pid=int(prior['pid']);assert pid>1
    # Read only: verify the original independent process before taking control.
    check="import pathlib,sys,json; p=pathlib.Path('/proc')/sys.argv[1]; args=(p/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0])); expected=json.loads(sys.argv[2]); assert [x.decode() for x in args]==expected; print('same runner alive')"
    assert r.run(['python3','-c',check,str(pid),json.dumps(prior['command'])]).decode().strip()=='same runner alive'
    return pid

def launch(archive,bundle=None,attach=False):
    binding=read_json(archive/'binding.json');p=provider.status()
    assert p['desiredStatus']=='RUNNING' and p['costPerHr']==binding['rate'] and p['gpuCount']==1
    assert read_json(PACKAGE/'results/LOCAL_READY.json')['passed']
    assert (archive/'GUARD_ARMED.json').exists()
    atomic_json(archive/('PROVIDER_REATTACH.json' if attach else 'PROVIDER_BEFORE_CUDA.json'),dict(time=time.time(),provider=p))
    r=connect();root='/workspace/exp2d13/local_scratch_20260906_attempt02';run=root+'/run'
    if attach:pid=validate_attachment(r,archive,root,run)
    last_progress=time.time()
    lock=threading.Lock();alive=threading.Event();alive.set()
    def hb(stage,last=None,timeout=600):
        with lock:atomic_json(archive/'CONTROLLER_HEARTBEAT.json',dict(time=time.time(),stage=stage,last_progress=last or time.time(),progress_timeout=timeout))
    hb('ATTACHING' if attach else 'STAGING')
    def tick():
        while alive.is_set():
            with lock:
                current=read_json(archive/'CONTROLLER_HEARTBEAT.json');current['time']=time.time();atomic_json(archive/'CONTROLLER_HEARTBEAT.json',current)
            time.sleep(15)
    threading.Thread(target=tick,daemon=True).start()
    try:
        if not attach:
            assert bundle is not None
            hb('STAGING');r.run(['mkdir','-p',root+'/code',root+'/inputs',run])
            bundle_remote=root+'/bundle-'+sha256(bundle)+'.tar.gz'
            r.transfer(bundle,bundle_remote)
            assert r.run(['sha256sum',bundle_remote]).decode().split()[0]==sha256(bundle)
            r.run(['tar','--no-same-owner','--no-same-permissions','-xzf',bundle_remote,'-C',root+'/code'])
            started=time.time()
            initial_path='/workspace/exp2d13/local_scratch_20260906_attempt01/inputs/local_initial.pt'
            assert r.run(['sha256sum',initial_path]).decode().split()[0]==sha256(archive/'local_initial.pt')
            transfer_seconds=time.time()-started
            original_candidates=r.run(['find','/workspace/exp2d11','-maxdepth','5','-name','initial.pt']).decode().splitlines()
            original=None
            for c in original_candidates:
                if r.run(['sha256sum',c],timeout=90).decode().split()[0]==INIT_SHA:original=c;break
            if original is None:
                original=root+'/inputs/original_initial.pt'
                r.transfer(Path('/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/initial.pt'),original)
            hb('INPUT_VERIFICATION')
            space=r.run(['du','-sb','/workspace'],timeout=120).decode().split()[0]
            quota=int(p['networkVolume']['size']*1_000_000_000)
            assert int(space)+8_500_000_000<quota,'persistent volume quota headroom insufficient'
            atomic_json(archive/'STAGING.json',dict(time=time.time(),root=root,original_path=original,local_initial_transfer_seconds=transfer_seconds,persistent_used_bytes=int(space),persistent_quota_bytes=quota))
            r.put_json(root+'/binding.json',binding)
            cmd=['python3','-m','experiment_2d13.runner','--run',run,'--binding',root+'/binding.json','--initial',initial_path,'--original',original,'--h','/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u01000.pt','--data','/workspace/build-nanogpt/edu_fineweb10B','--validation','/workspace/build-nanogpt/edu_fineweb10B/edufineweb_val_000000.npy']
            launcher='''import subprocess,os,sys
root=sys.argv[1]
f=open(root+'/runner.log','ab');p=subprocess.Popen(sys.argv[2:],cwd=root+'/code',stdout=f,stderr=f,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
print(p.pid)'''
            pid=int(r.run(['python3','-c',launcher,root,*cmd]));atomic_json(archive/'REMOTE_LAUNCH.json',dict(time=time.time(),root=root,pid=pid,command=cmd))
        atomic_json(archive/'CONTROLLER_ACTIVE.json',dict(pid=os.getpid(),time=time.time(),mode='attach' if attach else 'launch',remote_pid=pid,root=root,binding_identity=identity(binding),provider_loaded_code=provider.loaded_code_identity()))
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);future=None;previous=None;last_export=0
        while True:
            if time.time()>=binding['hard_deadline']-600:
                r.put_json(run+'/STOP_REQUEST',dict(reason='deadline'));raise TimeoutError('controller export/shutdown reserve reached')
            try:pr=r.json(run+'/PROGRESS.json')
            except Exception:pr={}
            # Evaluation writes under evaluations; independently inspect both.
            try:
                ep=r.json(run+'/evaluations/PROGRESS.json')
                if ep.get('time',0)>pr.get('time',0):pr=ep
            except Exception:pass
            marker={k:v for k,v in pr.items() if k!='time'}
            if marker!=previous:previous=marker;last_progress=time.time()
            hb(pr.get('stage','CUDA_START'),last_progress,600 if pr.get('stage')=='CUDA_PREFLIGHT' else 300)
            atomic_json(archive/'LIVE_PROGRESS.json',pr)
            if future is not None and future.done():future.result();future=None
            if future is None and time.time()-last_export>20:
                future=pool.submit(export_all,r,run,archive);last_export=time.time()
            if (archive/'raw/FAILURE.json').exists():raise RuntimeError('CUDA runner failed; evidence exported')
            if (archive/'raw/GPU_WORK_FINISHED.json').exists():
                if future:future.result()
                manifest=export_all(r,run,archive)
                # Rehash the actual persistent bytes independently, including large checkpoints.
                program='''import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]);names=json.loads(sys.stdin.read());out={}
for name in names:
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 out[name]=h.hexdigest()
print(json.dumps(out))'''
                hashes=json.loads(r.run(['python3','-c',program,run],input=json.dumps(list(manifest)).encode(),timeout=180))
                assert all(hashes[n]==v['sha256']==sha256(archive/'raw'/n) for n,v in manifest.items())
                atomic_json(archive/'EXPORT_VERIFICATION.json',dict(passed=True,time=time.time(),persistent_root=run,volume_id=provider.VOLUME,files=manifest,independent_persistent_rehash=True,independent_local_rehash=True))
                break
            time.sleep(5)
        pool.shutdown(wait=True)
    except BaseException as e:
        atomic_json(archive/'CONTROLLER_FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        try:export_all(r,run,archive)
        except Exception:pass
        raise
    finally:
        stopped=provider.stop(archive/'STOP_VERIFICATION.json','experiment finished or bounded failure; preserve volume')
        alive.clear()
        secs=stopped['verified_at']-binding['billing_start']
        atomic_json(archive/'RUNTIME_ACCOUNTING.json',dict(cumulative_billed_seconds=secs,gpu_hours=secs/3600,rate=binding['rate'],compute_cost=secs/3600*binding['rate'],within_ceiling=secs<=21600 and secs/3600*binding['rate']<=10,billing_start=binding['billing_start'],verified_stop=stopped['verified_at'],includes_user_held_preparation=True,provider_invoice_available=False,storage_excluded=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--bundle',type=Path);p.add_argument('--attach',action='store_true');a=p.parse_args();p.error('choose --attach or --bundle') if a.attach==bool(a.bundle) else None;launch(a.archive,a.bundle,a.attach)
