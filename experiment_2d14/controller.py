"""Exact-bound local controller, background verified exports, prompt final shutdown."""
import argparse,concurrent.futures,json,os,shlex,subprocess,sys,time,traceback,threading
from .common import *
from . import provider

class Remote:
    def __init__(self,binding):
        self.host=binding['host'];self.port=binding['port']
        self.common=['-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10',
                     '-o','ServerAliveCountMax=3','-i',binding['key']]
    def run(self,args,timeout=60,input=None):
        p=subprocess.run(['ssh',*self.common,'-p',str(self.port),'root@'+self.host,shlex.join([str(a) for a in args])],
            input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        if p.returncode:raise RuntimeError('remote command failed: '+p.stderr.decode(errors='replace')[-2000:])
        return p.stdout
    def json(self,path):return json.loads(self.run(['cat',path]))
    def put_json(self,path,value):
        code="import os,sys;from pathlib import Path;p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.controller-tmp');t.write_bytes(sys.stdin.buffer.read());os.replace(t,p)"
        self.run(['python3','-c',code,path],input=json.dumps(value).encode())
    def transfer(self,source,dest,download=False):
        shell=shlex.join(['ssh',*self.common,'-p',str(self.port)])
        if download:
            Path(dest).parent.mkdir(parents=True,exist_ok=True);source='root@'+self.host+':'+str(source)
        else:dest='root@'+self.host+':'+str(dest)
        for attempt in range(3):
            try:
                p=subprocess.run(['rsync','-a','--partial','--append','--timeout=45','-e',shell,str(source),str(dest)],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
                if p.returncode==0:return
            except subprocess.TimeoutExpired:continue
        raise RuntimeError('bounded resumable transfer failed')

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

def launch(archive,bundle):
    binding=read_json(archive/'binding.json');p=provider.status(binding)
    assert p['desiredStatus']=='RUNNING' and p['costPerHr']==binding['rate']
    assert read_json(PACKAGE/'results/LOCAL_READY.json')['passed']
    assert read_json(archive/'GUARD_ARMED.json')['binding_identity']==identity(binding)
    root=binding['remote_root'];run=root+'/run';r=Remote(binding)
    atomic_json(archive/'PROVIDER_BEFORE_CUDA.json',dict(time=time.time(),provider=p))
    lock=threading.Lock();alive=threading.Event();alive.set()
    def hb(stage,last=None,timeout=600):
        with lock:atomic_json(archive/'CONTROLLER_HEARTBEAT.json',dict(time=time.time(),stage=stage,last_progress=last or time.time(),progress_timeout=timeout))
    hb('STAGING')
    def tick():
        while alive.is_set():
            with lock:
                h=read_json(archive/'CONTROLLER_HEARTBEAT.json');h['time']=time.time();atomic_json(archive/'CONTROLLER_HEARTBEAT.json',h)
            time.sleep(15)
    threading.Thread(target=tick,daemon=True).start()
    finished=False;pool=None
    try:
        # Actual quota and other-run reservation are checked BEFORE new shared-volume writes.
        capacity_source=(PACKAGE/'resources.py').read_text().replace('from .common import *','from pathlib import Path')
        program=capacity_source+"\nimport json,sys\nprint(json.dumps(capacity(Path(sys.argv[1]),Path(sys.argv[2]))))\n"
        l_run='/workspace/exp2d13/local_scratch_20260906_attempt02/run'
        space=json.loads(r.run(['python3','-c',program,root,l_run],timeout=150))
        atomic_json(archive/'CAPACITY_PREFLIGHT.json',space)
        if not space['passed']:raise RuntimeError('shared-volume capacity shortfall: '+str(-space['headroom_bytes']))
        r.run(['mkdir','-p',root+'/code',run,'/tmp/exp2d14-'+binding['run_id']])
        bundle_remote=root+'/bundle-'+sha256(bundle)+'.tar.gz';r.transfer(bundle,bundle_remote)
        assert r.run(['sha256sum',bundle_remote]).decode().split()[0]==sha256(bundle)
        r.run(['tar','--no-same-owner','--no-same-permissions','-xzf',bundle_remote,'-C',root+'/code'])
        initial='/tmp/exp2d14-'+binding['run_id']+'/local_initial.pt'
        if (archive/'TRANSFER_BENCHMARK.json').exists():
            benchmark=read_json(archive/'TRANSFER_BENCHMARK.json')
            assert benchmark['passed'] and benchmark['sha256']==sha256(archive/'local_initial.pt')
            assert r.run(['sha256sum',initial]).decode().split()[0]==benchmark['sha256']
        else:
            t=time.time();r.transfer(archive/'local_initial.pt',initial)
            r.transfer(archive/'local_initial.pt.manifest.json',initial+'.manifest.json')
            assert r.run(['sha256sum',initial]).decode().split()[0]==sha256(archive/'local_initial.pt')
            upload=time.time()-t
            t=time.time();probe=archive/'transfer_probe.pt';r.transfer(initial,probe,True)
            elapsed=time.time()-t;assert sha256(probe)==sha256(archive/'local_initial.pt')
            benchmark=dict(passed=True,bytes=probe.stat().st_size,upload_seconds=upload,download_seconds=elapsed,
                download_bytes_per_second=probe.stat().st_size/elapsed,sha256=sha256(probe))
            probe.unlink();atomic_json(archive/'TRANSFER_BENCHMARK.json',benchmark)
        r.put_json(run+'/TRANSFER_BENCHMARK.json',benchmark)
        r.put_json(root+'/binding.json',binding);r.put_json(root+'/reservation.json',space)
        original='/workspace/exp2d11/scientific_20260905/inputs/initial.pt'
        command=['python3','-m','experiment_2d14.runner','--run',run,'--binding',root+'/binding.json',
            '--initial',initial,'--original',original,'--h','/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u01000.pt',
            '--data','/workspace/build-nanogpt/edu_fineweb10B','--validation','/workspace/build-nanogpt/edu_fineweb10B/edufineweb_val_000000.npy','--l-run',l_run]
        launcher="""import subprocess,os,sys
root=sys.argv[1]
f=open(root+'/runner.log','ab');env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','EXP2D14_LOCAL_STAGE':'/tmp/exp2d14-staging','TMPDIR':'/tmp/exp2d14-staging','XDG_CACHE_HOME':'/tmp/exp2d14-cache'}
os.makedirs(env['TMPDIR'],exist_ok=True)
p=subprocess.Popen(sys.argv[2:],cwd=root+'/code',stdout=f,stderr=f,start_new_session=True,env=env)
print(p.pid)"""
        pid=int(r.run(['python3','-c',launcher,root,*command]));atomic_json(archive/'REMOTE_LAUNCH.json',dict(time=time.time(),root=root,pid=pid,command=command))
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);future=None;previous=None;last_progress=time.time();last_export=0
        hb('CUDA_START')
        while True:
            if time.time()>=binding['hard_deadline']-600:
                r.put_json(run+'/STOP_REQUEST',dict(reason='deadline'));raise TimeoutError('export/shutdown reserve reached')
            try:pr=r.json(run+'/PROGRESS.json')
            except Exception:pr={}
            try:
                ep=r.json(run+'/evaluations/PROGRESS.json')
                if ep.get('time',0)>pr.get('time',0):pr=ep
            except Exception:pass
            marker={k:v for k,v in pr.items() if k!='time'}
            if marker!=previous:previous=marker;last_progress=time.time()
            hb(pr.get('stage','CUDA_START'),last_progress,650 if pr.get('stage')=='CUDA_PREFLIGHT' else 300)
            atomic_json(archive/'LIVE_PROGRESS.json',pr)
            if future is not None and future.done():future.result();future=None
            if future is None and time.time()-last_export>25:
                future=pool.submit(export_all,r,run,archive);last_export=time.time()
            if (archive/'raw/FAILURE.json').exists():raise RuntimeError('CUDA runner failed; evidence exported')
            if (archive/'raw/GPU_WORK_FINISHED.json').exists():
                if future:future.result()
                manifest=export_all(r,run,archive)
                program="""import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]);out={}
for name in json.loads(sys.stdin.read()):
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 out[name]=h.hexdigest()
print(json.dumps(out))"""
                hashes=json.loads(r.run(['python3','-c',program,run],input=json.dumps(list(manifest)).encode(),timeout=180))
                assert all(hashes[n]==v['sha256']==sha256(archive/'raw'/n) for n,v in manifest.items())
                atomic_json(archive/'EXPORT_VERIFICATION.json',dict(passed=True,time=time.time(),persistent_root=run,
                    volume_id=binding['volume_id'],files=manifest,independent_persistent_rehash=True,independent_local_rehash=True))
                finished=True;break
            time.sleep(5)
    except BaseException as e:
        atomic_json(archive/'CONTROLLER_FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        try:export_all(r,run,archive)
        except Exception:pass
        raise
    finally:
        if pool:pool.shutdown(wait=True)
        if finished or time.time()>=binding['hard_deadline']-600:
            stop=provider.stop(binding,archive/'STOP_VERIFICATION.json','GPU work and verified exports finished, or billing ceiling')
            seconds=stop['verified_at']-binding['billing_start']
            atomic_json(archive/'RUNTIME_ACCOUNTING.json',dict(cumulative_billed_seconds=seconds,gpu_hours=seconds/3600,
                rate=binding['rate'],compute_cost=seconds/3600*binding['rate'],within_ceiling=seconds<=8*3600 and seconds/3600*binding['rate']<=15,
                billing_start=binding['billing_start'],verified_stop=stop['verified_at'],includes_user_held_preparation=True,
                provider_invoice_available=False,storage_excluded=True))
        else:
            # User explicitly requires retaining the assigned GPU while still needed.
            # The independent hard deadline remains armed during bounded local repair.
            hb('LOCAL_REPAIR_RESERVED',timeout=8*3600)
            atomic_json(archive/'REPAIR_HOLD.json',dict(time=time.time(),user_requested_reservation=True,hard_deadline=binding['hard_deadline']))
        alive.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();launch(a.archive,a.bundle)
