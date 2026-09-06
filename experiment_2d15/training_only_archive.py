"""User-authorized historical archival, strictly gated by active scientific training."""
import argparse,json,os,shlex,subprocess,time,shutil
from pathlib import Path
from .common import atomic_json,append_json,sha256,read_json

def run(archive,workspace,max_bytes=10_000_000_000):
    b=read_json(archive/'binding.json');assert b['pod_id']=='l55wgmjewejiv2'
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',b['ssh_key'],'-p',str(b['ssh_port']),f"root@{b['ssh_host']}"]
    def remote(code):return subprocess.check_output(ssh+['python -c '+shlex.quote(code)],timeout=180).decode()
    # We consider older checkpoint directories only. Current H inputs, 2D13/14,
    # the dataset and every 2D15 path are categorically protected.
    prefixes=['/workspace/exp2d1_run/','/workspace/exp2d1a_run/','/workspace/exp2d1r_run/','/workspace/exp2d2a_run/','/workspace/exp2d2b_run/','/workspace/exp2d2c_run/','/workspace/exp2d2d_run/','/workspace/exp2d2e_run/']
    guard_source=r"""import pathlib,json,time,hashlib
root=pathlib.Path(ROOT_VALUE)
assert [x.split(b'=',1)[1].decode() for x in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0') if x.startswith(b'RUNPOD_POD_ID=')]==['l55wgmjewejiv2']
def active():
 for arm in ['L_nf4','R_nf4']:
  p=root/arm/'PROGRESS.json'
  if p.exists():
   v=json.loads(p.read_text())
   if v.get('stage')=='TRAIN' and time.time()-v['time']<15 and not (root/arm/'GPU_COMPLETE.json').exists():return True
 return False
def wait_training():
 while not active():
  if (root/'controller/REMOTE_GPU_COMPLETE.json').exists():raise RuntimeError('GPU work completed; background archival stops')
  time.sleep(1)
def guarded_hash(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while True:
   wait_training()
   chunk=f.read(8<<20)
   if not chunk:break
   h.update(chunk)
 return h.hexdigest()
""".replace('ROOT_VALUE',repr(b['remote_root']))
    def training():return remote(guard_source+"\nprint(int(active()))").strip()=='1'
    while not training():
        if (archive/'STOP_VERIFICATION.json').exists():return
        time.sleep(10)
    inventory=json.loads(remote('import pathlib,json; print(json.dumps([dict(path=str(p),bytes=p.stat().st_size) for r in '+repr(prefixes)+' for p in pathlib.Path(r).rglob("*.pt") if p.is_file() and not p.is_symlink()]))'))
    candidates={}
    for root in [workspace/'runpod-checkpoint-archive',workspace/'runpod-backups',workspace/'checkpoint-backups',workspace/'runpod-workspace-recovery']:
        for p in root.rglob('*.pt'):
            if 'experiment_2d15' not in p.parts and p.is_file() and not p.is_symlink():candidates.setdefault((p.name,p.stat().st_size),[]).append(p)
    moved=0;records=[]
    for row in sorted(inventory,key=lambda r:r['bytes']):
        if moved+row['bytes']>max_bytes:continue
        options=candidates.get((Path(row['path']).name,row['bytes']),[])
        if not options:continue # First pass frees verified duplicates without new storage pressure.
        while not training():
            if (archive/'STOP_VERIFICATION.json').exists():return
            time.sleep(10)
        digest=remote(guard_source+'\nprint(guarded_hash('+repr(row['path'])+'))').strip()
        match=next((p for p in options if sha256(p)==digest),None)
        if match is None:continue
        assert str(row['path']).startswith(tuple(prefixes))
        if not training():continue
        # Recheck the source hash immediately before exact-file unlink and archive evidence.
        code=guard_source+"\np=pathlib.Path("+repr(row['path'])+")\nassert guarded_hash(p)=="+repr(digest)+"\nassert p.stat().st_size=="+str(row['bytes'])+"\nwait_training()\np.unlink()\nprint('verified_removed')"
        assert remote(code).strip()=='verified_removed'
        r=dict(time=time.time(),remote_path=row['path'],local_verified_path=str(match),sha256=digest,bytes=row['bytes'],training_active=True)
        append_json(archive/'HISTORICAL_ARCHIVE_MOVES.jsonl',r);records.append(r);moved+=row['bytes']
    atomic_json(archive/'HISTORICAL_ARCHIVE_SUMMARY.json',dict(passed=True,bytes_freed=moved,files=len(records),all_local_hashes_verified=True,scope='older exact duplicate checkpoints only'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();run(a.archive,a.workspace)
