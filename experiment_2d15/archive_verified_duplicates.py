"""Archive exact historical duplicates on the Mac during active training only.

This operational helper is not included in the frozen GPU scientific code.
A durable local intent and independent local SHA precede every remote unlink.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

SAFE_ROOTS = {
    'exp2d1_assets', 'exp2d3a_run', 'exp2d3a_source',
    'exp2d4a_fixed_run', 'exp2d4a_routed_run',
    'exp2d4a_fixed_250m_run', 'exp2d4a_fixed_250m_accepted',
    'exp2d4a_routed_250m_accepted', 'exp2d5c_w2w2_100m',
    'exp2d6_b6_native_100m', 'exp2d7_boundary_alignment',
    'exp2d8_overlap_width', 'exp2d9_dynamic_gating',
    'exp2d9_dynamic_gating_250m', 'exp2d10_h_250m',
    'exp2d10_retrieval_gating_100m',
}

REMOTE = r'''
import hashlib,json,os,pathlib,stat,time
b=PAYLOAD
root=pathlib.Path(b['remote_root'])
assert [v.split(b'=',1)[1].decode() for v in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0') if v.startswith(b'RUNPOD_POD_ID=')]==['l55wgmjewejiv2']
p=pathlib.Path(b['path'])
assert p.is_absolute() and len(p.parts)>3 and p.parts[1]=='workspace' and p.parts[2] in b['safe_roots']
assert p.resolve()==p and not p.is_symlink()
def active():
 try:
  current=json.loads((root/'controller/ACTIVE_PROCESS.json').read_text())
  arm=current['arm'];assert arm in ('L_nf4','R_nf4')
  v=json.loads((root/arm/'PROGRESS.json').read_text())
  if v['stage']!='TRAIN' or not 0<=time.time()-v['time']<15:return None
  if (root/arm/'GPU_COMPLETE.json').exists():return None
  if any(f.stat().st_mtime>=v['time'] for f in (root/arm).glob('FAILURE-rank*.json')):return None
  proc=pathlib.Path('/proc/'+str(current['pid']))
  if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':return None
  return dict(arm=arm,update=v['update'],progress_time=v['time'],checked_at=time.time(),pid=current['pid'])
 except (OSError,ValueError,KeyError,AssertionError):return None
def wait_training():
 while True:
  v=active()
  if v:return v
  if (root/'controller/REMOTE_GPU_COMPLETE.json').exists():raise RuntimeError('Training finished; archival stops')
  time.sleep(1)
def signature():
 s=p.lstat();assert stat.S_ISREG(s.st_mode)
 return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
def digest():
 wait_training();before=signature();h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   wait_training();chunk=f.read(8<<20)
   if not chunk:break
   h.update(chunk)
 assert signature()==before
 return dict(signature=before,sha256=h.hexdigest())
receipts=root/'controller/historical_archive_round2';receipts.mkdir(exist_ok=True)
receipt=receipts/(hashlib.sha256(str(p).encode()).hexdigest()+'.json')
if b['operation']=='inspect':
 result=digest();assert result['signature']['bytes']==b['bytes'];print(json.dumps(result))
else:
 assert b['operation']=='remove' and b['local_proof']['sha256']==b['expected']['sha256']
 if not p.exists():
  result=json.loads(receipt.read_text());assert result['sha256']==b['expected']['sha256'];print(json.dumps(result))
 else:
  assert signature()==b['expected']['signature']
  current=digest();assert current==b['expected']
  training=wait_training();assert signature()==current['signature']
  result=dict(remote_path=str(p),local_verified_path=b['local_proof']['path'],sha256=current['sha256'],bytes=current['signature']['bytes'],training=training,time=time.time())
  intent=receipt.with_suffix('.intent.json');intent.write_text(json.dumps(result,indent=2)+'\n')
  training=wait_training();assert signature()==current['signature'];p.unlink()
  result['training']=training;result['removed_at']=time.time();receipt.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result))
'''

def atomic(path, value):
    temp=path.with_suffix(path.suffix+'.writing')
    with temp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def local_signature(path):
    s=path.stat();return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)

def run(archive,job):
    os.nice(10)
    binding=json.loads((archive/'binding.json').read_text())
    assert binding['pod_id']=='l55wgmjewejiv2'
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-i',binding['ssh_key'],'-p',str(binding['ssh_port']),'root@'+binding['ssh_host']]
    records_dir=job/'records';records_dir.mkdir(exist_ok=True)
    def remote(row,operation,**kwargs):
        payload=dict(remote_root=binding['remote_root'],path=row['path'],bytes=row['bytes'],safe_roots=sorted(SAFE_ROOTS),operation=operation,**kwargs)
        code=REMOTE.replace('PAYLOAD',repr(payload))
        return json.loads(subprocess.check_output(ssh+['ionice -c 3 nice -n 19 python -'],input=code.encode()))
    candidates=json.loads((job/'CANDIDATES.json').read_text())
    cache={};moved=[];skipped=[]
    for row in candidates:
        p=Path(row['path']);assert p.parts[1]=='workspace' and p.parts[2] in SAFE_ROOTS and '..' not in p.parts
        key=hashlib.sha256(row['path'].encode()).hexdigest();record=records_dir/(key+'.json')
        if record.exists():moved.append(json.loads(record.read_text()));continue
        atomic(job/'PROGRESS.json',dict(stage='VERIFYING',remote_path=row['path'],time=time.time(),bytes_freed=sum(r['bytes'] for r in moved)))
        expected=remote(row,'inspect');match=None
        for name in row['local_candidates']:
            local=Path(name)
            assert local.is_relative_to(archive.parents[2]) and '/experiment_2d15/' not in name
            if local.is_symlink() or not local.is_file():continue
            before=local_signature(local)
            if before[0]!=row['bytes']:continue
            digest=cache.get((name,before))
            if digest is None:
                digest=sha(local);assert local_signature(local)==before;cache[(name,before)]=digest
            if digest==expected['sha256']:
                match=dict(path=name,sha256=digest,signature=before);break
        if match is None:
            skipped.append(dict(remote_path=row['path'],reason='No local content hash matched'))
            atomic(job/'SKIPPED.json',skipped);continue
        # Re-read the chosen Mac copy independently; no cached hash authorizes deletion.
        assert local_signature(Path(match['path']))==tuple(match['signature'])
        assert sha(Path(match['path']))==match['sha256']
        assert local_signature(Path(match['path']))==tuple(match['signature'])
        atomic(records_dir/(key+'.intent.json'),dict(row=row,expected=expected,local_proof=match,time=time.time()))
        result=remote(row,'remove',expected=expected,local_proof=match)
        assert result['sha256']==match['sha256']
        assert sha(Path(match['path']))==match['sha256']
        atomic(record,result);moved.append(result)
        atomic(job/'SUMMARY.json',dict(complete=False,files=len(moved),bytes_freed=sum(r['bytes'] for r in moved),skipped=len(skipped)))
        print(json.dumps(result),flush=True)
    # A separate read-only absence check proves every acknowledged source is gone.
    code='import pathlib,json; print(json.dumps({p:pathlib.Path(p).exists() for p in '+repr([r['remote_path'] for r in moved])+'}))'
    absence=json.loads(subprocess.check_output(ssh+['python -c '+shlex.quote(code)]))
    assert not any(absence.values())
    atomic(job/'INDEPENDENT_ABSENCE_CHECK.json',dict(time=time.time(),exists=absence,all_local_hashes_rechecked_after_removal=True))
    atomic(job/'SUMMARY.json',dict(complete=True,files=len(moved),bytes_freed=sum(r['bytes'] for r in moved),skipped=len(skipped),new_download_bytes=0))
    atomic(job/'PROGRESS.json',dict(stage='COMPLETE',time=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.archive,args.job)
    except Exception as e:
        atomic(args.job/'ERROR.json',dict(time=time.time(),type=type(e).__name__,message=str(e),pod_stop_requested=False));raise
