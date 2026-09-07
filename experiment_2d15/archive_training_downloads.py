"""Bounded historical downloads while scientific training is active.

The existing duplicate archiver performs a second remote/local verification
before any source removal. This helper never controls training or the pod.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .archive_verified_duplicates import REMOTE, SAFE_ROOTS, atomic, sha, run as remove_verified

STREAM = r'''
import sys
wait_training()
before=signature();assert before==b['expected']['signature']
h=hashlib.sha256();total=0;checks=0;first=None;last=None
with p.open('rb') as f:
 while total<before['bytes']:
  start=time.monotonic();training=wait_training()
  if first is None:first=training
  last=training;checks+=1
  chunk=f.read(min(1<<20,before['bytes']-total));assert chunk
  sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
  h.update(chunk);total+=len(chunk)
  time.sleep(max(0,len(chunk)/b['rate_limit']-(time.monotonic()-start)))
assert signature()==before and h.hexdigest()==b['expected']['sha256']
print(json.dumps(dict(bytes=total,sha256=h.hexdigest(),training_checks=checks,
 first_training_check=first,last_training_check=last,source_signature=before)),file=sys.stderr,flush=True)
'''


def run(archive, job):
    os.nice(10)
    binding=json.loads((archive/'binding.json').read_text())
    assert binding['pod_id']=='l55wgmjewejiv2'
    plan=json.loads((job/'PLAN.json').read_text())
    rows=json.loads((job/'CANDIDATES.json').read_text())
    assert [r['path'] for r in rows]==plan['source_scope']
    assert sum(r['bytes'] for r in rows)==plan['planned_bytes']<=4_500_000_000
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-i',binding['ssh_key'],'-p',str(binding['ssh_port']),'root@'+binding['ssh_host']]
    records=job/'downloads';records.mkdir(exist_ok=True)
    for row in rows:
        dest=Path(row['local_candidates'][0])
        assert dest.is_relative_to(archive.parents[2]/'runpod-checkpoint-archive/network-volume-archival-20260907')
        assert not dest.is_symlink()
        key=hashlib.sha256(row['path'].encode()).hexdigest()
        receipt=records/(key+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            assert dest.is_file() and sha(dest)==saved['sha256']
            continue
        assert not dest.exists()
        remaining=sum(r['bytes'] for r in rows if not Path(r['local_candidates'][0]).exists())
        assert shutil.disk_usage(archive).free-remaining>=plan['mac_reserve_bytes']+plan['remaining_scientific_export_allowance_bytes']
        payload=dict(remote_root=binding['remote_root'],path=row['path'],bytes=row['bytes'],safe_roots=sorted(SAFE_ROOTS),operation='inspect')
        expected=json.loads(subprocess.check_output(ssh+['ionice -c 3 nice -n 19 python -'],input=REMOTE.replace('PAYLOAD',repr(payload)).encode()))
        payload.update(expected=expected,rate_limit=plan['rate_limit_bytes_per_second'])
        source=REMOTE.split("receipts=root/")[0].replace('PAYLOAD',repr(payload))+STREAM
        atomic(job/'PROGRESS_DOWNLOAD.json',dict(stage='DOWNLOADING',path=row['path'],time=time.time(),pod_stop_requested=False))
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.downloading')
        assert not temp.exists()
        with (records/(key+'.stderr')).open('wb') as err, temp.open('xb') as out:
            proc=subprocess.Popen(ssh+['ionice -c 3 nice -n 19 python -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
            try:
                proc.stdin.write(source.encode());proc.stdin.close()
                digest=hashlib.sha256();size=0
                while True:
                    chunk=proc.stdout.read(1<<20)
                    if not chunk:break
                    assert shutil.disk_usage(archive).free>=plan['mac_reserve_bytes']+plan['remaining_scientific_export_allowance_bytes']
                    out.write(chunk);digest.update(chunk);size+=len(chunk)
                out.flush();os.fsync(out.fileno())
                assert proc.wait(timeout=60)==0
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        assert size==row['bytes'] and digest.hexdigest()==expected['sha256']
        assert sha(temp)==expected['sha256']
        stream=json.loads((records/(key+'.stderr')).read_text().splitlines()[-1])
        assert stream['bytes']==size and stream['sha256']==expected['sha256']
        os.replace(temp,dest)
        atomic(receipt,dict(time=time.time(),remote_path=row['path'],local_path=str(dest),bytes=size,sha256=expected['sha256'],independent_local_hash_verified=True,stream=stream))
        print(json.dumps(dict(downloaded=row['path'],bytes=size)),flush=True)
    # This separately hashes both copies again and gates every unlink on TRAIN.
    remove_verified(archive,job)
    result=json.loads((job/'SUMMARY.json').read_text())
    result['new_download_bytes']=plan['planned_bytes']
    result['mac_free_after']=shutil.disk_usage(archive).free
    atomic(job/'SUMMARY.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.archive,args.job)
    except Exception as e:
        atomic(args.job/'ERROR_DOWNLOAD.json',dict(time=time.time(),type=type(e).__name__,message=str(e),pod_stop_requested=False));raise
