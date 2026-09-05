"""Local exact-pod monitor, independent backup and verified stop for 2D10."""
import fcntl
import datetime, hashlib, json, os, pathlib, shlex, subprocess, sys, time
WORKSPACE=pathlib.Path('/Users/rahul/Documents/GPT-2 Enhancement')
REPO=WORKSPACE/'parallel_2d2_master_dev/2d3a_1b'
ARCHIVE=WORKSPACE/'runpod-checkpoint-archive/experiment_2d10_retrieval_aware_gating_100m'
OUT=REPO/'results/experiment_2d10_retrieval_aware_gating_100m'
REMOTE='/workspace/exp2d10_retrieval_gating_100m/run'
POD='nagj1hv18p3z2c'
sys.path.insert(0,str(REPO/'scripts'))
from experiment_2d5c_runpod_guard import RunPodClient
client=RunPodClient()
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-p','10130','root@154.54.102.30']
SCP=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-P','10130']
def write(name,value):
 path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def remote(command,timeout=60):
 return subprocess.run(SSH+[command],capture_output=True,text=True,timeout=timeout,check=True).stdout

def pod():
 p=client._call(['pod','get',POD,'--include-network-volume','-o','json'],'2D10 exact pod status')
 assert p['id']==POD and p['name']=='electrical_aqua_worm' and p['gpuCount']==2 and p['networkVolumeId']=='yhzyb27fb5'
 return {k:p.get(k) for k in ('id','name','gpuCount','desiredStatus','runtimeStatus','createdAt','lastStatusChange','networkVolumeId','volumeMountPath','networkVolume')}
def preflight():
 p=pod();assert p['desiredStatus']=='RUNNING' and p['runtimeStatus']=='running'
 write('STOP_CAPABILITY_PREFLIGHT.json',{'authenticated':True,'pod':p,'exact_stop_command':'runpodctl pod stop '+POD+' -o json','credential_source':'macOS Keychain runpod-codex-pod-stopper; credential never exported','observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps(p),flush=True)
 return p

def backup():
 fetched=ARCHIVE/'gpu_artifacts';fetched.mkdir(exist_ok=True)
 for folder in ['preflight','continuation','panel','training','evaluation']:
  if remote('test -d '+shlex.quote(REMOTE+'/'+folder)+' && echo yes || true').strip():
   subprocess.run(SCP+['-r','root@154.54.102.30:'+REMOTE+'/'+folder,str(fetched)],check=True,timeout=300)
 # Export compact canonical artifacts; leave raw gate scalars outside Git.
 for p in fetched.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.txt'):
   destination=OUT/p.name
   if destination.exists() and destination.read_bytes()!=p.read_bytes():
    (ARCHIVE / ('previous_'+p.name)).write_bytes(destination.read_bytes())
   destination.write_bytes(p.read_bytes())
 for name in ['preflight.log','workflow.log','workflow.status']:
  subprocess.run(SCP+['root@154.54.102.30:'+REMOTE+'/'+name,str(fetched/name)],check=False,timeout=120)
 manifests={}
 for arm in ['T','H']:
  name='scientific_cumulative_001300758528.pt';rpath=REMOTE+'/checkpoints/'+arm+'/'+name
  if not remote('test -f '+shlex.quote(rpath)+' && echo yes || true').strip():continue
  dest=ARCHIVE/arm;dest.mkdir(exist_ok=True)
  for suffix in ['.verification.json','.sha256']:
   subprocess.run(SCP+['root@154.54.102.30:'+rpath+suffix,str(dest/(name+suffix))],check=True,timeout=120)
  verification=json.loads((dest/(name+'.verification.json')).read_text())
  final=dest/name
  if not final.exists():
   tmp=dest/(name+'.transfer');subprocess.run(SCP+['root@154.54.102.30:'+rpath,str(tmp)],check=True,timeout=900)
   assert sha(tmp)==verification['sha256'];tmp.replace(final)
  digest=sha(final)
  remote_digest=remote('sha256sum '+shlex.quote(rpath)).split()[0]
  assert digest==remote_digest==verification['sha256'] and verification['strict_reopen']['passed']
  manifests[arm]={'local_checkpoint':str(final),'persistent_checkpoint':rpath,'sha256':digest,'persistent_sha256':remote_digest,'bytes':final.stat().st_size,'verified_independently':True,'verification':verification}
 write('CHECKPOINT_MANIFESTS.json',manifests)
 print('BACKUPS_VERIFIED '+','.join(manifests),flush=True)
 return manifests

def stop(expected):
 before=pod();assert before['createdAt']==expected['createdAt']
 if before['runtimeStatus']!='stopped':
  client._call(['pod','stop',POD,'-o','json'],'stop exact 2D10 pod')
 for _ in range(60):
  current=pod()
  if current['desiredStatus']=='EXITED' and current['runtimeStatus']=='stopped':
   write('STOP_VERIFICATION.json',{'passed':True,'pod':current,'persistent_volume_retained':current['networkVolumeId']=='yhzyb27fb5','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   print('GPU_STOP_VERIFIED '+POD,flush=True);return
  time.sleep(3)
 write('STOP_VERIFICATION.json',{'passed':False,'pod':current})
 raise RuntimeError('stop not verified')

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='preflight':preflight()
 elif mode=='stop':stop(json.loads((OUT/'STOP_CAPABILITY_PREFLIGHT.json').read_text())['pod'])
 elif mode=='backup':backup()
 elif mode=='monitor':
  lock=(ARCHIVE/'operations/monitor.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:sys.exit(0)
  expected=preflight();outcome='incomplete';error=None;code=None
  try:
   while True:
    try:status=remote('cat '+REMOTE+'/workflow.status 2>/dev/null || true').strip()
    except Exception as e:
     print('SSH_STATUS_RETRY '+type(e).__name__,flush=True);time.sleep(30);continue
    if status:
     code=int(status);break
    time.sleep(30)
   assert code is not None
   manifests=backup()
   if code!=0:raise RuntimeError('GPU workflow exited '+str(code))
   assert set(manifests)=={'T','H'}
   for name in ['S_REAL','D_REAL','T_REAL','H_REAL']:
    assert json.loads((OUT/(name+'.json')).read_text())['passed']
   outcome='gpu_work_complete'
  except BaseException as e:
   error=type(e).__name__+': '+str(e)
   print('WORKFLOW_ERROR '+error,flush=True)
   try:backup()
   except Exception as b:print('RECOVERY_BACKUP_ERROR '+type(b).__name__,flush=True)
  finally:
   stop_error=None
   try:stop(expected)
   except Exception as e:stop_error=type(e).__name__+': '+str(e);print('STOP_ERROR '+stop_error,flush=True)
   write('OPERATIONS_STATUS.json',{'outcome':outcome,'workflow_exit_code':code,'error':error,'stop_error':stop_error})
