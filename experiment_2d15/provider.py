"""Exact 2D15 pod binding and completion-only shutdown; no legacy guard imports."""
import os,subprocess,json,time
from .common import *
from .state import complete

def check_binding(b):
    assert b['experiment']=='2D15' and b['pod_id']=='l55wgmjewejiv2' and b['pod_name']=='free_lime_hare'
    assert b['gpu_count']==4 and b['volume_id']=='yhzyb27fb5'
    assert b['remote_root']=='/workspace/exp2d15/'+b['run_id']

def call(args):
    credential=subprocess.run(['/usr/bin/security','find-generic-password','-s','runpod-codex-pod-stopper','-a','rahul','-w'],capture_output=True,check=True).stdout.decode().strip()
    r=subprocess.run(['/opt/homebrew/bin/runpodctl',*args],env=dict(os.environ,RUNPOD_API_KEY=credential),capture_output=True,timeout=45)
    if r.returncode:raise RuntimeError('Assigned RunPod operation failed; credentials and raw provider output suppressed')
    return json.loads(r.stdout)

def status(b):
    check_binding(b);p=call(['pod','get',b['pod_id'],'--include-machine','--include-network-volume','-o','json'])
    assert p['id']==b['pod_id'] and p['name']==b['pod_name'] and p['gpuCount']==4 and p['networkVolumeId']==b['volume_id']
    assert p['machine']['gpuId']=='NVIDIA A100-SXM4-80GB'
    # Capacity is metadata, never an exact equality condition in the stop path.
    return {k:p.get(k) for k in ['id','name','desiredStatus','runtimeStatus','gpuCount','networkVolumeId','costPerHr','networkVolume','lastStatusChange']}

def stop_completed(b,state,path):
    assert complete(state),'Both-arm completion and verified exports required'
    events=[]
    for _ in range(30):
        p=status(b)
        if p['desiredStatus']=='EXITED' and p['runtimeStatus']=='stopped':
            result=dict(passed=True,provider=p,time=time.time(),events=events,binding_identity=identity(b))
            atomic_json(path,result);return result
        call(['pod','stop',b['pod_id'],'-o','json']);events.append(dict(time=time.time(),pod_id=b['pod_id'],stop_requested=True));time.sleep(3)
    raise RuntimeError('Completed assigned pod stop not yet verified')
