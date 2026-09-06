"""Explicit 2D14 resource binding; no discovery-based mutation or 2D13 imports."""
import argparse,os,sys,time
from .common import *
sys.path.insert(0,str(REPO/'scripts'))
from experiment_2d5c_runpod_guard import RunPodClient
PROTECTED_PODS=frozenset(('y96gntb89tzuvj',))

def check_binding(binding):
    assert binding['experiment']=='2D14' and binding['pod_id'] not in PROTECTED_PODS
    assert binding['gpu_count']==1 and binding['volume_id']=='yhzyb27fb5'
    assert binding['remote_root']=='/workspace/exp2d14/'+binding['run_id']
    assert binding['hard_deadline']<=binding['billing_start']+min(8*3600,15/binding['rate']*3600)

def call(argv):
    return RunPodClient()._call(argv,'authorized exact assigned 2D14 resource operation')

def status(binding):
    check_binding(binding)
    p=call(['pod','get',binding['pod_id'],'--include-machine','--include-network-volume','-o','json'])
    assert p['id']==binding['pod_id'] and p['name']==binding['pod_name']
    assert p['gpuCount']==binding['gpu_count'] and p['networkVolumeId']==binding['volume_id']
    assert p['machine']['gpuId']=='NVIDIA A100-SXM4-80GB'
    assert p['networkVolume']['size']==200 and p['networkVolume']['dataCenterId']=='US-MD-1'
    return {k:p.get(k) for k in ('id','name','desiredStatus','runtimeStatus','gpuCount','networkVolumeId','costPerHr','machine','networkVolume','lastStatusChange','imageName')}

def stopped(p):return p['desiredStatus']=='EXITED' and p['runtimeStatus']=='stopped'

def stop(binding,path,reason):
    check_binding(binding);events=[]
    for _ in range(30):
        try:
            p=status(binding)
            if stopped(p):
                r=dict(passed=True,verified_at=time.time(),provider=p,reason=reason,events=events,
                       binding_identity=identity(binding))
                atomic_json(path,r);return r
            # Mutation ID comes ONLY from the validated bound run, never a list.
            call(['pod','stop',binding['pod_id'],'-o','json'])
            events.append(dict(time=time.time(),stop_requested=True,pod_id=binding['pod_id']))
        except Exception as e:events.append(dict(time=time.time(),error_type=type(e).__name__))
        atomic_json(path,dict(passed=False,events=events,reason=reason));time.sleep(3)
    raise RuntimeError('assigned pod stop not verified')

def guard(archive):
    b=read_json(archive/'binding.json');check_binding(b);status(b)
    atomic_json(archive/'GUARD_ARMED.json',dict(pid=os.getpid(),time=time.time(),binding_identity=identity(b)))
    while True:
        done=archive/'STOP_VERIFICATION.json'
        if done.exists() and read_json(done).get('passed'):return
        reason=None
        if time.time()>=b['hard_deadline']-300:reason='cumulative billing ceiling; five-minute shutdown reserve'
        hb=archive/'CONTROLLER_HEARTBEAT.json'
        if hb.exists():
            h=read_json(hb)
            if h['stage']!='LOCAL_REPAIR_RESERVED':
                if time.time()-h['time']>180:reason='active controller heartbeat stale'
                elif time.time()-h['last_progress']>h['progress_timeout']:reason='active stage stalled'
        # During LOCAL_PREPARATION the user's explicit reservation override applies.
        if reason:
            atomic_json(archive/'GUARD_TRIGGER.json',dict(time=time.time(),reason=reason))
            stop(b,done,reason);return
        time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    guard(p.parse_args().archive)
