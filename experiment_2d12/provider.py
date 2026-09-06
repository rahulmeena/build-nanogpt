"""Existing Keychain-backed provider interface bound to one GPU and retained volume."""
import sys,time
from .common import *
sys.path.insert(0,str(REPO/'scripts'))
from experiment_2d5c_runpod_guard import RunPodClient
POD='j8lsb4rz1a0mf8';VOLUME='yhzyb27fb5'

def call(argv):return RunPodClient()._call(argv,'2D12 authorized exact-resource operation')

def status():
    p=call(['pod','get',POD,'--include-machine','--include-network-volume','-o','json'])
    assert p['id']==POD and p['gpuCount']==1 and p['networkVolumeId']==VOLUME
    assert p['networkVolume']['size']==190
    return {k:p.get(k) for k in ('id','name','desiredStatus','runtimeStatus','gpuCount','networkVolumeId','costPerHr','machine','networkVolume','lastStatusChange','imageName')}

def stopped(p):return p['desiredStatus']=='EXITED' and p['runtimeStatus']=='stopped'

def stop_verified(path):
    events=[]
    for attempt in range(30):
        try:
            p=status()
            if stopped(p):
                r=dict(passed=True,verified_at=time.time(),provider=p,attempts=events,persistent_volume_retained=True)
                atomic_json(path,r);return r
            call(['pod','stop',POD,'-o','json']);events.append(dict(time=time.time(),stop_requested=True))
        except Exception as e:events.append(dict(time=time.time(),error_type=type(e).__name__))
        atomic_json(path,dict(passed=False,attempts=events));time.sleep(3)
    raise RuntimeError('provider stop not verified')

def stop_reason(now,binding,heartbeat):
    if now>=binding['hard_deadline']-300:return 'independent deadline stop with 5 minute hard-ceiling reserve'
    if heartbeat is None and now-binding['billing_start']>180:return 'controller missing after startup'
    if heartbeat is not None:
        if now-heartbeat['time']>120:return 'controller heartbeat stale'
        if now-heartbeat.get('last_progress',binding['billing_start'])>heartbeat.get('progress_timeout',600):return 'stage progress stalled'
    return None
