"""Exact assigned pod; independent cumulative-budget stop guard."""
import hashlib,marshal,os,sys,time
from .common import *
sys.path.insert(0,str(REPO/'scripts'))
from experiment_2d5c_runpod_guard import RunPodClient
POD='y96gntb89tzuvj';VOLUME='yhzyb27fb5'
def call(argv):return RunPodClient()._call(argv,'2D13 authorized assigned-resource operation')
def status():
    p=call(['pod','get',POD,'--include-machine','--include-network-volume','-o','json'])
    # Capacity can change independently of identity. In particular it must never
    # prevent the assigned pod from reaching the stop command.
    assert p['id']==POD and p['gpuCount']==1
    assert p['networkVolumeId']==VOLUME and p['networkVolume']['id']==VOLUME
    return {k:p.get(k) for k in ('id','name','desiredStatus','runtimeStatus','gpuCount','networkVolumeId','costPerHr','machine','networkVolume','lastStatusChange','imageName')}
def stopped(p):return p['desiredStatus']=='EXITED' and p['runtimeStatus']=='stopped'
def loaded_code_identity():
    """Fingerprint executable objects, not merely potentially newer disk bytes."""
    return {f.__name__:hashlib.sha256(marshal.dumps(f.__code__,2)).hexdigest() for f in (status,stopped,stop,guard)}
def stop(path,reason):
    events=[]
    for _ in range(30):
        try:
            p=status()
            if stopped(p):
                r=dict(passed=True,verified_at=time.time(),provider=p,reason=reason,events=events)
                atomic_json(path,r);return r
            call(['pod','stop',POD,'-o','json']);events.append(dict(time=time.time(),stop_requested=True))
        except Exception as e:events.append(dict(time=time.time(),error_type=type(e).__name__))
        atomic_json(path,dict(passed=False,reason=reason,events=events));time.sleep(3)
    raise RuntimeError('assigned provider stop could not be verified')
def guard(archive):
    b=read_json(archive/'binding.json')
    atomic_json(archive/'GUARD_ARMED.json',dict(pid=os.getpid(),time=time.time(),binding_identity=identity(b),provider_loaded_code=loaded_code_identity()))
    while True:
        if (archive/'STOP_VERIFICATION.json').exists() and read_json(archive/'STOP_VERIFICATION.json').get('passed'):return
        reason=None
        if time.time()>=b['hard_deadline']-300:reason='cumulative billing ceiling, five-minute stop reserve'
        # User explicitly authorized holding during local preparation; deadline still applies.
        h=archive/'CONTROLLER_HEARTBEAT.json'
        if h.exists():
            hb=read_json(h)
            if time.time()-hb['time']>180:reason='active controller heartbeat stale'
            elif time.time()-hb['last_progress']>hb['progress_timeout']:reason='active CUDA stage stalled'
        if reason:
            atomic_json(archive/'GUARD_TRIGGER.json',dict(time=time.time(),reason=reason));stop(archive/'STOP_VERIFICATION.json',reason);return
        time.sleep(3)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['guard']);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();guard(a.archive)
