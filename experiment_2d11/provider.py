"""Exact assigned-pod control; credentials remain in Keychain or process memory."""
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from .common import REPO,atomic_json


class Provider:
    def __init__(self,binding):
        self.binding=binding

    def _local_call(self,arguments):
        sys.path.insert(0,str(REPO/'scripts'))
        from experiment_2d5c_runpod_guard import RunPodClient
        return RunPodClient()._call(arguments,'2D11 exact assigned resource operation')

    def _rest(self,method,suffix):
        key=os.environ['RUNPOD_API_KEY']
        request=urllib.request.Request('https://rest.runpod.io/v1/pods/'+self.binding['pod_id']+suffix,
                    method=method,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
                    data=b'{}' if method=='POST' else None)
        with urllib.request.urlopen(request,timeout=20) as response:
            body=response.read()
        return json.loads(body) if body else {}

    def status(self):
        p=(self._rest('GET','') if os.environ.get('EXP2D11_REMOTE_GUARD')=='1' else
           self._local_call(['pod','get',self.binding['pod_id'],'--include-network-volume','-o','json']))
        assert p['id']==self.binding['pod_id'] and p['name']==self.binding['pod_name']
        assert p['gpuCount']==4 and p['networkVolumeId']==self.binding['volume_id']
        return {k:p.get(k) for k in ('id','name','desiredStatus','runtimeStatus','runtime','gpuCount',
                'networkVolumeId','lastStatusChange','costPerHr','createdAt','networkVolume')}

    def stopped(self,p):
        return p['desiredStatus']=='EXITED' and (p.get('runtimeStatus')=='stopped' or
                (os.environ.get('EXP2D11_REMOTE_GUARD')=='1' and not p.get('runtime')))

    def stop(self):
        p=self.status()
        if self.stopped(p):return
        if os.environ.get('EXP2D11_REMOTE_GUARD')=='1':self._rest('POST','/stop')
        else:self._local_call(['pod','stop',self.binding['pod_id'],'-o','json'])

    def start(self):
        p=self.status()
        assert self.stopped(p), 'pod is already running'
        assert p['costPerHr']<=self.binding['max_total_pod_price_per_hour']
        return self._local_call(['pod','start',self.binding['pod_id'],'-o','json'])

    def stop_verified(self,output,retry_delay=10,max_attempts=None):
        attempts=[]
        while max_attempts is None or len(attempts)<max_attempts:
            try:
                p=self.status()
                if self.stopped(p):
                    result=dict(passed=True,pod=p,verified_at=time.time(),attempts=attempts,
                                persistent_volume_retained=True)
                    atomic_json(output,result);return result
                self.stop();attempts.append(dict(time=time.time(),stop_requested=True))
            except Exception as e:
                attempts.append(dict(time=time.time(),error_type=type(e).__name__))
            atomic_json(output,dict(passed=False,alert='POD BILLING STOP NOT VERIFIED',attempts=attempts))
            time.sleep(retry_delay)
        raise RuntimeError('POD BILLING STOP NOT VERIFIED')
