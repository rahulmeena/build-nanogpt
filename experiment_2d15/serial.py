"""Remote durable L→fresh R scheduling, with no provider credentials or stop API."""
import argparse,os,subprocess,time,traceback
from .common import *
from .state import required_labels

def run(a):
    ctrl=a.root/'controller';assert read_json(ctrl/'BOTH_ARMS_PREFLIGHT_PASSED.json')['passed']
    for arm in ARMS:
        finished=a.root/arm/'GPU_COMPLETE.json'
        if not finished.exists():
            atomic_json(ctrl/'STAGE.json',dict(stage=arm+'_TRAIN_AND_SCORE',time=time.time(),pod_stop_requested=False))
            command=['torchrun','--standalone','--nproc_per_node=4','-m','experiment_2d15.runner','--arm',arm]
            for key in ['root','scratch','original','data','validation','h_dir']:command+=['--'+key.replace('_','-'),str(getattr(a,key))]
            with (ctrl/(arm+'.log')).open('ab') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                atomic_json(ctrl/'ACTIVE_PROCESS.json',dict(pid=child.pid,arm=arm,time=time.time()))
                rc=child.wait()
            if rc:raise RuntimeError(f'{arm} runner exited {rc}; preserve reservation for recovery')
        result=read_json(finished);assert result['passed'] and result['completed_updates']==5000
        assert set(result['evaluation_ledger'])==set(required_labels(arm))
        # Both persistent artifacts and independently checked Mac copies precede handoff.
        while not (ctrl/(arm+'_DURABLY_COMPLETE.json')).exists():time.sleep(10)
        assert read_json(ctrl/(arm+'_DURABLY_COMPLETE.json'))['passed']
    atomic_json(ctrl/'REMOTE_GPU_COMPLETE.json',dict(passed=True,time=time.time(),remaining_gpu_work=0))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','scratch','original','data','validation','h-dir']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        atomic_json(a.root/'controller'/f'FAILURE-{time.time_ns()}.json',dict(time=time.time(),error=str(e),traceback=traceback.format_exc(),pod_stop_requested=False));raise
