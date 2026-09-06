"""Scientific graph and completion-controller regressions on disposable CPU fixtures."""
import copy,tempfile,time
from unittest.mock import Mock,patch
import numpy as np
import torch
from .common import *
from .model import LocalOnly,CE1FreeH,fresh_base,from_state,optimizer_for,assert_optimizer
from .state import complete,maybe_stop,next_stage,required_labels
from .verification import objective_suite,temporal_suite
from .l_verification import objective_suite as l_objective
from .checkpoints import payload,write,validate,capture_rng,cpu_snapshot
from experiment_2d11.data import physical_batches

def controllers():
    done=lambda a:dict(updates=5000,gpu_complete=True,evaluations=required_labels(a),exports_verified=True)
    base=dict(local_ready=True,both_preflights=True,R_original_reset_verified=False,remaining_gpu_work=1)
    cases=[base,dict(base,L_nf4=done('L_nf4')),dict(base,L_nf4=done('L_nf4'),R_nf4=dict(updates=5000)),dict(base,idle=True),dict(base,budget_alert=True),dict(base,error='R incomplete')]
    for s in cases:
        stop=Mock();assert not maybe_stop(s,stop);stop.assert_not_called()
    s=dict(base,L_nf4=done('L_nf4'));assert next_stage(s)=='R_RESET_TO_ORIGINAL_INITIAL_STATE'
    s['R_original_reset_verified']=True;assert next_stage(s)=='R_TRAIN_AND_SCORE'
    s['R_nf4']=done('R_nf4');assert next_stage(s)=='JOINT_GPU_OUTPUTS_VERIFY'
    s.update(remaining_gpu_work=0,joint_identity_coverage_verified=True);stop=Mock();assert maybe_stop(s,stop);stop.assert_called_once()
    return dict(passed=True,incomplete_cases_never_stop=len(cases),resumable_L_R_transition=True,only_combined_predicate_stops=True)

def run():
    started=time.time();torch.set_num_threads(1);x=(torch.arange(140).reshape(2,70)*17+3)%32;y=(x+1)%32;results={}
    for arm in ARMS:
        m=(LocalOnly if arm=='L_nf4' else CE1FreeH)(fresh_base(True));m.arm=arm
        if arm=='L_nf4':
            with patch.object(m,'combine',side_effect=AssertionError('L used router')),patch.object(m,'project_recurrent_kv',side_effect=AssertionError('L used recurrent read')):r=l_objective(m,x,y)
        else:r=dict(objective=objective_suite(m,x,y),temporal=temporal_suite(m,x,y))
        o=optimizer_for(m,'cpu');assert_optimizer(o)
        # Round-trip complete four-rank schema, empty optimizer and exact tied tensors.
        state=payload(m,o,dict(shard=0,position=0,wraps=0,logical_batches=0),0,{}, {},dict(disposable=True),[capture_rng() for _ in range(4)])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'u00000.pt';manifest=write(f,state,tiny=True)
            saved=torch.load(f,map_location='cpu',weights_only=False);validate(saved,True)
            assert saved['arm']==arm and len(saved['rng_by_rank'])==4
        results[arm]=dict(graph=r,checkpoint_roundtrip=manifest['audit'])
    xx=np.repeat(np.arange(512)[:,None],70,axis=1);coverage=[]
    for rank in range(4):
        for micro,(a,b) in enumerate(physical_batches(xx,xx,rank)):
            expected=np.arange((micro*4+rank)*32,(micro*4+rank+1)*32)
            assert np.array_equal(a[:,0],expected);coverage.extend(a[:,0].tolist())
    assert sorted(coverage)==list(range(512))
    result=dict(passed=True,arms=results,controller=controllers(),geometry_exact=True,seconds=time.time()-started)
    atomic_json(PACKAGE/'results/CPU_AUDIT.json',result);print(dict(passed=True,seconds=result['seconds']),flush=True)
if __name__=='__main__':run()
