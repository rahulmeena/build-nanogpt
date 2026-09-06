"""Bounded CUDA probes on disposable inputs; scientific state restored afterward."""
import copy,gc,signal,time
import torch
from .common import *
from .model import load_h,from_state,tensor_identity
from .checkpoints import restore,cpu_snapshot
from .verification import objective_suite,temporal_suite,recovery
from .train import step
from .evaluate import ce,progress
from experiment_2d11.model import FreshH
from experiment_2d12.verification import logits,local_incremental_reference

def run(initial,hpath,root,binding):
    started=time.time()
    def timeout(*args):raise TimeoutError('CUDA preflight exceeded frozen 600-second bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    torch.cuda.reset_peak_memory_stats()
    m,o,p=restore(initial,'cuda');assert not o.state and tensor_identity(m.named_parameters())==FULL_INIT_SHA
    # Full-width short fixture tests loss/graph variants; complete production updates below.
    x=(torch.arange(2*70,device='cuda').reshape(2,70)*17+3)%50304;y=(x+1)%50304
    objective=objective_suite(m,x,y,True,lambda n,c,p:progress(root,'CUDA_PREFLIGHT',part='objective',noninitial=n,checkpointing=c,passes=p))
    atomic_json(root/'CUDA_OBJECTIVE.json',objective)
    temporal=temporal_suite(m,x,y,True);atomic_json(root/'CUDA_TEMPORAL.json',temporal)
    del m,o,p;gc.collect();torch.cuda.empty_cache()
    m,o,p=restore(initial,'cuda',verify=False)
    xx=((torch.arange(512*1024).reshape(512,1024)*17+3)%50304).long();yy=(xx+1)%50304
    performance=[]
    for u in (1,2,32):
        torch.cuda.synchronize();t=time.time()
        stats=step(m,o,xx,yy,u,progress=lambda i:progress(root,'CUDA_PREFLIGHT',part='full_update',update=u,micro=i))
        torch.cuda.synchronize()
        performance.append(dict(update=u,seconds=time.time()-t,stats=stats,peak_allocated=torch.cuda.max_memory_allocated()))
    atomic_json(root/'CUDA_UPDATE_GEOMETRY.json',dict(passed=True,performance=performance,
        physical_shape=[32,1024],microbatches=16,logical_rows=512,loss_divisor=16,clip_once=True,
        world_size=1,rank_averaging=False,disposable=True))
    del m,o,p;gc.collect();torch.cuda.empty_cache()
    m,o,p=restore(initial,'cuda',verify=False);del o,p
    progress(root,'CUDA_PREFLIGHT',part='recovery_31_to_32')
    resume=recovery(m,xx,yy,'cuda',tiny=False,root=Path('/tmp'))
    atomic_json(root/'CUDA_RESUME.json',resume)
    del m,xx,yy;gc.collect();torch.cuda.empty_cache()
    m,o,p=restore(initial,'cuda',verify=False);del o,p
    h=load_h(hpath,'cuda');benchmarks=[]
    for label,model,condition in [('R_ON',m,'H_ON'),('R_ALL_OFF',m,'H_ALL_OFF'),('H_ON',h,'H_ON'),('H_ALL_OFF',h,'H_ALL_OFF')]:
        model.eval();model.set_condition(condition)
        progress(root,'CUDA_PREFLIGHT',part='B128_inference',condition=label)
        ex=((torch.arange(128*1024,device='cuda').reshape(128,1024)*19+7)%50304).long();ey=(ex+1)%50304
        torch.cuda.reset_peak_memory_stats();t=time.time()
        sums,state=ce(model,ex,ey,root,'CUDA_PREFLIGHT',part='B128_benchmark',condition=label)
        torch.cuda.synchronize();seconds=time.time()-t
        assert torch.isfinite(sums).all() and model.incremental_cache_audit(state)['passed']
        del sums,state
        small=ex[:,:70].contiguous();a,_=logits(model,small,True)
        changed=small.clone();changed[1:]=(changed[1:]+11)%50304;b,_=logits(model,changed,True)
        assert torch.equal(a[0],b[0]);del b
        if condition=='H_ALL_OFF':ref=local_incremental_reference(model,small,True)
        else:
            sealed=FreshH(copy.deepcopy(model.base)).to('cuda').eval().requires_grad_(False)
            sealed.load_state_dict(model.state_dict());ref,_=logits(sealed,small,True);del sealed
        assert torch.equal(a,ref)
        benchmarks.append(dict(label=label,seconds=seconds,batch=128,context=1024,
            peak_allocated=torch.cuda.max_memory_allocated(),reference_exact=True,row_isolation_exact=True))
        del ex,ey,small,changed,a,ref;gc.collect();torch.cuda.empty_cache()
    assert tensor_identity(m.named_parameters())==FULL_INIT_SHA and tensor_identity(h.named_parameters())==H_MODEL_SHA
    two=performance[1]['seconds'];three=performance[2]['seconds'];times={r['label']:r['seconds'] for r in benchmarks}
    training=(969*two+31*three)*1.25
    scoring=(50*times['R_ON']+32*sum(times.values())+32*max(times['H_ALL_OFF'],times['R_ALL_OFF']))*1.25
    # Fifth condition L_LOCAL is included even if its checkpoint is currently unavailable.
    export=read_json(root/'TRANSFER_BENCHMARK.json') if (root/'TRANSFER_BENCHMARK.json').exists() else None
    assert export and export['passed']
    transfer_tail=max(600,2*1652000000/export['download_bytes_per_second'])
    remaining=training+scoring+transfer_tail+1800
    result=dict(passed=True,seconds=time.time()-started,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(),gpu_count=1,batch_size=128,performance=performance,evaluators=benchmarks,
        objective_fixture=[2,70,768],full_update_fixture=[512,1024,768],final_panel_scored=False,scientific_updates=0,
        projection=dict(training_seconds=training,evaluation_seconds=scoring,checkpoint_export_seconds=transfer_tail,
            contingency_seconds=1800,total_remaining_seconds=remaining,remaining_budget_seconds=binding['hard_deadline']-time.time(),
            fits=time.time()+remaining<binding['hard_deadline']-300),disposable=True)
    atomic_json(root/'PREFLIGHT.json',result);assert result['projection']['fits'],'full prescribed workload does not fit ceiling'
    signal.alarm(0);del m,h;gc.collect();torch.cuda.empty_cache();return result
