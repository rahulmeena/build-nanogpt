"""Bounded disposable CUDA checks; restores scientific u0 afterward."""
import gc,signal,time
import numpy as np
import torch
from .common import *
from .model import load_h,from_state,optimizer_for,tensor_identity
from .checkpoints import restore,payload,write,cpu_snapshot,capture_rng,restore_rng
from .train import step
from .verification import objective_suite,accumulation,tensor_errors
from .evaluate import ce,progress
from experiment_2d12.verification import focused,logits,local_incremental_reference
from experiment_2d11.model import FreshH
import copy

def run(initial,hpath,root,binding):
    t=time.time()
    def timed_out(signum,frame):raise TimeoutError('bounded CUDA preflight exceeded 600 seconds')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(600);torch.cuda.reset_peak_memory_stats()
    m,o,p=restore(initial,'cuda');assert all(g['fused'] is True for g in o.param_groups) and not o.state
    initial_tensors=tensor_identity(m.named_parameters())
    assert tensor_identity(m.base.named_parameters())==BASE_SHA
    x=((torch.arange(32*1024,device='cuda').reshape(32,1024)*17+3)%50304).long();y=(x+1)%50304
    objective=objective_suite(m,x,y,True,lambda n,c,p:progress(root,'CUDA_PREFLIGHT',part='objective',noninitial=n,checkpointing=c,passes=p))
    atomic_json(root/'CUDA_OBJECTIVE.json',objective)
    # Full 512-sequence global normalization through real B32 entrypoint.
    xx=((torch.arange(512*1024,device='cuda').reshape(512,1024)*17+3)%50304).long();yy=(xx+1)%50304
    progress(root,'CUDA_PREFLIGHT',part='accumulation')
    acc=accumulation(m,xx,yy,True);atomic_json(root/'CUDA_ACCUMULATION.json',acc)
    # Complete updates and save/reload across 715 -> 716 LR boundary.
    m,o,p=restore(initial,'cuda',verify=False);perf=[]
    for u in [1,2,3]:
        torch.cuda.synchronize();begin=time.time();stats=step(m,o,xx,yy,u,progress=lambda i:progress(root,'CUDA_PREFLIGHT',part='update benchmark',update=u,micro=i));torch.cuda.synchronize()
        perf.append(dict(update=u,seconds=time.time()-begin,stats=stats,peak_allocated=torch.cuda.max_memory_allocated()))
    for s in o.state.values():s['step'].fill_(714)
    for g in o.param_groups:g['lr']=learning_rate(713)
    step(m,o,xx,yy,715)
    plan=[__import__('json').loads(z) for z in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    saved=payload(m,o,plan[714]['after'],715,{},p['identities'],dict(disposable=True),'cuda')
    path=root/'preflight_checkpoint.pt';write(path,saved);del saved
    step(m,o,xx,yy,716);expected=cpu_snapshot(m.state_dict());expected_opt=cpu_snapshot(o.state_dict())
    del m,o;gc.collect();torch.cuda.empty_cache()
    m,o,saved=restore(path,'cuda');step(m,o,xx,yy,716)
    errors={}
    for n,v in m.state_dict().items():
        ref=expected[n].to(v.device);torch.testing.assert_close(v,ref,atol=2e-6,rtol=2e-5);errors[n]=float((v-ref).abs().max())
    for i,s in o.state_dict()['state'].items():
        for k,v in s.items():torch.testing.assert_close(v.cpu(),expected_opt['state'][i][k],atol=2e-6,rtol=2e-5)
    atomic_json(root/'CUDA_RESUME.json',dict(passed=True,boundary=[715,716],max_parameter_error=max(errors.values()),per_tensor=errors,optimizer_state_verified=True,synthetic_prefilled_step_counters=714,disposable=True))
    del m,o,saved,expected,expected_opt,xx,yy;gc.collect();torch.cuda.empty_cache()
    atomic_json(root/'PREFLIGHT_CHECKPOINT_DISCARD.json',dict(time=time.time(),sha256=sha256(path),reason='Disposable save/reload fixture passed; no scientific state stored here. Manifest and numerical audit retained.'))
    path.unlink()
    m,o,p=restore(initial,'cuda',verify=False);del o
    progress(root,'CUDA_PREFLIGHT',part='independent references')
    # Inherited focused fixture exercises actual production-width kernels with disposable tokens.
    structural=focused(m,device='cuda',bf16=True);atomic_json(root/'CUDA_STRUCTURAL.json',structural)
    m.set_condition('H_ALL_OFF');m.eval()
    h=load_h(hpath,'cuda');evaluators=[]
    selected_batch=None
    for batch in [128,64,32]:
        try:
            for label,model in [('L',m),('H',h)]:
                progress(root,'CUDA_PREFLIGHT',part='evaluation memory and timing',condition=label,batch=batch)
                ex=((torch.arange(batch*1024,device='cuda').reshape(batch,1024)*19+7)%50304).long();ey=(ex+1)%50304
                torch.cuda.reset_peak_memory_stats();started=time.time()
                sums,state=ce(model,ex,ey,root,'CUDA_PREFLIGHT',part='evaluation benchmark',condition=label,batch=batch)
                torch.cuda.synchronize();elapsed=time.time()-started
                assert torch.isfinite(sums).all() and model.incremental_cache_audit(state)['passed'];del state,sums
                # Identical B and T shape; changing unrelated rows cannot alter row 0.
                small=ex[:,:70].contiguous();sz,_=logits(model,small,True)
                changed=small.clone();changed[1:]=(changed[1:]+11)%50304;zz,_=logits(model,changed,True);assert torch.equal(sz[0],zz[0]);del zz
                if label=='L':reference=local_incremental_reference(model,small,True)
                else:
                    ref=FreshH(copy.deepcopy(model.base)).to('cuda').eval().requires_grad_(False);ref.load_state_dict(model.state_dict());reference,_=logits(ref,small,True);del ref
                assert torch.equal(sz,reference);del sz,reference
                evaluators.append(dict(condition=label,batch=batch,seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),row_isolation_exact=True,independent_reference_exact=True))
                del ex,ey,small,changed
            selected_batch=batch;break
        except torch.cuda.OutOfMemoryError:
            atomic_json(root/'EVAL_BATCH_FALLBACK.json',dict(batch=batch,reason='prescoring memory fit failed'))
            gc.collect();torch.cuda.empty_cache()
    assert selected_batch is not None
    assert tensor_identity(m.named_parameters())==initial_tensors and tensor_identity(h.named_parameters())==H_MODEL_SHA
    # Conservative complete-work estimate including 30-minute one-GPU contingency.
    ltime=next(v['seconds'] for v in reversed(evaluators) if v['condition']=='L');htime=next(v['seconds'] for v in reversed(evaluators) if v['condition']=='H')
    train_seconds=max(v['seconds'] for v in perf[1:])*1000*1.25
    eval_seconds=(ltime*(5*1280+4096)+htime*4096)/selected_batch*1.25
    checkpoint_export_tail=600;contingency=1800
    projected=train_seconds+eval_seconds+checkpoint_export_tail+contingency
    r=dict(passed=True,seconds=time.time()-t,torch_version=torch.__version__,cuda_version=torch.version.cuda,numpy_version=np.__version__,device=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),performance=perf,evaluators=evaluators,batch_size=selected_batch,projection=dict(training_seconds=train_seconds,evaluation_seconds=eval_seconds,checkpoint_export_seconds=checkpoint_export_tail,contingency_seconds=contingency,total_remaining_seconds=projected,remaining_budget_seconds=binding['hard_deadline']-time.time(),fits=time.time()+projected<binding['hard_deadline']-300),scientific_panel_scored=False,scientific_updates=0,disposable_complete_updates=6,objective_probe_optimizer_steps=17,fused_cuda_optimizer_verified=True)
    atomic_json(root/'PREFLIGHT.json',r)
    assert r['projection']['fits'],'conservative full workload does not fit remaining cumulative budget'
    signal.alarm(0);del m,h;gc.collect();torch.cuda.empty_cache();return r
