"""Disposable four-rank CUDA tests; never carries state into either fresh arm."""
import argparse,copy,gc,importlib,os,time
from datetime import timedelta
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import *
from .model import from_state,optimizer_for,assert_optimizer,tensor_identity
from .checkpoints import reset_original,cpu_snapshot,payload,write,restore,capture_rng,preserve_rng,restore_rng
from .train import step
from .verification import objective_suite,temporal_suite,reference
from .l_verification import objective_suite as l_objective,tensor_errors
from .evaluate import ce,progress
from experiment_2d11.data import physical_batches
from experiment_2d11.model import FreshH
from experiment_2d12.verification import logits,local_incremental_reference

def equal_ranks(m):
    hashes=[None]*4;dist.all_gather_object(hashes,tensor_identity(m.named_parameters()));assert len(set(hashes))==1
    return hashes[0]

def audit_dispatch(fn):
    module=importlib.import_module('torch.optim.adam');calls={}
    names=['_single_tensor_adam','_multi_tensor_adam','_fused_adam']
    from contextlib import ExitStack
    with ExitStack() as stack:
        mocks={n:stack.enter_context(patch.object(module,n,wraps=getattr(module,n))) for n in names}
        result=fn()
        calls={n:p.call_count for n,p in mocks.items()}
    assert calls['_fused_adam']==0 and calls['_multi_tensor_adam']+calls['_single_tensor_adam']==2,calls
    return result,calls

def normalization(m,wrapped,opt,original,x,y,rank,device,arm):
    actual,dispatch=audit_dispatch(lambda:step(wrapped,m,opt,x,y,1,rank,device))
    ga={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    expected_model=cpu_snapshot(m.state_dict());expected_opt=cpu_snapshot(opt.state_dict())
    ref=from_state(original,arm,device);ro=optimizer_for(ref,device);ro.zero_grad(set_to_none=True)
    for xx,yy in physical_batches(x,y,rank):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=(ref.forward_multi_pass(xx.to(device),targets=yy.to(device),num_passes=2,activation_checkpointing=True)['loss'] if arm=='L_nf4' else reference(ref,xx.to(device),yy.to(device),2,True))
        (loss/4).backward()
    for p in ref.parameters():
        if p.requires_grad:dist.all_reduce(p.grad);p.grad/=4
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
    assert abs(norm-actual['grad_norm'])/max(norm,1e-30)<=.01
    dd=aa=bb=ab=0.;per={}
    for n,p in ref.named_parameters():
        if p.requires_grad:
            z=tensor_errors(ga[n],p.grad.cpu());per[n]=z
            dd+=z['diff_sq'];aa+=z['actual_sq'];bb+=z['ref_sq'];ab+=z['dot']
    relative=(dd/max(bb,1e-300))**.5;cosine=ab/max((aa*bb)**.5,1e-300)
    assert relative<=.01 and cosine>=.9999,(relative,cosine)
    ro.step();maximum=0.
    for n,p in ref.state_dict().items():
        maximum=max(maximum,float((p.cpu()-expected_model[n]).abs().max()))
        torch.testing.assert_close(p.cpu(),expected_model[n],atol=2e-6,rtol=2e-5)
    moments={}
    for i,s in ro.state_dict()['state'].items():
        moments[str(i)]={k:tensor_errors(v.cpu(),expected_opt['state'][i][k]) for k,v in s.items()}
    del ref,ro,ga,expected_model,expected_opt;gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,actual=actual,dispatch=dispatch,reference_clip_norm=norm,gradient_relative_l2=relative,gradient_cosine=cosine,per_tensor=per,parameter_max_abs=maximum,optimizer_differences=moments)

def run(a):
    configure_cuda_determinism();rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    assert torch.__version__=='2.8.0+cu128' and torch.cuda.get_device_name()=='NVIDIA A100-SXM4-80GB'
    dist.init_process_group('nccl',timeout=timedelta(seconds=1800),device_id=device)
    root=a.root/'controller/preflight';root.mkdir(parents=True,exist_ok=True);started=time.time();results={}
    xx=((np.arange(512*1024).reshape(512,1024)*17+3)%50304).astype(np.int64);yy=(xx+1)%50304
    for arm in ARMS:
        arm_started=time.time();progress(root/f'rank{rank}',arm,part='objective')
        m,o,p=reset_original(a.original,arm,device);x=torch.as_tensor(xx[:2,:70],device=device);y=torch.as_tensor(yy[:2,:70],device=device)
        graph=(l_objective(m,x,y,True) if arm=='L_nf4' else dict(objective=objective_suite(m,x,y,True),temporal=temporal_suite(m,x,y,True)))
        atomic_json(root/f'{arm}_GRAPH_rank{rank}.json',graph)
        del m,o,p;gc.collect();torch.cuda.empty_cache()
        m,o,p=reset_original(a.original,arm,device);initial=cpu_snapshot(m.state_dict());del p
        wrapped=DDP(m,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False)
        progress(root/f'rank{rank}',arm,part='full_normalization')
        t=time.time();norm=normalization(m,wrapped,o,initial,xx,yy,rank,device,arm);torch.cuda.synchronize()
        norm['seconds']=time.time()-t;norm['rank_model_identity']=equal_ranks(m)
        atomic_json(root/f'{arm}_NORMALIZATION_rank{rank}.json',norm);del initial
        # Prefill disposable counters, then exercise the real two→three-pass boundary.
        for s in o.state.values():s['step'].fill_(30)
        for g in o.param_groups:g['lr']=learning_rate(29)
        progress(root/f'rank{rank}',arm,part='u31_u32_reload')
        performance=[]
        for u in [31]:
            t=time.time();stats=step(wrapped,m,o,xx,yy,u,rank,device);torch.cuda.synchronize();performance.append(dict(update=u,seconds=time.time()-t,stats=stats))
        rngs=[None]*4;dist.all_gather_object(rngs,capture_rng(device))
        plan=[__import__('json').loads(z) for z in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
        path=a.scratch/f'{arm}_u00031.pt'
        if rank==0:write(path,payload(m,o,plan[30]['after'],31,{}, {},dict(disposable=True),rngs))
        dist.barrier();t=time.time();stats=step(wrapped,m,o,xx,yy,32,rank,device);torch.cuda.synchronize();performance.append(dict(update=32,seconds=time.time()-t,stats=stats))
        same_rank=equal_ranks(m);continued_ddp=wrapped._get_ddp_logging_data();expected=cpu_snapshot(m.state_dict());eo=cpu_snapshot(o.state_dict())
        del wrapped,m,o;gc.collect();torch.cuda.empty_cache()
        m,o,p=restore(path,device,rank,arm);wrapped=DDP(m,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False);restore_rng(p['rng_by_rank'][rank],device)
        step(wrapped,m,o,xx,yy,32,rank,device);maximum=0.
        for n,v in m.state_dict().items():
            torch.testing.assert_close(v.cpu(),expected[n],atol=2e-6,rtol=2e-5);maximum=max(maximum,float((v.cpu()-expected[n]).abs().max()))
        for i,s in o.state_dict()['state'].items():
            for k,v in s.items():torch.testing.assert_close(v.cpu(),eo['state'][i][k],atol=2e-6,rtol=2e-5)
        resumed_identity=equal_ranks(m);restored_ddp=wrapped._get_ddp_logging_data()
        recovery_audit=dict(passed=True,parameter_max_abs=maximum,rank_exact=True,uninterrupted_identity=same_rank,resumed_identity=resumed_identity,bitwise_across_restart=resumed_identity==same_rank,atol=2e-6,rtol=2e-5,continued_ddp=continued_ddp,restored_ddp=restored_ddp,prefilled_disposable_counter=30)
        atomic_json(root/f'{arm}_RECOVERY_rank{rank}.json',recovery_audit)
        del wrapped,m,o,p,expected,eo;gc.collect();torch.cuda.empty_cache()
        # Two identical fresh-DDP restores must be bitwise repeatable. Comparing a
        # rebuilt uninterrupted reducer with a new reducer uses frozen resume tolerances.
        m,o,p=restore(path,device,rank,arm);wrapped=DDP(m,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False);restore_rng(p['rng_by_rank'][rank],device)
        step(wrapped,m,o,xx,yy,32,rank,device)
        assert equal_ranks(m)==resumed_identity,'Repeated identical restore is not bitwise deterministic'
        recovery_audit['repeated_restore_bitwise_exact']=True
        atomic_json(root/f'{arm}_RECOVERY_rank{rank}.json',recovery_audit)
        del wrapped,m,o,p;gc.collect();torch.cuda.empty_cache()
        m,o,p=reset_original(a.original,arm,device);del o,p;bench=[]
        ex=torch.as_tensor(xx[:128],device=device);ey=torch.as_tensor(yy[:128],device=device)
        for condition in (['H_ALL_OFF'] if arm=='L_nf4' else ['H_ON','H_ALL_OFF']):
            m.set_condition(condition);m.eval();progress(root/f'rank{rank}',arm,part='B128_incremental',condition=condition)
            torch.cuda.reset_peak_memory_stats();t=time.time();sums,state=ce(m,ex,ey,root/f'rank{rank}',arm,condition=condition)
            torch.cuda.synchronize();seconds=time.time()-t;assert torch.isfinite(sums).all() and m.incremental_cache_audit(state)['passed'];del sums,state
            small=ex[:,:70].contiguous();actual,_=logits(m,small,True)
            changed=small.clone();changed[1:]=(changed[1:]+11)%50304;other,_=logits(m,changed,True);assert torch.equal(actual[0],other[0]);del other
            changed=small.clone();changed[:,35:]=(changed[:,35:]+7)%50304;other,_=logits(m,changed,True);assert torch.equal(actual[:,:35],other[:,:35]);del other
            again,_=logits(m,small,True);assert torch.equal(actual,again);del again
            if condition=='H_ALL_OFF':ref=local_incremental_reference(m,small,True)
            else:
                sealed=FreshH(copy.deepcopy(m.base)).to(device).eval().requires_grad_(False);sealed.load_state_dict(m.state_dict());ref,_=logits(sealed,small,True);del sealed
            assert torch.equal(actual,ref)
            bench.append(dict(condition=condition,seconds=seconds,batch=128,context=1024,peak_allocated=torch.cuda.max_memory_allocated(),reference_exact=True,row_isolation_exact=True,causality_exact=True,cache_reset_exact=True))
            del actual,ref;gc.collect();torch.cuda.empty_cache()
        assert tensor_identity(m.named_parameters())==FULL_INIT_SHA
        result=dict(passed=True,arm=arm,seconds=time.time()-arm_started,normalization=norm,performance=performance,recovery=recovery_audit,inference=bench,scientific_state_not_used=True)
        atomic_json(root/f'{arm}_rank{rank}.json',result);results[arm]=result
        del m,ex,ey;gc.collect();torch.cuda.empty_cache();dist.barrier()
    # Historical H's exact DDP forward and optimizer dispatch, independently of new objectives.
    m,o,p=reset_original(a.original,'R_nf4',device);h=FreshH(copy.deepcopy(m.base)).to(device);h.load_state_dict(m.state_dict());del m,o,p
    ho=optimizer_for(h,device);hw=DDP(h,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False)
    from experiment_2d11.train import training_step
    hs,dispatch=audit_dispatch(lambda:training_step(hw,h,ho,xx,yy,rank,1,device));assert_optimizer(ho,1);hid=equal_ranks(h)
    assert all(results[arm]['normalization']['dispatch']==dispatch for arm in ARMS), 'New-arm optimizer dispatch differs from historical H'
    atomic_json(root/f'H_DISPATCH_rank{rank}.json',dict(passed=True,dispatch=dispatch,stats=hs,rank_identity=hid))
    del hw,h,ho;gc.collect();torch.cuda.empty_cache()
    # Freeze both complete four-rank fresh execution states before L may start.
    b=read_json(a.root/'controller/binding.json')
    for arm in ARMS:
        m,o,original=reset_original(a.original,arm,device)
        rngs=[None]*4;dist.all_gather_object(rngs,capture_rng(device))
        path=a.root/arm/'checkpoints/u00000.pt'
        if rank==0:
            assert not path.exists(),'Scientific initial state is immutable'
            write(path,payload(m,o,original['loader'],0,{},execution_ids(arm),dict(training_seconds=0.,evaluation_seconds=0.,binding=b),rngs))
        dist.barrier();del m,o,original;gc.collect();torch.cuda.empty_cache()
    dist.barrier()
    if rank==0:
        two=results['R_nf4']['performance'][0]['seconds'];three=results['R_nf4']['performance'][1]['seconds'];local=results['L_nf4']['performance'][0]['seconds']
        training=5000*local+4844*two+156*three
        li=results['L_nf4']['inference'][0]['seconds'];on,off=[r['seconds'] for r in results['R_nf4']['inference']]
        evaluation=9*3*(li+on)+3*8*(li+2*on+2*off)
        summary=dict(passed=True,time=time.time(),seconds=time.time()-started,arms=list(ARMS),world_size=4,scientific_updates=0,full_training_shape=[32,1024],accumulation=4,incremental_batch=128,dispatch=dispatch,projection=dict(training_seconds=training,evaluation_seconds=evaluation,estimated_remaining_pod_hours=(training+evaluation+1800)/3600,soft_alert_only=True))
        atomic_json(a.root/'controller/BOTH_ARMS_PREFLIGHT_PASSED.json',summary)
    dist.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','scratch','original']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
