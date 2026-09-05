"""Bounded all-GPU disposable correctness, resume, and workload measurements."""
import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from .common import *
from .model import *
from .data import LogicalLoader,physical_batches
from .checkpoints import *
from .evaluate import ce_batch,hella_batch,historical_g_monitor
from .train import training_step
from .budget import project_remaining


def timed(device,call):
    torch.cuda.synchronize(device);start=time.time();result=call();torch.cuda.synchronize(device)
    return time.time()-start,result


def gradient_equivalence(wrapped,model,x,y,rank,device,passes,tolerance):
    reference=copy.deepcopy(model)
    model.zero_grad(set_to_none=True)
    for micro,(a,b) in enumerate(physical_batches(x,y,rank)):
        with wrapped.no_sync() if micro<3 else nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=wrapped(a.to(device),b.to(device),num_passes=passes)
            (loss/4).backward()
    # All four devices compute the same bounded single-rank accumulation reference.
    reference.zero_grad(set_to_none=True)
    for start in range(0,512,32):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_=reference(torch.from_numpy(x[start:start+32]).to(device),torch.from_numpy(y[start:start+32]).to(device),num_passes=passes)
        (loss/16).backward()
    squares=torch.zeros(3,device=device,dtype=torch.float64);maximum=torch.zeros((),device=device)
    rank_error=torch.zeros((),device=device)
    for (name,p),(name2,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==name2
        if not p.requires_grad:
            assert p.grad is None and q.grad is None;continue
        assert p.grad is not None and q.grad is not None
        delta=(p.grad-q.grad).float()
        squares[0]+=delta.double().square().sum();squares[1]+=q.grad.double().square().sum()
        maximum=torch.maximum(maximum,delta.abs().max())
        identical=p.grad.detach().clone();dist.broadcast(identical,src=0)
        rank_error=torch.maximum(rank_error,(p.grad-identical).abs().max())
    relative=(squares[0]/squares[1].clamp_min(1e-30)).sqrt().item()
    error=maximum.item();agreement=rank_error.item()
    assert relative<=tolerance['ddp_gradient_relative_l2'] and error<=tolerance['ddp_gradient_max_abs']
    assert agreement==0.
    del reference;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    return dict(passes=passes,relative_l2=relative,max_absolute=error,all_rank_max_error=agreement)


class GTraining(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,x,y):return self.base(x,y)[1]


def run(args):
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    assert int(os.environ['WORLD_SIZE'])==4
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl',timeout=timedelta(seconds=180),device_id=device)
    started=time.time();run=Path(args.run);out=run/'preflight';out.mkdir(parents=True,exist_ok=True)
    binding=read_json(args.binding);config=read_json(FROZEN/'config.json');tol=config['tolerances']
    assert torch.__version__=='2.8.0+cu128' and torch.cuda.device_count()==4
    assert all('A100' in torch.cuda.get_device_name(i) and torch.cuda.get_device_properties(i).total_memory>=79*(1<<30) for i in range(4))
    deadline=min(started+1200,binding['hard_deadline']-7200)
    heartbeat(run,rank,'CUDA_PREFLIGHT',0,deadline)
    shards=read_json(FROZEN/'shards.json');loader=LogicalLoader(args.data,shards['train'])
    model=FreshH(fresh_base()).to(device);optimizer=optimizer_for(model,device)
    assert sha256(args.initial)==read_json(FROZEN/'initial_identity.json')['sha256']
    restore_initial(model,optimizer,args.initial,loader,device)
    shared=tensor_identity(model.named_parameters());values=[None]*4;dist.all_gather_object(values,shared)
    assert len(set(values))==1 and shared==read_json(FROZEN/'initial_identity.json')['model_identity']
    wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False)
    before=loader.state_dict();x,y,_=loader.next_global();loader.load_state_dict(before)
    equivalence=[gradient_equivalence(wrapped,model,x,y,rank,device,2,tol),
                 gradient_equivalence(wrapped,model,x,y,rank,device,3,tol)]
    timings=[];resume_error=None;checkpoint_manifest=None
    for u in range(1,34):
        if time.time()>deadline:raise TimeoutError('bounded GPU preflight')
        x,y,_=loader.next_global()
        seconds,stats=timed(device,lambda:training_step(wrapped,model,optimizer,x,y,rank,u,device))
        for r in model.routers.values():
            assert torch.isfinite(r.W1.grad).all() and torch.isfinite(r.W2.grad).all()
            assert (r.W1.grad.eq(0).all() if u==1 else r.W1.grad.abs().sum()>0)
        assert all(p.grad is None and p not in optimizer.state for n,p in model.named_parameters() if not p.requires_grad)
        timings.append(dict(update=u,seconds=seconds,**stats))
        heartbeat(run,rank,'CUDA_PREFLIGHT',u,deadline)
        if u==31:
            rngs=[None]*4;dist.all_gather_object(rngs,capture_rng(device))
            if rank==0:
                payload=make_payload(model,optimizer,loader,31,{},rngs,{'preflight':True},{})
                checkpoint_manifest=write_checkpoint(out/'u00031.pt',payload)
            dist.barrier()
        if u==32:
            expected=cpu_snapshot(model.state_dict());expected_opt=cpu_snapshot(optimizer.state_dict())
            expected_loader=loader.state_dict();expected_rng=capture_rng(device)
            restore_checkpoint(model,optimizer,out/'u00031.pt',loader,device,rank,{'preflight':True})
            xx,yy,_=loader.next_global();assert np.array_equal(x,xx) and np.array_equal(y,yy)
            training_step(wrapped,model,optimizer,xx,yy,rank,32,device)
            errors=[(v-expected[n].to(device)).abs().max() for n,v in model.state_dict().items()]
            resume_error=torch.stack(errors).max().item()
            assert resume_error<=tol['resume_max_abs'] and loader.state_dict()==expected_loader
            for group in optimizer.state_dict()['state']:
                for name,v in optimizer.state_dict()['state'][group].items():
                    target=expected_opt['state'][group][name].to(v.device)
                    assert torch.allclose(v,target,atol=tol['resume_max_abs'],rtol=0)
            del expected,expected_opt
    # A short ordinary GPT-2 reference with the same physical/global batch on four GPUs.
    del wrapped,optimizer;gc.collect();torch.cuda.empty_cache()
    gm=GTraining(fresh_base()).to(device);go=optimizer_for(gm,device)
    gd=DDP(gm,device_ids=[rank],broadcast_buffers=False)
    g_times=[]
    for u in range(3):
        go.zero_grad(set_to_none=True);torch.cuda.synchronize();s=time.time()
        for micro,(a,b) in enumerate(physical_batches(x,y,rank)):
            with gd.no_sync() if micro<3 else nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=gd(a.to(device),b.to(device))
                (loss/4).backward()
        torch.nn.utils.clip_grad_norm_(gm.parameters(),1.,error_if_nonfinite=True);go.step()
        torch.cuda.synchronize();g_times.append(time.time()-s)
    del gd,go;g=IncrementalG(gm.base);del gm
    model.eval();g.eval();torch.cuda.empty_cache()
    ce_times={};hs_times={};g_ce_times={};g_hs_times={}
    validation=np.load(Path(args.data)/shards['validation']['filename'],mmap_mode='r')
    vx=np.asarray(validation[:128*1024],dtype=np.int64).reshape(128,1024)
    vy=np.asarray(validation[1:128*1024+1],dtype=np.int64).reshape(128,1024)
    for b in config['eval_batch_candidates']:
        bx=torch.from_numpy(vx[:b]).to(device);by=torch.from_numpy(vy[:b]).to(device)
        sec,_=timed(device,lambda:ce_batch(model,bx,by));ce_times[b]=sec
        sec,_=timed(device,lambda:ce_batch(g,bx,by));g_ce_times[b]=sec
        heartbeat(run,rank,'CUDA_PREFLIGHT',f'ce-batch-{b}',deadline)
    best_ce=min(ce_times,key=lambda b:ce_times[b]/b)
    examples=read_json(args.hella);ordered=sorted(examples,key=lambda e:e['length'])
    # Fixed length quantiles and the longest examples; no scores influence batch selection.
    samples={}
    for b in config['hella_example_batch_candidates']:
        indices=np.linspace(0,len(ordered)-1,b,dtype=int)
        rows=[ordered[int(i)] for i in indices];samples[b]=rows
        sec,_=timed(device,lambda:hella_batch(model,rows,device));hs_times[b]=sec
        sec,_=timed(device,lambda:hella_batch(g,rows,device));g_hs_times[b]=sec
        heartbeat(run,rank,'CUDA_PREFLIGHT',f'hella-batch-{b}',deadline)
    best_hs=min(hs_times,key=lambda b:hs_times[b]/b)
    # Batching correctness against independent serial-choice/example states.
    small=samples[8][:4]
    batching=[]
    for name,candidate in [('H',model),('G',g)]:
        batched=hella_batch(candidate,small,device)
        serial=torch.cat([hella_batch(candidate,[row],device) for row in small])
        error=(batched-serial).abs().max().item()
        agreement=(batched.argmin(1)==serial.argmin(1)).float().mean().item()
        assert error<=tol['hella_batched_score_abs'] and agreement==tol['hella_prediction_agreement']
        batching.append(dict(model=name,max_score_error=error,prediction_agreement=agreement))
    bx=torch.from_numpy(vx[:4]).to(device);by=torch.from_numpy(vy[:4]).to(device)
    with torch.inference_mode():
        with torch.autocast('cuda',dtype=torch.bfloat16):parallel=g.base(bx)[0]
        state=g.init_incremental_state(4);logits=[]
        for pos in range(1024):
            with torch.autocast('cuda',dtype=torch.bfloat16):z,state=g.incremental_step(bx[:,pos],state)
            logits.append(z)
        incremental=torch.cat(logits,1)
        logit_error=(parallel.float()-incremental.float()).abs().max().item()
        pc=F.cross_entropy(parallel.float().flatten(0,1),by.flatten())
        ic=F.cross_entropy(incremental.float().flatten(0,1),by.flatten())
        ce_error=abs(pc.item()-ic.item())
        assert logit_error<=tol['g_incremental_logits_abs'] and ce_error<=tol['g_incremental_ce_abs']
        assert sum(c.key.numel()*c.key.element_size()*2 for c in state.caches)//4==37711872
    del parallel,incremental,logits;torch.cuda.empty_cache()
    with preserve_rng(device):
        saved_g=load_g(args.baseline,device)
        historical=historical_g_monitor(saved_g,Path(args.data)/shards['validation']['filename'],rank,device)
        atomic_json(out/f'HISTORICAL_G_MONITOR_RANK{rank}.json',historical)
        del saved_g
    # Same B/context/identities, cache priming excluded from timed processing, no collectors.
    inference={}
    for name,candidate in [('G',g),('H',model)]:
        state=candidate.init_incremental_state(4,device=device,dtype=torch.bfloat16)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            prime_start=time.time()
            for pos in range(768):_,state=candidate.incremental_step(bx[:,pos],state)
            torch.cuda.synchronize();prime=time.time()-prime_start;s=time.time()
            for pos in range(768,1024):_,state=candidate.incremental_step(bx[:,pos],state)
            torch.cuda.synchronize();sec=time.time()-s
        inference[name]=dict(batch=4,context=1024,prime_tokens=768,timed_tokens=256,
                             prime_seconds=prime,seconds=sec,tokens_per_second=1024/sec,diagnostics=False,disposable=True)
        if name=='H':assert candidate.incremental_cache_audit(state)['actual_unique_storage_bytes']//4==33289728
    measurements=dict(two_pass_seconds=statistics.median(r['seconds'] for r in timings if r['pass_count']==2 and r['update']>2),
                      three_pass_seconds=next(r['seconds'] for r in timings if r['pass_count']==3),
                      g_update_seconds=statistics.median(g_times[1:]),
                      monitor_ce_seconds=math.ceil(320/best_ce)*ce_times[best_ce],
                      h_fresh_ce_seconds=math.ceil(1024/best_ce)*ce_times[best_ce],
                      g_fresh_ce_seconds=math.ceil(1024/best_ce)*g_ce_times[best_ce],
                      # Longest-length batches bound shorter scientific buckets conservatively.
                      h_hellaswag_seconds=math.ceil(2511/best_hs)*hs_times[best_hs],
                      g_hellaswag_seconds=math.ceil(2511/best_hs)*g_hs_times[best_hs])
    atomic_json(out/f'BENCHMARK_RANK{rank}.json',dict(measurements=measurements,timings=timings,
                gradient_equivalence=equivalence,resume_max_error=resume_error,batching=batching,
                g_incremental=dict(max_logit_error=logit_error,ce_error=ce_error),inference=inference,
                ce_batch_times=ce_times,hellaswag_batch_times=hs_times,g_ce_batch_times=g_ce_times,g_hellaswag_batch_times=g_hs_times,
                evaluation_settings=dict(ce_batch=best_ce,hella_examples=best_hs),
                seconds=time.time()-started,scientific_updates=0,disposable=True))
    dist.barrier()
    if rank==0:
        all_rows=[read_json(out/f'BENCHMARK_RANK{r}.json') for r in range(4)]
        # Freeze ONE common batch size using all ranks, then price that exact choice.
        best_ce=min(config['eval_batch_candidates'],key=lambda b:max(r['ce_batch_times'][str(b)]/b for r in all_rows))
        best_hs=min(config['hella_example_batch_candidates'],key=lambda b:max(r['hellaswag_batch_times'][str(b)]/b for r in all_rows))
        for key in measurements:measurements[key]=max(r['measurements'][key] for r in all_rows)
        hce=max(r['ce_batch_times'][str(best_ce)] for r in all_rows)
        gce=max(r['g_ce_batch_times'][str(best_ce)] for r in all_rows)
        hhs=max(r['hellaswag_batch_times'][str(best_hs)] for r in all_rows)
        ghs=max(r['g_hellaswag_batch_times'][str(best_hs)] for r in all_rows)
        measurements.update(monitor_ce_seconds=math.ceil(320/best_ce)*hce,
            h_fresh_ce_seconds=math.ceil(1024/best_ce)*hce,g_fresh_ce_seconds=math.ceil(1024/best_ce)*gce,
            h_hellaswag_seconds=math.ceil(2511/best_hs)*hhs,g_hellaswag_seconds=math.ceil(2511/best_hs)*ghs)
        checkpoint_manifest=read_json(str(out/'u00031.pt')+'.manifest.json')
        transfer=read_json(run/'TRANSFER_BENCHMARK.json')
        measurements.update(checkpoint_bytes=checkpoint_manifest['bytes'],checkpoint_write_seconds=checkpoint_manifest['write_verify_seconds'],
                            download_bytes_per_second=transfer['bytes_per_second'])
        elapsed=time.time()-binding['billing_start']+binding['prior_billed_seconds']
        projection=project_remaining(measurements,elapsed,ceiling=binding['cumulative_ceiling_seconds'])
        result=dict(passed=projection['fits'],config_identity=identity(config),measurements=measurements,
                    projection=projection,evaluation_settings=dict(ce_batch=best_ce,hella_examples=best_hs),
                    stage_bounds=dict(ce=max(300,measurements['h_fresh_ce_seconds']*2),
                                      hella=max(300,measurements['h_hellaswag_seconds']*2)),
                    seconds=time.time()-started,disposable=True,scientific_updates=0,
                    completed_checks=['B32 two/three-pass backward','all-rank gradients','global-batch reference',
                    'fresh identity','active/inactive gradients','31-to-32 complete resume','G/H batch scoring',
                    'G parallel/incremental','state sizes','same-hardware G/H timing'])
        historical_losses=sorted([v for rank in range(4) for v in read_json(out/f'HISTORICAL_G_MONITOR_RANK{rank}.json')['losses']],key=lambda v:v['logical_batch'])
        acc=torch.zeros((),dtype=torch.float32)
        for v in historical_losses:acc+=torch.tensor(v['loss'],dtype=torch.float32)/20
        result['saved_g_historical_monitor_reproduction']=dict(score=acc.item(),original_float32_accumulation=True)
        atomic_json(run/'PREFLIGHT.json',result)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run','binding','data','initial','hella','baseline']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
