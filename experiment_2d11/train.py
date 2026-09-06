"""One four-rank trajectory with autonomous evaluation, reviews, and exports."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import *
from .data import LogicalLoader, physical_batches
from .model import FreshH, fresh_base, optimizer_for, load_g
from .checkpoints import (CheckpointWriter,make_payload,capture_rng,preserve_rng,restore_initial,
                          restore_checkpoint,prune_verified_rolling)
from .evaluate import evaluate_ce,evaluate_hella,merge
from .budget import project_remaining


def training_step(wrapped,model,optimizer,x,y,rank,u,device,world=4,micro_b=32):
    model.train();optimizer.zero_grad(set_to_none=True)
    passes=pass_count(u)
    total=torch.zeros(passes+1,device=device)
    for micro,(xx,yy) in enumerate(physical_batches(x,y,rank,world,micro_b)):
        with wrapped.no_sync() if hasattr(wrapped,'no_sync') and micro<3 else nullcontext():
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss,losses=wrapped(xx.to(device),yy.to(device),num_passes=passes,activation_checkpointing=True)
            (loss/4).backward()
        total[0]+=loss.detach()/4;total[1:]+=losses/4
    # DDP has already averaged gradients exactly once. This reduction is logging only.
    if dist.is_initialized():
        dist.all_reduce(total,op=dist.ReduceOp.SUM)
        total/=dist.get_world_size()
    active=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
    lr=learning_rate(u-1)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step()
    return dict(weighted_multipass_training_loss=total[0].item(),pass_losses=total[1:].tolist(),
                pass_count=passes,lr=lr,lr_index=u-1,grad_norm=norm.item())


def run(args):
    configure_cuda_determinism()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl',timeout=timedelta(seconds=300),device_id=device)
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    binding=read_json(args.binding);preflight=read_json(run/'PREFLIGHT.json')
    assert preflight['passed'] and preflight['projection']['fits']
    assert preflight['config_identity']==identity(read_json(FROZEN/'config.json'))
    deadline=binding['hard_deadline']-1800
    assert time.time()<deadline
    identities={name:sha256(FROZEN/name) for name in ['config.json','shards.json','stream_plan.jsonl',
                'fresh_panel.json','monitor_panel.json','hellaswag_manifest.json','initial_identity.json']}
    identities['implementation_commit']=binding['implementation_commit']
    identities['implementation_patch_sha256']=binding.get('implementation_patch_sha256')
    model=FreshH(fresh_base()).to(device);optimizer=optimizer_for(model,device)
    shards=read_json(FROZEN/'shards.json');loader=LogicalLoader(args.data,shards['train'])
    plan=[json.loads(x) for x in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    if args.resume:
        saved=restore_checkpoint(model,optimizer,args.resume,loader,device,rank,identities)
    else:
        assert sha256(args.initial)==read_json(FROZEN/'initial_identity.json')['sha256']
        saved=restore_initial(model,optimizer,args.initial,loader,device)
    u=saved['completed_updates'];ledger=saved['evaluation_ledger'];resources=saved['resources']
    resources.setdefault('training_seconds',0.);resources.setdefault('evaluation_seconds',0.)
    wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False)
    # DDP construction consumes no intended scientific RNG; reset it explicitly from saved state.
    if args.resume:
        from .checkpoints import restore_rng
        restore_rng(saved['rng_by_rank'][rank],device)
    else:
        restore_initial(model,optimizer,args.initial,loader,device)
    writer=CheckpointWriter() if rank==0 else None
    checkpoints=run/'checkpoints';checkpoints.mkdir(exist_ok=True)
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    ce_panel=read_json(FROZEN/'monitor_panel.json');fresh=read_json(FROZEN/'fresh_panel.json')
    hm=read_json(FROZEN/'hellaswag_manifest.json');examples=read_json(args.hella)
    assert sha256(args.hella)==hm['tokenized_sha256']
    evaluation_dir=run/'evaluations'/args.attempt;evaluation_dir.mkdir(parents=True,exist_ok=True)
    eval_settings=preflight['evaluation_settings']
    validation=Path(args.data)/shards['validation']['filename']

    def sync_value(value):
        objects=[value];dist.broadcast_object_list(objects,src=0);return objects[0]

    def save():
        with preserve_rng(device):
            heartbeat(run,rank,'CHECKPOINT',u,min(deadline+1200,time.time()+300))
            if rank==0:writer.wait(300)
            rngs=[None]*4;dist.all_gather_object(rngs,capture_rng(device))
            if rank==0:
                path=checkpoints/f'u{u:05d}.pt'
                if path.exists():
                    # On exact restart, the existing complete checkpoint is retained.
                    assert read_json(str(path)+'.manifest.json')['audit']['completed_updates']==u
                else:
                    payload=make_payload(model,optimizer,loader,u,ledger,rngs,identities,resources)
                    writer.submit(path,payload)
            dist.barrier()

    def evaluate(kind,key,panel=None,other=None,diagnostics=False):
        if key in ledger:return ledger[key]['score']
        candidate=other if other is not None else model
        with preserve_rng(device):
            budget=preflight['stage_bounds'][kind]
            stage_deadline=min(binding['hard_deadline']-600,time.time()+budget)
            heartbeat(run,rank,'SCHEDULED_EVALUATION',0,stage_deadline,label=key)
            if kind=='ce':
                ids=panel['partition']['owners'][rank]
                part=evaluate_ce(candidate,panel,validation,ids,eval_settings['ce_batch'],device,
                                 run,rank,key,diagnostics,stage_deadline)
                expected=list(range(len(panel['sequences'])))
            else:
                ids=hm['partition']['owners'][rank]
                part=evaluate_hella(candidate,examples,ids,eval_settings['hella_examples'],device,
                                    run,rank,key,stage_deadline)
                expected=list(range(10042))
            filename=key.replace(':','_')
            atomic_json(evaluation_dir/f'{filename}-rank{rank}.json',part)
            dist.barrier()
            summary=None
            if rank==0:
                parts=[read_json(evaluation_dir/f'{filename}-rank{r}.json') for r in range(4)]
                summary=merge(parts,expected,kind)
                summary.update(key=key,completed_updates=u,accounting=accounting(u),
                               checkpoint_base='saved G19072' if other is not None else 'current H',identities=identities,
                               completed_at=time.time())
                result_path=evaluation_dir/f'{filename}.json';atomic_json(result_path,summary)
                summary=dict(score=summary['score'],path=str(result_path),sha256=sha256(result_path),
                             elapsed_seconds=summary['elapsed_seconds'])
            summary=sync_value(summary)
            ledger[key]=summary;resources['evaluation_seconds']+=summary['elapsed_seconds']
            return summary['score']

    def review():
        if u not in (1000,2000):return 'CONTINUE'
        ce=ledger[f'ce:{u}']['score']
        hella=ledger.get(f'hella:{u}',{}).get('score')
        g500=ledger['ce:1000']['score']-REFERENCES[1000]['ce']
        decision=quality_decision(u,ce,hella,g500)
        decision.update(h_ce=ce,g_ce=REFERENCES[u]['ce'],h_hellaswag=hella,
                        g_hellaswag=REFERENCES[u]['hella'],resources=resources,
                        billed_seconds=time.time()-binding['billing_start']+binding['prior_billed_seconds'],
                        revised_projection=project_remaining(preflight['measurements'],
                            time.time()-binding['billing_start']+binding['prior_billed_seconds'],u,
                            ce_done=[int(k.split(':')[1]) for k in ledger if k.startswith('ce:')],
                            hella_done=[int(k.split(':')[1]) for k in ledger if k.startswith('hella:')]))
        decision['aggregate_gpu_hours']=decision['billed_seconds']*4/3600
        if rank==0:
            title='500M' if u==1000 else '1B'
            atomic_json(run/f'INTERIM_{title}.json',decision)
            atomic_bytes(run/f'INTERIM_{title}.md',(f"# Approximately {title} review\n\n"
                f"{decision['outcome']}. Updates {u}; logical targets {TARGETS*u:,}.\n\n"
                f"H CE {ce:.8f}; historical G CE {REFERENCES[u]['ce']:.8f}. "
                f"H HellaSwag {hella}; historical G {REFERENCES[u]['hella']}.\n\n"
                f"CE pass-target evaluations: {accounting(u)['ce_pass_targets']:,}.\n\n{INTERPRETATION}\n").encode())
        return decision['outcome']

    outcome='INVALID_INCOMPLETE'
    try:
        while True:
            if u in CE_SCHEDULE:evaluate('ce',f'ce:{u}',ce_panel)
            severe=u==1000 and quality_decision(u,ledger['ce:1000']['score'])['stop']
            if u in HELLA_SCHEDULE and not severe:evaluate('hella',f'hella:{u}')
            if severe:
                ledger['hella:1000']={'skipped':True,'reason':'severe approximately-500M CE screen'}
            decision=review()
            if u!=ENDPOINT and (u%500==0 or u in PRESERVE):save()
            if decision=='EARLY_STOPPED_FOR_FUTILITY':
                outcome=decision;break
            if u==ENDPOINT:
                evaluate('ce','final:H:ce',fresh,diagnostics=True)
                with preserve_rng(device):
                    g=load_g(args.baseline,device)
                evaluate('ce','final:G:ce',fresh,other=g)
                evaluate('hella','final:G:hella',other=g)
                del g
                save();outcome='COMPLETED';break
            revised=None
            if u and u%500==0:
                revised=project_remaining(preflight['measurements'],time.time()-binding['billing_start']+binding['prior_billed_seconds'],u,
                    [int(k.split(':')[1]) for k in ledger if k.startswith('ce:')],
                    [int(k.split(':')[1]) for k in ledger if k.startswith('hella:')])
                if rank==0:atomic_json(run/'LATEST_PROJECTION.json',dict(completed_updates=u,**revised))
            budget_stop=revised is not None and not revised['fits']
            should_stop=budget_stop or stop_requested or (run/'STOP_REQUEST').exists() or time.time()>=deadline
            flag=torch.tensor(int(should_stop),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if flag.item():
                save();outcome='BUDGET_STOPPED' if budget_stop or time.time()>=deadline or (run/'STOP_REQUEST').exists() else 'USER_STOPPED';break
            started=time.time();u+=1
            x,y,data_identity=loader.next_global(plan[u-1])
            stats=training_step(wrapped,model,optimizer,x,y,rank,u,device)
            elapsed=time.time()-started;resources['training_seconds']+=elapsed
            heartbeat(run,rank,'TRAIN',u,time.time()+300)
            if rank==0:
                writer.poll()
                stats.update(accounting(u),data=data_identity,training_seconds=elapsed,
                             cumulative_training_seconds=resources['training_seconds'],
                             targets_per_second=TARGETS/elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             attempt=args.attempt,timestamp=time.time())
                append_json(run/f'metrics-{args.attempt}.jsonl',stats)
                if u%25==0:
                    deleted=prune_verified_rolling(checkpoints,run/'export_acks')
                    if deleted:append_json(run/'pruned_rolling.jsonl',dict(update=u,deleted=deleted))
        if rank==0:writer.wait(300)
        dist.barrier()
        if rank==0:
            atomic_json(run/'SCIENTIFIC_OUTCOME.json',dict(outcome=outcome,**accounting(u),
                        resources=resources,evaluation_ledger=ledger,identities=identities,
                        final_cursor=loader.state_dict(),terminal_plan_matches=loader.state_dict()==plan[u-1]['after'] if u else True))
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run','binding','data','initial','baseline','hella','attempt']:p.add_argument('--'+name,required=True)
    p.add_argument('--resume');run(p.parse_args())
