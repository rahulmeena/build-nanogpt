"""Freeze untrained tensor provenance, the H 5000-update prefix and a fresh panel."""
import argparse,json,os,time,gc
import numpy as np
import torch
from .common import *
from .model import from_state,tensor_identity,trainability,optimizer_for,optimizer_names
from experiment_2d11.data import array_hash
from experiment_2d12.prepare import recover

def run(workspace):
    torch.set_num_threads(4);started=time.time()
    original=workspace/'runpod-checkpoint-archive/experiment_2d11/initial.pt'
    validation=workspace/'runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy'
    assert sha256(original)==INIT_SHA and original.stat().st_size==498868405
    assert sha256(validation)==VAL_SHA
    p=torch.load(original,map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==0 and not p['optimizer']['state']
    groups=[{k:v for k,v in g.items() if k!='params'} for g in p['optimizer']['param_groups']]
    records=[]
    for u in MILESTONES:
        f=workspace/f'runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u{u:05d}.pt'
        assert sha256(f)==H_CHECKPOINTS[u][0]
        h=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
        assert h['completed_updates']==u and h['world_size']==4 and len(h['rng_by_rank'])==4
        opts=[{k:v for k,v in g.items() if k!='params'} for g in h['optimizer']['param_groups']]
        assert [{k:v for k,v in g.items() if k!='lr'} for g in opts]==[{k:v for k,v in g.items() if k!='lr'} for g in groups]
        assert all(g['lr']==learning_rate(u-1) for g in opts)
        m=from_state(h['model'],'R_nf4');o=optimizer_for(m,'cpu',groups)
        assert optimizer_names(m,o)==h['optimizer_names']
        assert tensor_identity(m.named_parameters())==H_CHECKPOINTS[u][1]
        assert all(int(s['step'])==u for s in h['optimizer']['state'].values())
        records.append(dict(update=u,file_sha256=sha256(f),model_identity=H_CHECKPOINTS[u][1],optimizer_groups=opts,all_optimizer_steps=u,world_size=4,cursor=h['loader']))
        del m,o,h;gc.collect()
    atomic_json(FROZEN/'OPTIMIZER.json',dict(initial_groups=groups,historical_checkpoints=records,initial_optimizer_empty=True,dispatch_pending_cuda=True))
    for arm in ARMS:
        m=from_state(p['model'],arm);a=trainability(m)
        assert a['registered']==124697386 and a['trainable']==(124475904 if arm=='L_nf4' else 124697382)
        assert tensor_identity(m.base.named_parameters())==BASE_SHA and tensor_identity(m.named_parameters())==FULL_INIT_SHA
        o=optimizer_for(m,'cpu');assert not o.state
        expected=[[n for n in g if n in a['active_names']] for g in p['optimizer_names']]
        assert optimizer_names(m,o)==expected
        atomic_json(FROZEN/f'{arm}_TRAINABILITY.json',a)
        del m,o;gc.collect()
    rows=[json.loads(x) for x in (REPO/'experiment_2d11/results/run/metrics-attempt01.jsonl').read_text().splitlines()][:5000]
    plan=[json.loads(x) for x in (REPO/'experiment_2d11/frozen/stream_plan.jsonl').read_text().splitlines()][:5000]
    assert len(rows)==len(plan)==5000
    for u,(r,s) in enumerate(zip(rows,plan),1):
        assert r['completed_updates']==s['update']==u
        assert r['data']['before']==s['before'] and r['data']['after']==s['after']
        assert r['data']['target_count']==524288 and r['pass_count']==pass_count(u)
        # macOS/Linux libm can differ at the last binary64 bit; retain H's actual LR.
        assert abs(r['lr']-learning_rate(u-1))<=4*__import__('math').ulp(r['lr'])
    assert plan[-1]['after']==dict(shard=26,position=22937600,wraps=0,logical_batches=40000)
    assert rows[-1]['data']['actual_token_sha256']=='20dbd52ccb6f7196e8014ddd63dd103886e0b8405c49192d96d875c8b1b2786f'
    for c in records:assert c['cursor']==plan[c['update']-1]['after']
    atomic_json(FROZEN/'HOST_LR_CHECK.json',dict(platform=__import__('platform').platform(),differences=[dict(update=r['completed_updates'],H_lr=r['lr'],local_lr=learning_rate(r['completed_updates']-1)) for r in rows if r['lr']!=learning_rate(r['completed_updates']-1)],policy='Historical actual values frozen; exact per-step equality required on training Linux runtime'))
    atomic_bytes(FROZEN/'stream_plan.jsonl',b''.join(canonical(r)+b'\n' for r in plan))
    atomic_bytes(FROZEN/'expected_batches.jsonl',b''.join(canonical(dict(update=r['completed_updates'],data=r['data'],lr=r['lr']))+b'\n' for r in rows))
    mon=read_json(REPO/'experiment_2d11/frozen/monitor_panel.json');assert len(mon['sequences'])==1280
    atomic_json(FROZEN/'MONITOR.json',mon)
    history=[]
    for u in SCHEDULE:
        f=REPO/f'experiment_2d11/results/run/evaluations/attempt01/ce_{u}.json';v=read_json(f)
        assert v['completed_updates']==u and len(v['rows'])==1280
        history.append(dict(update=u,ce=sum(r['nll'] for r in v['rows'])/(1280*1024),sha256=sha256(f),historical_batch_shape_differs=True))
    atomic_json(FROZEN/'H_MONITOR.json',history)
    spans=set();sources=[];seen=set();scanned=0
    for root,dirs,files in os.walk(workspace):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','node_modules','.venv') and (Path(root)/d).resolve()!=REPO.resolve() and not ((Path(root)/d).name=='experiment_2d15')]
        for name in sorted(files):
            f=Path(root)/name
            if not name.endswith('.json') or f.stat().st_size>=80*1024**2:continue
            scanned+=1;raw=f.read_bytes()
            if not any(k in raw for k in (b'target_span',b'batch_indices_in_evaluation_order',b'start_token_offset')):continue
            digest=sha256(f)
            if digest in seen:continue
            seen.add(digest);v=json.loads(raw);s=set();recover(v,s)
            if s:spans.update(s);sources.append(dict(path=str(f.relative_to(workspace)),sha256=digest,spans=len(s)))
    arr=np.load(validation,mmap_mode='r',allow_pickle=False);merged=[]
    for a,b in sorted(spans):
        assert 0<=a<b<=len(arr)
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    available=(len(arr)-1)//65536
    eligible={i for i in range(available) if not any(max(i*65536+1,a)<min((i+1)*65536+1,b) for a,b in merged)}
    chosen=[int(i) for i in np.random.default_rng(20260930).permutation(available) if i in eligible][:64]
    assert len(chosen)==64,'No fresh 4096 sequence panel remains'
    sequences=[]
    for group,i in enumerate(chosen):
        for j in range(64):
            s=i*65536+j*1024
            sequences.append(dict(id=len(sequences),canonical_group=group,batch_index=i,sequence=j,input_start=s,target_span_half_open=[s+1,s+1025],targets=1024,token_sha256=array_hash(arr[s:s+1024],arr[s+1:s+1025])))
    panel=dict(name='2D15 fresh shared three-age panel',dataset_sha256=VAL_SHA,sequences=sequences,batch_indices_in_evaluation_order=chosen,targets=4194304,identity=identity(sequences),selection_seed=20260930,numpy_version=np.__version__)
    if (FROZEN/'PANEL.json').exists():assert read_json(FROZEN/'PANEL.json')==panel
    atomic_json(FROZEN/'PANEL.json',panel)
    atomic_json(FROZEN/'PANEL_DISJOINTNESS.json',dict(passed=True,sources=sources,excluded_target_spans=sorted(spans),merged_union=merged,eligible_batches=len(eligible),selected=chosen,seed=20260930,scanned=scanned,exclusion_identity=identity(sorted(spans)),scientific_scores_inspected=False))
    atomic_json(FROZEN/'TOLERANCES.json',dict(cpu_atol=2e-6,cpu_rtol=2e-5,cuda_loss_atol=1e-5,cuda_loss_rtol=1e-5,cuda_gradient_relative_l2=.01,cuda_gradient_cosine=.9999,cuda_resume_atol=2e-6,cuda_resume_rtol=2e-5,ddp_rank_exact=True,cuda_incremental_reference_exact=True,sequence_seed=20261001,group_seed=20261002,bootstrap_resamples=50000,family=18,delta=.0001))
    atomic_json(FROZEN/'EXECUTION.json',dict(arms=list(ARMS),endpoint=5000,world_size=4,microbatch=32,context=1024,accumulation=4,global_targets=524288,torch='2.8.0+cu128',gpu='NVIDIA A100-SXM4-80GB',fused=False,foreach=None,initial_sha256=INIT_SHA,monitor_updates=list(SCHEDULE),milestones=list(MILESTONES),evaluation_batch=128,ddp=dict(broadcast_buffers=False,find_unused_parameters=False,static_graph=False),compile=False,activation_checkpointing='non_reentrant',matmul_precision='high',panel_identity=panel['identity']))
    atomic_json(PACKAGE/'results/INPUT_AUDIT.json',dict(passed=True,seconds=time.time()-started,initial_sha256=INIT_SHA,base_identity=BASE_SHA,full_identity=FULL_INIT_SHA,checkpoints=records,optimizer_options_verified=True,all_5000_stream_plan_rows=True,panel_identity=panel['identity'],shards_required=27,scientific_training_started=False))
    print(dict(passed=True,seconds=time.time()-started,eligible=len(eligible),panel_identity=panel['identity']),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workspace',type=Path,required=True);run(a.parse_args().workspace)
