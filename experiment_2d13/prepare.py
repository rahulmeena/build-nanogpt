"""Freeze all inputs, exclusions and exact H prefix locally, with no scoring."""
import os,sys,subprocess,time,random
import numpy as np
import torch
from .common import *
from .model import from_state,tensor_identity,optimizer_for,trainability
from .checkpoints import payload,write
from experiment_2d12.prepare import recover
from experiment_2d11.data import array_hash

def prepare(workspace,archive):
    t=time.time();torch.set_num_threads(4)
    source=workspace/'parallel_2d2_master_dev/2d12_h10b_ablation'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD']).decode().strip()=='622f34b29ec54816dcf293aef9228939c72a6dd8'
    inputs={n:workspace/p for n,p in dict(initial='runpod-checkpoint-archive/experiment_2d11/initial.pt',H='runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u01000.pt',validation='runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy').items()}
    for n,h,size in [('initial',INIT_SHA,498868405),('H',H_SHA,1651147173),('validation',VAL_SHA,200000128)]:
        assert sha256(inputs[n])==h and inputs[n].stat().st_size==size,(n,size)
    p=torch.load(inputs['initial'],map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==0 and p['loader']==dict(shard=0,position=0,wraps=0,logical_batches=0)
    m=from_state(p['model']);assert tensor_identity(m.base.named_parameters())==BASE_SHA
    a=trainability(m);assert (a['registered'],a['trainable'],a['frozen'])==(124697386,124475904,221482)
    atomic_json(FROZEN/'TRAINABILITY.json',a)
    hp=torch.load(inputs['H'],map_location='cpu',mmap=True,weights_only=False)
    assert hp['completed_updates']==1000 and hp['accounting']['logical_targets']==524288000
    hmodel=from_state(hp['model']);assert tensor_identity(hmodel.named_parameters())==H_MODEL_SHA
    del hmodel,hp
    hrows=[__import__('json').loads(l) for l in (source/'experiment_2d11/results/run/metrics-attempt01.jsonl').read_text().splitlines()][:1000]
    plan=[__import__('json').loads(l) for l in (source/'experiment_2d11/frozen/stream_plan.jsonl').read_text().splitlines()][:1000]
    assert [r['completed_updates'] for r in hrows]==list(range(1,1001))
    for i,(h,s) in enumerate(zip(hrows,plan),1):
        assert h['data']['before']==s['before'] and h['data']['after']==s['after'] and h['lr']==learning_rate(i-1)
    assert hrows[-1]['data']['actual_token_sha256']=='b29fc811de065ce888ea4556365e3f72ac221bf1675769707a6371635ae8ef45'
    assert plan[-1]['after']==dict(shard=5,position=24576000,wraps=0,logical_batches=8000)
    atomic_bytes(FROZEN/'stream_plan.jsonl',b''.join(canonical(r)+b'\n' for r in plan))
    atomic_bytes(FROZEN/'expected_batches.jsonl',b''.join(canonical(dict(update=r['completed_updates'],data=r['data'],lr=r['lr']))+b'\n' for r in hrows))
    secs=sum(r['training_seconds'] for r in hrows)
    atomic_json(FROZEN/'H_PREFIX.json',dict(metric_source_sha256=sha256(source/'experiment_2d11/results/run/metrics-attempt01.jsonl'),updates=1000,training_seconds=secs,gpu_hours=secs*4/3600,passes=sum(r['pass_count'] for r in hrows),three_pass_updates=sum(r['pass_count']==3 for r in hrows)))
    mon=read_json(source/'experiment_2d11/frozen/monitor_panel.json')
    assert len(mon['sequences'])==1280 and mon['batch_indices_in_evaluation_order']==list(range(20))
    atomic_json(FROZEN/'MONITOR.json',mon)
    historical=[]
    for u in SCHEDULE:
        f=source/f'experiment_2d11/results/run/evaluations/attempt01/ce_{u}.json';v=read_json(f)
        assert v['completed_updates']==u and v['identities']['monitor_panel.json']==sha256(source/'experiment_2d11/frozen/monitor_panel.json')
        assert [r['id'] for r in v['rows']]==list(range(1280)) and all(r['count']==1024 for r in v['rows'])
        for rank in range(4):
            part=read_json(source/f'experiment_2d11/results/run/evaluations/attempt01/ce_{u}-rank{rank}.json')
            assert part['mode']=='incremental_bf16_fp32_ce_fp64_sum' and part['panel_identity']==mon['identity']
        historical.append(dict(update=u,ce=sum(r['nll'] for r in v['rows'])/(1280*1024),source=str(f),sha256=sha256(f),mode='incremental_bf16_fp32_ce_fp64_sum',panel_identity=mon['identity']))
    atomic_json(FROZEN/'H_MONITOR.json',historical)
    spans=set();sources=[];seen={};scanned=0;errors=[]
    for root,dirs,files in os.walk(workspace):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','node_modules','.venv') and (Path(root)/d).resolve() not in (REPO.resolve(),archive.resolve())]
        for name in sorted(files):
            if not name.endswith('.json'):continue
            f=Path(root)/name
            if f.stat().st_size>=80*1024**2:continue
            scanned+=1;raw=f.read_bytes()
            if not any(k in raw for k in (b'target_span',b'batch_indices_in_evaluation_order',b'start_token_offset')):continue
            h=sha256(f)
            if h in seen:seen[h]['aliases'].append(str(f.relative_to(workspace)));continue
            try:v=__import__('json').loads(raw)
            except ValueError:errors.append(str(f));continue
            s=set();recover(v,s)
            if not s:continue
            spans.update(s);entry=dict(path=str(f.relative_to(workspace)),sha256=h,spans=len(s),aliases=[]);sources.append(entry);seen[h]=entry
    assert not errors,errors
    arr=np.load(inputs['validation'],mmap_mode='r',allow_pickle=False)
    merged=[]
    for a,b in sorted(spans):
        assert 0<=a<b<=len(arr)
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    available=(len(arr)-1)//65536
    eligible={i for i in range(available) if not any(max(i*65536+1,a)<min((i+1)*65536+1,b) for a,b in merged)}
    chosen=[int(i) for i in np.random.default_rng(20260924).permutation(available) if i in eligible][:64];assert len(chosen)==64
    rows=[]
    for group,i in enumerate(chosen):
        for j in range(64):
            s=i*65536+j*1024
            rows.append(dict(id=len(rows),canonical_group=group,batch_index=i,sequence=j,input_start=s,target_span_half_open=[s+1,s+1025],targets=1024,token_sha256=array_hash(arr[s:s+1024],arr[s+1:s+1025])))
    panel=dict(name='2D13 fresh paired 4096-sequence panel',dataset_sha256=VAL_SHA,sequences=rows,batch_indices_in_evaluation_order=chosen,targets=4194304,identity=identity(rows),selection_seed=20260924,numpy_version=np.__version__)
    if (FROZEN/'PANEL.json').exists():assert read_json(FROZEN/'PANEL.json')==panel
    atomic_json(FROZEN/'PANEL.json',panel)
    atomic_json(FROZEN/'PANEL_DISJOINTNESS.json',dict(passed=True,sources=sources,excluded_target_spans=sorted(spans),merged_union=merged,eligible_batches=len(eligible),selected=chosen,seed=20260924,scanned=scanned,scientific_scores_inspected=False))
    atomic_json(FROZEN/'SOURCE_IDENTITIES.json',dict(inputs={n:dict(path=str(f),sha256=sha256(f),bytes=f.stat().st_size) for n,f in inputs.items()},base_identity=BASE_SHA,H_model_identity=H_MODEL_SHA,L_initial_tensor_identity=tensor_identity(m.named_parameters()),source_commit='622f34b29ec54816dcf293aef9228939c72a6dd8'))
    atomic_json(FROZEN/'TOLERANCES.json',dict(cpu_atol=2e-6,cpu_rtol=2e-5,cuda_loss_atol=1e-5,cuda_loss_rtol=1e-5,cuda_gradient_relative_l2=.01,cuda_gradient_cosine=.9999,cuda_resume_atol=2e-6,cuda_resume_rtol=2e-5,cuda_independent_incremental_exact=True,cuda_on_reference_exact=True,cuda_block_atol=.15,cuda_block_rtol=.02,analysis_resamples=50000,sequence_seed=20260925,group_seed=20260926,reference=.0001))
    opt=optimizer_for(m,'cpu');assert not opt.state
    torch.set_rng_state(p['cpu_rng']);random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
    ids=dict(stream_plan=sha256(FROZEN/'stream_plan.jsonl'),expected_batches=sha256(FROZEN/'expected_batches.jsonl'),initial_sha=INIT_SHA,panel_identity=panel['identity'],configuration=identity(read_json(FROZEN/'EXECUTION.json')))
    out=archive/'local_initial.pt'
    if not out.exists():write(out,payload(m,opt,p['loader'],0,{},ids,dict(training_seconds=0.,evaluation_seconds=0.),'cpu'))
    print(dict(passed=True,seconds=time.time()-t,initial_checkpoint=str(out),eligible_batches=len(eligible),sources=len(sources)),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();prepare(a.workspace,a.archive)
