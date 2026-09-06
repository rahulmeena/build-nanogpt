"""Local provenance and fresh panel freeze; never evaluates panel or starts GPU."""
import argparse,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .model import load_model,tensor_identity
from experiment_2d11.data import array_hash,collect_historical_spans


def recover(value,spans):
    collect_historical_spans(value,spans)
    def extra(v):
        if isinstance(v,dict):
            for k in ('excluded_target_spans','excluded_target_spans_half_open','reserved_target_spans_half_open'):
                for s in v.get(k,[]):
                    if isinstance(s,list) and len(s)==2:spans.add(tuple(map(int,s)))
            if 'target_span_half_open' in v:spans.add(tuple(map(int,v['target_span_half_open'])))
            for x in v.values():extra(x)
        elif isinstance(v,list):
            for x in v:extra(x)
    extra(value)


def prepare(workspace):
    started=time.time();torch.set_num_threads(4)
    source=workspace/'parallel_2d2_master_dev/2d11_from_scratch'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD']).decode().strip()==SOURCE_COMMIT
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','experiment-2d11-from-scratch-h-vs-gpt2-10b-final^{}']).decode().strip()==SOURCE_COMMIT
    cp=workspace/'runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u19072.pt'
    val=workspace/'runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy'
    assert sha256(cp)==CHECKPOINT_SHA and sha256(val)==VAL_SHA
    model=load_model(cp);model_hash=tensor_identity(model.named_parameters());del model
    spans=set();sources=[];seen={}
    old=read_json(source/'experiment_2d11/frozen/panel_disjointness.json')
    for s in old['sources']:
        p=source/s['path'];assert sha256(p)==s['sha256']
    required=[source/'experiment_2d11/frozen'/n for n in ('panel_disjointness.json','monitor_panel.json','fresh_panel.json')]
    required += [source/s['path'] for s in old['sources']]
    # Scan all project JSON manifests, including ignored backup/worktree trees.
    # Content hashes deduplicate copies; every source with recovered spans is logged.
    candidates=set(required);scanned=0
    for root,dirs,files in os.walk(workspace):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','node_modules','.venv') and (Path(root)/d).resolve()!=REPO.resolve()]
        for name in files:
            if name.endswith('.json'):
                p=Path(root)/name
                if p.stat().st_size<80*1024**2:candidates.add(p)
    errors=[]
    for p in sorted(candidates):
        scanned+=1
        raw=p.read_bytes()
        # Restrict interpretation to data with explicit target-offset fields.
        if not any(k in raw for k in (b'target_span',b'batch_indices_in_evaluation_order',b'start_token_offset')):continue
        h=sha256(p)
        if h in seen:
            seen[h]['aliases'].append(str(p.relative_to(workspace)));continue
        try:value=read_json(p)
        except ValueError:
            errors.append(str(p));continue
        recovered=set();recover(value,recovered)
        if not recovered:continue
        spans.update(recovered)
        entry=dict(path=str(p.relative_to(workspace)),sha256=h,spans=len(recovered),aliases=[])
        sources.append(entry);seen[h]=entry
    assert not errors,errors
    tokens=np.load(val,mmap_mode='r',allow_pickle=False)
    # Merge union for efficient interval intersection. Original spans remain archived.
    merged=[]
    for a,b in sorted(spans):
        assert 0<=a<b<=len(tokens),(a,b)
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    available=(len(tokens)-1)//65536
    eligible=[i for i in range(available) if not any(max(i*65536+1,a)<min((i+1)*65536+1,b) for a,b in merged)]
    eligible_set=set(eligible)
    selected=[int(i) for i in np.random.default_rng(20260921).permutation(available) if i in eligible_set][:64]
    assert len(selected)==64
    rows=[]
    for group,i in enumerate(selected):
        for j in range(64):
            s=i*65536+j*1024
            rows.append(dict(id=len(rows),canonical_group=group,batch_index=i,sequence=j,input_start=s,
                target_span_half_open=[s+1,s+1025],targets=1024,
                token_sha256=array_hash(tokens[s:s+1024],tokens[s+1:s+1025])))
    assert len({r['input_start'] for r in rows})==len({r['token_sha256'] for r in rows})==4096
    ordered=sorted(r['target_span_half_open'] for r in rows)
    assert all(b<=c for (a,b),(c,d) in zip(ordered,ordered[1:]))
    panel=dict(name='2D12 fresh paired panel',dataset_sha256=VAL_SHA,sequences=rows,
        batch_indices_in_evaluation_order=selected,targets=4194304,identity=identity(rows),
        selection_seed=20260921,numpy_version=np.__version__)
    # Never overwrite an existing reservation with a silently changed panel.
    if (FROZEN/'PANEL.json').exists():assert read_json(FROZEN/'PANEL.json')==panel
    atomic_json(FROZEN/'PANEL.json',panel)
    atomic_json(FROZEN/'PANEL_DISJOINTNESS.json',dict(passed=True,sources=sources,json_files_scanned=scanned,
        excluded_target_spans=sorted(spans),merged_union=merged,eligible_batches=len(eligible),
        total_complete_batches=available,selected=selected,numpy_version=np.__version__,seed=20260921,
        score_inspection=False,panel_identity=panel['identity'],source_hashes_verified=True))
    atomic_json(FROZEN/'SOURCE_IDENTITIES.json',dict(source_commit=SOURCE_COMMIT,
        source_tag='experiment-2d11-from-scratch-h-vs-gpt2-10b-final',checkpoint_path=str(cp),
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_bytes=cp.stat().st_size,model_tensor_identity=model_hash,
        completed_updates=19072,historical_logical_targets=9999220736,registered_parameters=124697386,
        tied_embeddings=True,strict_load=True,validation_path=str(val),validation_sha256=VAL_SHA))
    atomic_json(FROZEN/'INTERVENTIONS.json',dict(conditions={k:list(v) for k,v in CONDITIONS.items()},
        block_numbering='zero based in code',off_weights=[1,0],router_bypassed=True,read_skipped=True,
        windows={'B1':2,'B3':32,'B5':64,'other':1024},writers={'B1':'B12','B3':'B10','B5':'B8'},
        recurrent_lags={'B1':[1,1023],'B3':[31,1023],'B5':[63,1023]},all_cache_bookkeeping_retained=True,
        trajectories='independent empty state per batch and condition',scientific_training_updates=0))
    atomic_json(FROZEN/'ANALYSIS_CONFIG.json',dict(numpy=np.__version__,torch=torch.__version__,python=sys.version,
        resamples=50000,sequence_seed=20260922,group_seed=20260923,percentile_method='linear',
        interval_percentiles=[[2.5,97.5],[.625,99.375]],reference=.0001,common_indices_all_contrasts=True,
        cpu_atol=2e-6,cpu_rtol=2e-5,on_exact=True,cuda_tolerances='frozen on disposable preflight before panel scoring'))
    print(dict(passed=True,eligible_batches=len(eligible),panel_identity=panel['identity'],sources=len(sources),seconds=time.time()-started),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);prepare(p.parse_args().workspace)
