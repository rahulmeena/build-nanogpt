"""Local-only input freeze. No provider operations and no scientific scoring."""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import numpy as np
import torch
import tiktoken
from .common import *
from .data import full_plan, array_hash, partition_examples, collect_historical_spans
from .model import fresh_base, FreshH, tensor_identity, original_symbols, optimizer_for, optimizer_names, load_g


def freeze_panel(tokens, indices, name):
    rows = []
    for index in indices:
        for sequence in range(64):
            start = int(index)*65536+sequence*1024
            x,y = tokens[start:start+1024],tokens[start+1:start+1025]
            rows.append(dict(id=len(rows), batch_index=int(index), sequence=sequence,
                             input_start=start, target_span_half_open=[start+1,start+1025],
                             targets=1024, token_sha256=array_hash(x,y)))
    return dict(name=name, dataset_sha256=VAL_SHA, sequences=rows,
                batch_indices_in_evaluation_order=list(map(int,indices)),
                canonical_target_spans_half_open=[[int(i)*65536+1,(int(i)+1)*65536+1] for i in indices],
                targets=len(rows)*1024, identity=identity(rows), partition=partition_examples(rows))


def prepare(workspace, archive):
    torch.set_num_threads(4)
    FROZEN.mkdir(exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    baseline = workspace/'baseline_artifacts/gpt2_124m_fineweb10b_20260810T141222Z'
    checkpoint = baseline/'checkpoints/model_19072.pt'
    assert checkpoint.stat().st_size == 497958271 and sha256(checkpoint) == BASELINE_SHA
    g = load_g(checkpoint)
    assert sum(p.numel() for p in g.parameters()) == 124475904
    assert all(torch.isfinite(p).all() for p in g.parameters())
    del g
    metrics = [json.loads(x) for x in (baseline/'metrics.jsonl').read_text().splitlines()]
    assert Counter(r['kind'] for r in metrics) == {'train':19073,'val':78,'hellaswag':78}
    assert all(math.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
    for u, expected in REFERENCES.items():
        assert next(r['val_loss'] for r in metrics if r['kind']=='val' and r['step']==u) == expected['ce']
        assert next(r['hellaswag_accuracy'] for r in metrics if r['kind']=='hellaswag' and r['step']==u) == expected['hella']
    rows = list(csv.DictReader((baseline/'baseline_learning_curve.csv').open()))
    for row in rows:
        u=int(row['step'])
        for kind,field in [('val','val_loss'),('hellaswag','hellaswag_accuracy')]:
            assert float(row[field]) == next(r[field] for r in metrics if r['kind']==kind and r['step']==u)
    atomic_json(FROZEN/'baseline_identity.json',dict(path=str(checkpoint),sha256=BASELINE_SHA,
                bytes=checkpoint.stat().st_size, completed_updates=19072, logical_targets=9999220736,
                source_commit=SOURCE_COMMIT, original_seed=1337,
                metrics_sha256=sha256(baseline/'metrics.jsonl'), csv_rows_verified=78,
                all_metrics_rows_parsed=len(metrics), intermediate_1000_2000_available=False))
    atomic_json(FROZEN/'baseline_metrics.json',metrics)
    metadata = read_json(FROZEN/'dataset.json')['shards']
    actual_path = REPO/'results/experiment_2d11/initial_probe/output/ACTUAL_SHARDS.json'
    actual = read_json(actual_path)
    assert actual['complete']
    observed = {r['filename']:r for r in actual['shards']}
    for r in metadata:
        a=observed[r['filename']]
        assert a['tokens']==r['tokens'] and a['dtype']==r['dtype'] and a['bytes']==r['disk_size_bytes']
    assert observed['edufineweb_val_000000.npy']['sha256']==VAL_SHA
    train = sorted([r for r in actual['shards'] if 'train' in r['filename']],key=lambda r:r['filename'])
    assert sum(r['tokens'] for r in train)==9853989344
    atomic_json(FROZEN/'shards.json',dict(train=train, validation=observed['edufineweb_val_000000.npy'],
                actual_remote_content_hashed=True, provenance='existing original persistent FineWeb shards; metadata checked against original report',
                measured_source='/workspace/build-nanogpt/edu_fineweb10B'))
    plan = full_plan(train)
    plan_path=FROZEN/'stream_plan.jsonl'
    atomic_bytes(plan_path,b''.join(canonical(x)+b'\n' for x in plan))
    atomic_json(FROZEN/'stream_summary.json',dict(rows=len(plan),sha256=sha256(plan_path),
                first=plan[0],last=plan[-1],three_pass_updates=sum(r['pass_count']==3 for r in plan),
                plan_is_metadata=True,actual_per_update_token_hashes_pending=True,**accounting(19072)))
    val = workspace/'runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy'
    assert sha256(val)==VAL_SHA
    tokens=np.load(val,mmap_mode='r')
    panel_path=FROZEN/'fresh_panel.json'
    if not panel_path.exists():
        spans={(i*65536+1,(i+1)*65536+1) for i in range(128)}
        sources=[]
        for directory in [REPO/'results',REPO/'configs']:
            for p in sorted(directory.rglob('*.json')):
                if 'experiment_2d11' in p.parts:
                    continue
                try:
                    value=read_json(p)
                except (ValueError,OSError):
                    continue
                recovered=set()
                collect_historical_spans(value,recovered)
                # Explicit excluded sets from sealed disjointness audits remain reserved.
                for span in value.get('excluded_target_spans',[]) if isinstance(value,dict) else []:
                    recovered.add(tuple(map(int,span)))
                if recovered:
                    spans.update(recovered)
                    sources.append(dict(path=str(p.relative_to(REPO)),sha256=sha256(p),spans=len(recovered)))
        available=(len(tokens)-1)//65536
        selected=[]
        for i in np.random.default_rng(20260917).permutation(available):
            a,b=int(i)*65536+1,(int(i)+1)*65536+1
            if any(max(a,c)<min(b,d) for c,d in spans):
                continue
            selected.append(int(i))
            if len(selected)==64:
                break
        assert len(selected)==64, 'cannot construct required fresh panel'
        panel=freeze_panel(tokens,selected,'fresh final paired CE')
        assert len({r['token_sha256'] for r in panel['sequences']})==4096
        atomic_json(panel_path,panel)
        atomic_json(FROZEN/'panel_disjointness.json',dict(passed=True,selection_seed=20260917,
                    numpy_version=np.__version__,score_inspection=False,one_panel_constructed=True,
                    excluded_target_spans=sorted(spans),sources=sources,selected=selected,
                    fresh_panel_identity=panel['identity']))
    atomic_json(FROZEN/'monitor_panel.json',freeze_panel(tokens,range(20),'original B64 first 20 batches'))
    hella_path=workspace/'runpod-backups/golden_tomato_cat_1j5ss684o2e3j0_20260831/workspace/build-nanogpt/hellaswag/hellaswag_val.jsonl'
    examples=[json.loads(x) for x in hella_path.read_text().splitlines()]
    assert len(examples)==10042 and all(x['split']=='val' for x in examples)
    enc=tiktoken.get_encoding('gpt2')
    code=(FROZEN/'original_hellaswag.py.txt').read_text()
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='render_example')
    namespace=dict(torch=torch,enc=enc)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-render>','exec'),namespace)
    rows=[]
    for i,e in enumerate(examples):
        raw,tok,mask,label=namespace['render_example'](e)
        choices=[]
        for ending in raw['ending_tokens']:
            choices.append(dict(tokens=raw['ctx_tokens']+ending,prompt_length=len(raw['ctx_tokens'])))
        assert max(map(lambda c:len(c['tokens']),choices))<=1024
        assert tok.size(1)==max(len(c['tokens']) for c in choices)
        rows.append(dict(id=i,original_id=e['ind'],label=int(label),choices=choices,length=tok.size(1)))
    token_file=archive/'hellaswag_tokenized.json'
    atomic_json(token_file,rows)
    atomic_json(FROZEN/'hellaswag_manifest.json',dict(examples=len(rows),split='val',
                raw_sha256=sha256(hella_path),tokenized_sha256=sha256(token_file),
                tokenizer='gpt2',tiktoken_version=tiktoken.__version__,
                original_scoring_source_sha256=sha256(FROZEN/'original_train_gpt2.py.txt'),
                original_renderer_sha256=sha256(FROZEN/'original_hellaswag.py.txt'),
                partition=partition_examples(rows),max_length=max(x['length'] for x in rows),
                historical_scoring='BF16 logits; cross_entropy outside autocast in BF16; int64 completion mask; BF16 sum and BF16 quotient',
                tokenized_file=str(token_file)))
    initial=archive/'initial.pt'
    if not initial.exists():
        base=fresh_base()
        expected=tensor_identity(base.named_parameters())
        h=FreshH(base)
        assert tensor_identity(h.base.named_parameters())==expected
        independent=fresh_base()
        assert tensor_identity(independent.named_parameters())==expected
        del independent
        assert sum(p.numel() for p in h.parameters())==124697386
        assert sum(p.numel() for p in h.parameters() if p.requires_grad)==124697382
        assert h.base.lm_head.weight is h.base.transformer.wte.weight
        optimizer=optimizer_for(h,'cpu')
        random.seed(1337);np.random.seed(1337)
        payload=dict(schema='2d11-initial-v1',model=h.state_dict(),optimizer=optimizer.state_dict(),
                     optimizer_names=optimizer_names(h,optimizer),completed_updates=0,
                     cpu_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                     cuda_seed=1337,base_identity=expected,model_identity=tensor_identity(h.named_parameters()),
                     loader=dict(shard=0,position=0,wraps=0,logical_batches=0),evaluation_ledger={})
        temp=initial.with_suffix('.tmp');torch.save(payload,temp);temp.replace(initial)
        atomic_json(FROZEN/'initial_identity.json',dict(path=str(initial),sha256=sha256(initial),
                    base_identity=expected,mapped_base_identity=expected,base_equality=True,
                    model_identity=payload['model_identity'],registered_parameters=124697386,
                    active_parameters=124697382,router_parameters=221478,unused_compatibility_parameters=4,
                    seed=1337,source_commit=SOURCE_COMMIT,torch_version=torch.__version__,
                    historical_step_zero_available=False,router_seeds=[20260916,20260918,20260920],
                    eligible_initial_mixture=[.5,.5],empty_memory_mixture=[1.,0.],interpretation=INTERPRETATION))
    config=dict(deterministic_algorithms=True,cublas_workspace_config=':4096:8',experiment='2D11',endpoint=ENDPOINT,targets_per_update=TARGETS,ranks=4,
                microbatch=32,accumulation=4,sequence_length=1024,ce_schedule=CE_SCHEDULE,
                hellaswag_schedule=HELLA_SCHEDULE,preserve_updates=PRESERVE,checkpoint_interval=500,
                max_cumulative_pod_hours=24,max_aggregate_gpu_hours=96,contingency_hours=2,
                soft_stop_reserve_seconds=1800,torch_compile=False,matmul_precision='high',
                torch_cuda_version='2.8.0+cu128',training_numpy_version='2.1.2',
                lr_max_steps=19073,lr_warmup_steps=715,lr_peak=6e-4,lr_floor=6e-5,
                betas=[.9,.95],epsilon=1e-8,clip_norm=1.,weight_decay=.1,
                interpretation=INTERPRETATION,quality_references=REFERENCES,
                eval_batch_candidates=[32,64,128],hella_example_batch_candidates=[8,16,32],
                tolerances=dict(cpu_atol=2e-6,cpu_rtol=2e-5,ddp_gradient_relative_l2=.005,
                                ddp_gradient_max_abs=.002,resume_max_abs=1e-6,
                                g_incremental_ce_abs=.002,g_incremental_logits_abs=.15,
                                hella_batched_score_abs=.03125,hella_prediction_agreement=1.))
    atomic_json(FROZEN/'config.json',config)
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(FROZEN/'local_requirements.txt').open('w'),check=True)
    print(json.dumps(dict(prepared=True,archive=str(archive),fresh_panel=read_json(panel_path)['identity'],
                          initial_sha256=sha256(initial),input_freeze=identity(config)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();prepare(a.workspace,a.archive)
