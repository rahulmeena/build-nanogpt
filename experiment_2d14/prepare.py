"""Verify original tensors and prospectively frozen rows without scoring them."""
import argparse,json,random,time
import numpy as np
import torch
from .common import *
from .model import from_state,tensor_identity,trainability,optimizer_for
from .checkpoints import write,payload
from experiment_2d11.data import array_hash

def prepare(workspace,archive):
    torch.set_num_threads(4);start=time.time()
    sources=dict(initial=workspace/'runpod-checkpoint-archive/experiment_2d11/initial.pt',
        H=workspace/'runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u01000.pt',
        validation=workspace/'runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy')
    for name,digest,size in [('initial',INIT_SHA,498868405),('H',H_SHA,1651147173),('validation',VAL_SHA,200000128)]:
        assert sha256(sources[name])==digest and sources[name].stat().st_size==size
    reference=read_json(FROZEN/'REFERENCE_MANIFEST.json')
    for row in reference['files']:
        assert sha256(FROZEN/row['name'])==row['sha256']
    p=torch.load(sources['initial'],map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==0 and not p['optimizer']['state']
    m=from_state(p['model']);a=trainability(m)
    assert (a['registered'],a['trainable'],a['frozen'])==(124697386,124697382,4)
    assert tensor_identity(m.named_parameters())==FULL_INIT_SHA
    assert tensor_identity(m.base.named_parameters())==BASE_SHA
    for router in m.routers.values():
        assert torch.count_nonzero(router.W2)==0 and torch.count_nonzero(router.b2)==0
        assert torch.equal(router.b2.softmax(0),torch.tensor([.5,.5]))
    atomic_json(FROZEN/'TRAINABILITY.json',a)
    hp=torch.load(sources['H'],map_location='cpu',mmap=True,weights_only=False)
    h=from_state(hp['model']);assert hp['completed_updates']==1000 and hp['accounting']['logical_targets']==524288000
    assert tensor_identity(h.named_parameters())==H_MODEL_SHA;del h,hp
    metrics=REPO/'experiment_2d11/results/run/metrics-attempt01.jsonl'
    rows=[json.loads(z) for z in metrics.read_text().splitlines()][:1000]
    plan=[json.loads(z) for z in (REPO/'experiment_2d11/frozen/stream_plan.jsonl').read_text().splitlines()][:1000]
    assert len(rows)==len(plan)==1000
    for u,(r,s) in enumerate(zip(rows,plan),1):
        assert r['completed_updates']==u and r['data']['before']==s['before'] and r['data']['after']==s['after']
        assert r['lr']==learning_rate(u-1) and r['pass_count']==pass_count(u)
    assert rows[-1]['data']['actual_token_sha256']=='b29fc811de065ce888ea4556365e3f72ac221bf1675769707a6371635ae8ef45'
    assert plan[-1]['after']==dict(shard=5,position=24576000,wraps=0,logical_batches=8000)
    atomic_bytes(FROZEN/'stream_plan.jsonl',b''.join(canonical(r)+b'\n' for r in plan))
    atomic_bytes(FROZEN/'expected_batches.jsonl',b''.join(canonical(dict(update=r['completed_updates'],data=r['data'],lr=r['lr']))+b'\n' for r in rows))
    seconds=sum(r['training_seconds'] for r in rows)
    assert abs(seconds-2556.652936697)<1e-8
    atomic_json(FROZEN/'H_PREFIX_VERIFIED.json',dict(updates=1000,training_seconds=seconds,gpu_hours=seconds*4/3600,
        metric_source_sha256=sha256(metrics),three_pass_updates=sum(r['pass_count']==3 for r in rows)))
    arr=np.load(sources['validation'],mmap_mode='r',allow_pickle=False)
    for name,size,panel_id in [('PANEL.json',4096,'1d301019ceccca0c59801bcd1aa38a28b4e0bd90b9e54ed0c28dc27a60454602'),
                              ('MONITOR.json',1280,'db6a61e95fa56a0fc3fb15c31a322d5248a7ad36f330ef25eeca96a799891381')]:
        panel=read_json(FROZEN/name);assert panel['identity']==panel_id and len(panel['sequences'])==size
        for r in panel['sequences']:
            pos=r['input_start'];assert array_hash(arr[pos:pos+1024],arr[pos+1:pos+1025])==r['token_sha256']
    historical=[]
    for u in SCHEDULE:
        path=REPO/f'experiment_2d11/results/run/evaluations/attempt01/ce_{u}.json';v=read_json(path)
        assert len(v['rows'])==1280 and v['completed_updates']==u
        historical.append(dict(update=u,ce=sum(r['nll'] for r in v['rows'])/(1280*1024),sha256=sha256(path),
            mode='historical four-GPU incremental; grouping differs from new B128 evaluator'))
    atomic_json(FROZEN/'H_MONITOR.json',historical)
    opt=optimizer_for(m,'cpu');assert not opt.state
    torch.set_rng_state(p['cpu_rng']);random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
    ids=dict(original_initial_sha256=INIT_SHA,initial_model_tensor_identity=FULL_INIT_SHA,base_identity=BASE_SHA,
        stream_plan=sha256(FROZEN/'stream_plan.jsonl'),expected_batches=sha256(FROZEN/'expected_batches.jsonl'),
        configuration=identity(read_json(FROZEN/'EXECUTION.json')),panel_identity=read_json(FROZEN/'PANEL.json')['identity'])
    output=archive/'local_initial.pt'
    if not output.exists():write(output,payload(m,opt,p['loader'],0,{},ids,dict(training_seconds=0.,evaluation_seconds=0.),'cpu'))
    assert sha256(sources['initial'])==INIT_SHA and sha256(sources['H'])==H_SHA
    atomic_json(PACKAGE/'results/INPUT_AUDIT.json',dict(passed=True,seconds=time.time()-start,initial_identity=FULL_INIT_SHA,
        source_backbone_identity=BASE_SHA,source_files={n:dict(path=str(f),sha256=sha256(f),bytes=f.stat().st_size) for n,f in sources.items()},
        initial_execution_sha256=sha256(output),fresh_routers_50_50=True,source_optimizer_empty=True,final_panel_scored=False,
        initial_cuda_seed=p['cuda_seed'],historical_H_gpu_hours=seconds*4/3600))
    print(dict(passed=True,seconds=time.time()-start,initial_execution=str(output)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
    a=ap.parse_args();prepare(a.workspace,a.archive)
