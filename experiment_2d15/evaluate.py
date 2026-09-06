"""Identical incremental mode, atomic identity-bound per-sequence outputs."""
import math,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .common import *
from .model import tensor_identity
from .checkpoints import preserve_rng
from experiment_2d11.data import array_hash

def progress(root,stage,**meta):atomic_json(root/'PROGRESS.json',dict(time=time.time(),stage=stage,**meta))

@torch.inference_mode()
def ce(m,x,y,root=None,stage='PREFLIGHT',**meta):
    state=m.init_incremental_state(len(x),device=x.device,dtype=torch.bfloat16)
    nll=torch.zeros(len(x),device=x.device,dtype=torch.float64)
    for t in range(x.shape[1]):
        with torch.autocast(x.device.type,dtype=torch.bfloat16):z,state=m.incremental_step(x[:,t],state)
        nll+=F.cross_entropy(z[:,0].float(),y[:,t],reduction='none').double()
        if root is not None and t%64==0:progress(root,stage,position=t,**meta)
    return nll,state

def validate_batch(value,selected,binding):
    assert value['binding']==binding
    rows=value['rows'];assert len(rows)==len(selected)
    for r,s in zip(rows,selected):
        assert r['id']==s['id'] and r['input_start']==s['input_start'] and r['token_sha256']==s['token_sha256']
        assert r['canonical_group']==s.get('canonical_group',s['id']//64)
        assert r['count']==1024 and math.isfinite(r['nll']) and r['nll']>=0 and r['ce']==r['nll']/1024
    return rows

def collect(root,label,panel,binding,batch):
    rows=[];expected=[]
    for start in range(0,len(panel['sequences']),batch):
        name=f'batch_{start:04d}.json';expected.append(name)
        rows.extend(validate_batch(read_json(root/label/name),panel['sequences'][start:start+batch],binding))
    assert sorted(p.name for p in (root/label).glob('batch_*.json'))==sorted(expected)
    assert [r['id'] for r in rows]==list(range(len(panel['sequences'])))
    nll=math.fsum(r['nll'] for r in rows);count=sum(r['count'] for r in rows)
    return dict(passed=True,label=label,binding=binding,rows=rows,nll=nll,count=count,ce=nll/count,ppl=math.exp(nll/count))

def evaluate(m,panel,validation,root,label,checkpoint,batch,code_identity,rank,u,arm):
    before=tensor_identity(m.named_parameters());training=m.training;t=time.time()
    checkpoint=Path(checkpoint);digest=sha256(checkpoint)
    binding=dict(arm=arm,milestone=u,world_size=4,partition='B128_group_index_modulo_4',dataset_sha256=VAL_SHA,label=label,checkpoint_sha256=digest,model_tensor_identity=before,code_identity=code_identity,panel_identity=panel['identity'],configuration_identity=identity(read_json(FROZEN/'EXECUTION.json')),batch_size=batch,mode='true_incremental_bf16_fp32_ce_fp64_nll',condition=m.condition)
    arr=np.load(validation,mmap_mode='r',allow_pickle=False)
    with preserve_rng('cuda'):
        m.eval()
        for group_index,start in enumerate(range(0,len(panel['sequences']),batch)):
            if group_index%4!=rank:continue
            selected=panel['sequences'][start:start+batch];path=root/label/f'batch_{start:04d}.json'
            if path.exists():validate_batch(read_json(path),selected,binding);continue
            x=np.stack([arr[r['input_start']:r['input_start']+1024] for r in selected]).astype(np.int64)
            y=np.stack([arr[r['input_start']+1:r['input_start']+1025] for r in selected]).astype(np.int64)
            for r,xx,yy in zip(selected,x,y):assert array_hash(xx,yy)==r['token_sha256']
            started=time.time();sums,state=ce(m,torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),root/f'rank{rank}','EVALUATE',label=label,batch_start=start)
            values=sums.cpu().tolist();cache=m.incremental_cache_audit(state);assert cache['passed']
            result=dict(binding=binding,seconds=time.time()-started,cache=cache,rows=[dict(id=r['id'],canonical_group=r.get('canonical_group',r['id']//64),input_start=r['input_start'],token_sha256=r['token_sha256'],nll=n,count=1024,ce=n/1024) for r,n in zip(selected,values)])
            validate_batch(result,selected,binding);atomic_json(path,result);del sums,state
            progress(root/f'rank{rank}','BATCH_COMPLETE',label=label,completed=start+len(selected))
        assert tensor_identity(m.named_parameters())==before and sha256(checkpoint)==digest
        m.train(training)
    dist.barrier()
    summary=None
    if rank==0:
        summary=collect(root,label,panel,binding,batch);summary.update(seconds=time.time()-t,checkpoint_unchanged=True,tensors_unchanged=True)
        atomic_json(root/(label+'_COMPLETE.json'),summary)
    values=[summary];dist.broadcast_object_list(values,src=0)
    return values[0]
