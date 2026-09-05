"""Independent-rank true incremental CE and batched HellaSwag scoring."""
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import atomic_json, heartbeat, identity, read_json
from .data import array_hash


class GateCollector:
    """Fixed-size device reductions only; no attention dumps or per-token CPU copies."""
    def __init__(self):
        self.values = {}

    def record(self, block, diagnostics):
        available = diagnostics['available']
        for group, mask in [('eligible', available), ('forced_local', ~available)]:
            for name in ('lambda_L','lambda_R','cast_lambda_L','cast_lambda_R','entropy','logit_difference'):
                v = diagnostics[name].detach().double()
                m = mask.expand_as(v)
                count = m.sum().double()
                total = (v*m).sum()
                square = (v.square()*m).sum()
                low = torch.where(m,v,torch.inf).amin()
                high = torch.where(m,v,-torch.inf).amax()
                key = (block, group, name)
                if key not in self.values:
                    self.values[key] = [count,total,square,low,high]
                else:
                    a=self.values[key]
                    a[0]+=count;a[1]+=total;a[2]+=square
                    a[3]=torch.minimum(a[3],low);a[4]=torch.maximum(a[4],high)

    def finish(self):
        rows=[]
        for (block,group,name),v in self.values.items():
            count,total,square,low,high=torch.stack(v).cpu().tolist()
            if count:
                rows.append(dict(block=block+1,group=group,metric=name,count=int(count),
                                 sum=total,sum_square=square,min=low,max=high))
        return rows


@torch.inference_mode()
def ce_batch(model, x, y, bf16=True):
    device=x.device
    state=model.init_incremental_state(x.size(0),device=device,
                                       dtype=torch.bfloat16 if bf16 else torch.float32)
    sums=torch.zeros(x.size(0),device=device,dtype=torch.float64)
    for position in range(x.size(1)):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
            logits,state=model.incremental_step(x[:,position],state)
        # Explicit FP32 CE and FP64 accumulation for paired and monitoring CE.
        sums+=F.cross_entropy(logits[:,0].float(),y[:,position],reduction='none').double()
    return sums, state


def collate_hella(examples, device):
    length=max(e['length'] for e in examples)
    tokens=torch.zeros((len(examples)*4,length),dtype=torch.long)
    masks=torch.zeros_like(tokens)
    for i,e in enumerate(examples):
        for choice,c in enumerate(e['choices']):
            row=i*4+choice;n=len(c['tokens'])
            tokens[row,:n]=torch.tensor(c['tokens'])
            masks[row,c['prompt_length']:n]=1
    return tokens.to(device),masks.to(device)


@torch.inference_mode()
def hella_batch(model, examples, device, bf16=True):
    tokens,masks=collate_hella(examples,device)
    state=model.init_incremental_state(tokens.size(0),device=device,
                                       dtype=torch.bfloat16 if bf16 else torch.float32)
    losses=[]
    for position in range(tokens.size(1)-1):
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):
            logits,state=model.incremental_step(tokens[:,position],state)
        # Original scoring runs CE OUTSIDE autocast, directly on BF16 logits.
        losses.append(F.cross_entropy(logits[:,0].contiguous(),tokens[:,position+1],reduction='none'))
    losses=torch.stack(losses,dim=1)
    scores=[]
    for i,e in enumerate(examples):
        # Retain each original four-choice padding length and BF16 reduction shape.
        end=e['length']-1
        m=masks[i*4:i*4+4,1:end+1]
        score=(losses[i*4:i*4+4,:end]*m).sum(1)/m.sum(1)
        scores.append(score)
    return torch.stack(scores)


def evaluate_ce(model, panel, validation, ids, batch_size, device, run, rank, label,
                diagnostics=False, deadline=None):
    started=time.time()
    deadline=deadline or started+1800
    rows=panel['sequences']
    arr=np.load(validation,mmap_mode='r',allow_pickle=False)
    model.eval()
    collector=GateCollector() if diagnostics else None
    if diagnostics:
        model.gate_collector=collector
    result=[]
    last_state=None
    try:
        for start in range(0,len(ids),batch_size):
            if time.time()>=deadline:
                raise TimeoutError('CE evaluation stage deadline')
            selected=[rows[i] for i in ids[start:start+batch_size]]
            x=np.stack([arr[r['input_start']:r['input_start']+1024] for r in selected]).astype(np.int64)
            y=np.stack([arr[r['input_start']+1:r['input_start']+1025] for r in selected]).astype(np.int64)
            for r,xx,yy in zip(selected,x,y):
                assert array_hash(xx,yy)==r['token_sha256']
            sums,last_state=ce_batch(model,torch.from_numpy(x).to(device),torch.from_numpy(y).to(device))
            sums=sums.cpu().tolist()
            result.extend(dict(id=r['id'],nll=nll,count=1024,ce=nll/1024) for r,nll in zip(selected,sums))
            heartbeat(run,rank,'SCHEDULED_EVALUATION',start+len(selected),deadline,label=label)
        output=dict(label=label,rank=rank,panel_identity=panel['identity'],rows=result,
                    seconds=time.time()-started,mode='incremental_bf16_fp32_ce_fp64_sum',
                    gate_statistics=collector.finish() if collector else None)
        if diagnostics and last_state is not None:
            audit=model.incremental_cache_audit(last_state)
            assert audit['passed']
            per_sequence=audit['actual_unique_storage_bytes']//last_state.batch_size
            assert per_sequence==33289728
            output['persistent_bf16_bytes_per_sequence']=per_sequence
            output['router_parameter_norms']={n:p.detach().float().norm().item() for n,p in model.named_parameters() if n.startswith('routers.')}
        return output
    finally:
        if diagnostics:
            model.gate_collector=None


def evaluate_hella(model, examples, ids, batch_size, device, run, rank, label, deadline=None):
    started=time.time();deadline=deadline or started+2400
    model.eval();result=[]
    for start in range(0,len(ids),batch_size):
        if time.time()>=deadline:
            raise TimeoutError('HellaSwag evaluation stage deadline')
        selected=[examples[i] for i in ids[start:start+batch_size]]
        scores=hella_batch(model,selected,device).float().cpu().tolist()
        for e,s in zip(selected,scores):
            pred=min(range(4),key=lambda i:s[i])
            result.append(dict(id=e['id'],original_id=e['original_id'],label=e['label'],
                               scores=s,prediction=pred,correct=pred==e['label']))
        heartbeat(run,rank,'SCHEDULED_EVALUATION',start+len(selected),deadline,label=label)
    return dict(label=label,rank=rank,rows=result,seconds=time.time()-started,
                mode='incremental_bf16_original_completion_scoring',
                cross_entropy_dtype='bfloat16',sum_dtype='bfloat16',quotient_dtype='bfloat16')


def merge(parts, expected_ids, kind):
    rows=sorted([row for part in parts for row in part['rows']],key=lambda r:r['id'])
    assert [r['id'] for r in rows]==sorted(expected_ids), 'evaluation IDs duplicated or omitted'
    if kind=='ce':
        score=sum(r['nll'] for r in rows)/sum(r['count'] for r in rows)
    else:
        score=sum(r['correct'] for r in rows)/len(rows)
    return dict(score=score,rows=rows,example_count=len(rows),
                elapsed_seconds=max(p['seconds'] for p in parts),ownership_verified=True,
                partition_identity=identity([[r['id'] for r in p['rows']] for p in parts]))


@torch.inference_mode()
def historical_g_monitor(g, validation, rank, device):
    """Reproduce original B64, 20-batch reset parallel causal validation evaluator."""
    arr=np.load(validation,mmap_mode='r');losses=[];g.eval();started=time.time()
    for logical_batch in range(rank,20,4):
        start=logical_batch*64*1024
        x=torch.from_numpy(np.asarray(arr[start:start+64*1024],dtype=np.int64).reshape(64,1024)).to(device)
        y=torch.from_numpy(np.asarray(arr[start+1:start+64*1024+1],dtype=np.int64).reshape(64,1024)).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):_,loss=g.base(x,y)
        losses.append(dict(logical_batch=logical_batch,loss=loss.item()))
    return dict(rank=rank,losses=losses,seconds=time.time()-started,
                mode='original_parallel_B64_bf16_autocast_fp32_ce',checkpoint='saved_G19072')
