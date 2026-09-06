"""L-specific complete checkpoints, immutable bounded writer and strict reopens."""
import copy,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from experiment_2d11.checkpoints import capture_rng,restore_rng,preserve_rng,cpu_snapshot
from .common import *
from .model import from_state,optimizer_for,optimizer_names,tensor_identity,trainability,frozen_identity

def validate(p,tiny=False):
    assert p['schema']=='2d13-L-complete-v1' and p['world_size']==1
    u=p['completed_updates'];assert p['accounting']==accounting(u) and p['next_lr_index']==u
    if not tiny and (u>0 or 'code_identity' in p['identities']):
        assert all(g['fused'] is True for g in p['optimizer']['param_groups']), 'scientific CUDA optimizer must be fused'
    assert len(p['rng_by_rank'])==1 and p['loader']['logical_batches']==8*u
    m=from_state(p['model'],tiny=tiny);a=trainability(m)
    if not tiny:assert (a['registered'],a['trainable'],a['frozen'])==(124697386,124475904,221482)
    assert frozen_identity(m)==p['frozen_identity']
    o=optimizer_for(m,'cpu');assert p['optimizer_names']==optimizer_names(m,o)
    o.load_state_dict(p['optimizer'])
    for group_index,g in enumerate(o.param_groups):
        assert g['weight_decay']==(.1 if group_index==0 else 0.)
        assert g['betas']==(.9,.95) and g['eps']==1e-8
        assert g['lr']==learning_rate(max(u-1,0))
    for n,v in m.named_parameters():
        assert torch.isfinite(v).all(),n
        if not v.requires_grad:assert v not in o.state
        elif u:
            s=o.state[v];assert int(s['step'])==u,(n,s['step'],u)
            assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all(),n
    if u==0:assert not o.state
    if 'stream_plan' in p['identities'] and not tiny:
        plan=[__import__('json').loads(l) for l in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
        assert p['loader']==(plan[u-1]['after'] if u else plan[0]['before'])
    return dict(passed=True,completed_updates=u,model_tensor_identity=tensor_identity(m.named_parameters()),**a)

def payload(m,o,cursor,u,ledger,identities,resources,device):
    return dict(schema='2d13-L-complete-v1',model=cpu_snapshot(m.state_dict()),optimizer=cpu_snapshot(o.state_dict()),optimizer_names=optimizer_names(m,o),completed_updates=u,accounting=accounting(u),next_lr_index=u,loader=copy.deepcopy(cursor),rng_by_rank=[capture_rng(device)],cuda_seed=1337,world_size=1,evaluation_ledger=copy.deepcopy(ledger),identities=copy.deepcopy(identities),resources=copy.deepcopy(resources),frozen_identity=frozen_identity(m))

def write(path,p,tiny=False):
    t=time.time();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),str(path)
    tmp=path.with_suffix('.writing')
    with tmp.open('wb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
    reopened=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
    audit=validate(reopened,tiny);digest=sha256(tmp);os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    manifest=dict(file=path.name,sha256=digest,bytes=path.stat().st_size,audit=audit,seconds=time.time()-t,completed_updates=p['completed_updates'])
    atomic_json(str(path)+'.manifest.json',manifest);return manifest

class Writer:
    def __init__(self):self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
    def wait(self):
        if self.future is not None:r=self.future.result(timeout=300);self.future=None;return r
    def poll(self):
        if self.future is not None and self.future.done():return self.wait()
    def submit(self,path,p):
        self.wait();self.future=self.pool.submit(write,path,p)

def restore(path,device,expected=None,verify=True):
    manifest=read_json(str(path)+'.manifest.json');assert sha256(path)==manifest['sha256']
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if verify:validate(p)
    if expected is not None:assert p['identities']==expected
    m=from_state(p['model'],device);o=optimizer_for(m,device)
    assert optimizer_names(m,o)==p['optimizer_names']
    if p['completed_updates']==0:
        # CPU u0 is a shape/RNG template with no momentum. Construct a genuinely
        # fresh optimizer on the destination, never import CPU implementation flags.
        assert not p['optimizer']['state'] and not o.state
    else:
        assert all(g['fused'] is True for g in p['optimizer']['param_groups'])
        o.load_state_dict(p['optimizer'])
    if torch.device(device).type=='cuda':assert all(g['fused'] is True for g in o.param_groups)
    restore_rng(p['rng_by_rank'][0],device)
    if torch.device(device).type=='cuda' and p['rng_by_rank'][0]['cuda'] is None:torch.cuda.manual_seed(p['cuda_seed'])
    return m,o,p
