"""Immutable CPU snapshots, atomic writes, strict reopen, and exact restoration."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from .common import atomic_json, sha256, accounting, read_json, PRESERVE
from .model import FreshH, fresh_base, optimizer_for, optimizer_names, tensor_identity, uninitialized_h


def capture_rng(device=None):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device is not None and torch.device(device).type=='cuda' else None)


def restore_rng(state,device=None):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'],device)


@contextmanager
def preserve_rng(device=None):
    before=capture_rng(device)
    try:
        yield
    finally:
        restore_rng(before,device)


def cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):
        return {k:cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def validate_payload(payload,tiny=False):
    u=payload['completed_updates']
    assert payload['accounting']==accounting(u)
    model=uninitialized_h(tiny)
    model.load_state_dict(payload['model'],strict=True,assign=True)
    assert torch.equal(model.base.lm_head.weight,model.base.transformer.wte.weight)
    model.base.lm_head.weight=model.base.transformer.wte.weight
    opt=optimizer_for(model,'cpu')
    assert payload['optimizer_names']==optimizer_names(model,opt)
    opt.load_state_dict(payload['optimizer'])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for name,p in model.named_parameters():
        if not p.requires_grad:
            assert p not in opt.state and p.item()==0
        elif u:
            state=opt.state[p]
            assert int(state['step'])==u,(name,int(state['step']),u)
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    assert payload['loader']['logical_batches']==8*u
    assert len(payload['rng_by_rank'])==payload['world_size']
    return dict(passed=True,completed_updates=u,registered_parameters=sum(p.numel() for p in model.parameters()),
                active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                model_tensor_identity=tensor_identity(model.named_parameters()),optimizer_groups_verified=True)


def write_checkpoint(path,payload,tiny=False):
    start=time.time();path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError('immutable checkpoint already exists: '+str(path))
    temp=path.with_suffix('.writing')
    with temp.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    # Reopen the serialized file, rather than validating just the in-memory object.
    reopened=torch.load(temp,map_location='cpu',weights_only=False)
    audit=validate_payload(reopened,tiny=tiny)
    digest=sha256(temp)
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    manifest=dict(file=path.name,persistent_path=str(path),bytes=path.stat().st_size,sha256=digest,
                  audit=audit,accounting=accounting(payload['completed_updates']),
                  write_verify_seconds=time.time()-start,created_at=time.time())
    atomic_json(str(path)+'.manifest.json',manifest)
    return manifest


class CheckpointWriter:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='immutable-checkpoint')
        self.future=None

    def poll(self):
        if self.future is not None and self.future.done():
            result=self.future.result();self.future=None;return result

    def wait(self,timeout=300):
        if self.future is None:return None
        result=self.future.result(timeout=timeout);self.future=None;return result

    def submit(self,path,payload,tiny=False):
        if self.future is not None:
            raise RuntimeError('checkpoint writer overlap exceeds bounded queue')
        self.future=self.pool.submit(write_checkpoint,path,payload,tiny)


def make_payload(model,opt,loader,u,ledger,rngs,identities,resources):
    return dict(schema='2d11-complete-v1',model=cpu_snapshot(model.state_dict()),
                optimizer=cpu_snapshot(opt.state_dict()),optimizer_names=optimizer_names(model,opt),
                completed_updates=u,accounting=accounting(u),loader=copy.deepcopy(loader.state_dict()),
                rng_by_rank=cpu_snapshot(rngs),world_size=len(rngs),evaluation_ledger=copy.deepcopy(ledger),
                identities=copy.deepcopy(identities),resources=copy.deepcopy(resources),
                next_lr_index=u,next_pass_count=None if u==19072 else (3 if (u+1)%32==0 else 2))


def restore_initial(model,opt,path,loader,device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    assert payload['optimizer_names']==optimizer_names(model,opt)
    opt.load_state_dict(payload['optimizer'])
    assert not opt.state
    loader.load_state_dict(payload['loader'])
    torch.set_rng_state(payload['cpu_rng']);random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng'])
    if torch.device(device).type=='cuda':torch.cuda.manual_seed(payload['cuda_seed'])
    return dict(completed_updates=0,evaluation_ledger={},resources={})


def restore_checkpoint(model,opt,path,loader,device,rank,identities):
    manifest=read_json(str(path)+'.manifest.json')
    assert sha256(path)==manifest['sha256'] and manifest['audit']['passed']
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['identities']==identities
    assert payload['optimizer_names']==optimizer_names(model,opt)
    assert payload['world_size']==4
    model.load_state_dict(payload['model'],strict=True);opt.load_state_dict(payload['optimizer'])
    loader.load_state_dict(payload['loader']);restore_rng(payload['rng_by_rank'][rank],device)
    return payload


def prune_verified_rolling(directory,acknowledgments):
    """Only this run's redundant rolling files, after a local SHA acknowledgment."""
    directory=Path(directory);acks=Path(acknowledgments)
    paths=sorted(directory.glob('u*.pt'))
    rolling=[p for p in paths if int(p.stem[1:]) not in PRESERVE]
    retained=rolling[-2:]
    if len(retained)<2:return []
    for p in retained:
        manifest=read_json(str(p)+'.manifest.json')
        ack=acks/(p.name+'.json')
        if not ack.exists() or read_json(ack).get('sha256')!=manifest['sha256']:
            return []
    deleted=[]
    for p in rolling[:-2]:
        m=read_json(str(p)+'.manifest.json')
        ack=acks/(p.name+'.json')
        if ack.exists() and read_json(ack).get('sha256')==m['sha256']:
            p.unlink();deleted.append(dict(file=p.name,sha256=m['sha256']))
    return deleted
