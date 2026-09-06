"""Complete four-rank states, tied storage, atomic saves and exact restoration."""
import copy,os,time,tempfile,shutil
from concurrent.futures import ThreadPoolExecutor
import torch
from .common import *
from .model import from_state,optimizer_for,optimizer_names,tensor_identity,trainability,frozen_identity,assert_optimizer
from experiment_2d11.checkpoints import capture_rng,restore_rng,preserve_rng,cpu_snapshot

def validate(p,tiny=False):
    assert p['schema']=='2d15-complete-v1' and p['world_size']==4 and p['arm'] in ARMS
    u=p['completed_updates'];assert p['accounting']==accounting(u,p['arm']) and p['next_lr_index']==u
    assert p['next_pass_count']==(None if u==5000 else 1 if p['arm']=='L_nf4' else pass_count(u+1))
    assert len(p['rng_by_rank'])==4 and p['loader']['logical_batches']==8*u
    m=from_state(p['model'],p['arm'],tiny=tiny);a=trainability(m)
    assert frozen_identity(m)==p['frozen_identity']
    if not tiny:assert a['registered']==124697386 and a['trainable']==(124475904 if p['arm']=='L_nf4' else 124697382)
    o=optimizer_for(m,'cpu');assert p['optimizer_names']==optimizer_names(m,o)
    o.load_state_dict(p['optimizer']);assert_optimizer(o,u)
    for n,v in m.named_parameters():
        assert torch.isfinite(v).all(),n
        if not v.requires_grad:assert v not in o.state
        elif u:
            s=o.state[v];assert int(s['step'])==u,(n,u)
            assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all(),n
    if u==0:
        assert not o.state
        if not tiny:assert tensor_identity(m.named_parameters())==FULL_INIT_SHA
    if not tiny:
        plan=[__import__('json').loads(l) for l in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
        assert p['loader']==(plan[u-1]['after'] if u else plan[0]['before'])
    return dict(passed=True,arm=p['arm'],completed_updates=u,model_tensor_identity=tensor_identity(m.named_parameters()),**a)

def payload(m,o,cursor,u,ledger,ids,resources,rngs):
    state=cpu_snapshot(m.state_dict())
    # Keep the actual tied storage in the serialized artifact as well as in memory.
    state['base.lm_head.weight']=state['base.transformer.wte.weight']
    return dict(schema='2d15-complete-v1',arm=m.arm,model=state,optimizer=cpu_snapshot(o.state_dict()),optimizer_names=optimizer_names(m,o),completed_updates=u,accounting=accounting(u,m.arm),next_lr_index=u,next_pass_count=None if u==5000 else 1 if m.arm=='L_nf4' else pass_count(u+1),loader=copy.deepcopy(cursor),rng_by_rank=cpu_snapshot(rngs),cuda_seed=1337,world_size=4,evaluation_ledger=copy.deepcopy(ledger),identities=copy.deepcopy(ids),resources=copy.deepcopy(resources),frozen_identity=frozen_identity(m))

def write(path,p,tiny=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    started=time.time()
    stage=Path(os.environ.get('EXP2D15_STAGE','/tmp/exp2d15-stage'));stage.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=stage) as d:
        f=Path(d)/'state.pt'
        with f.open('wb') as out:torch.save(p,out);out.flush();os.fsync(out.fileno())
        reloaded=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
        audit=validate(reloaded,tiny);digest=sha256(f);del reloaded
        temp=path.with_suffix('.writing')
        with f.open('rb') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20);dst.flush();os.fsync(dst.fileno())
        assert sha256(temp)==digest;os.replace(temp,path)
    manifest=dict(file=path.name,path=str(path),sha256=digest,bytes=path.stat().st_size,audit=audit,seconds=time.time()-started,completed_updates=p['completed_updates'],arm=p['arm'])
    atomic_json(str(path)+'.manifest.json',manifest);return manifest

class Writer:
    def __init__(self):self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
    def wait(self):
        if self.future is not None:r=self.future.result();self.future=None;return r
    def poll(self):
        if self.future is not None and self.future.done():return self.wait()
    def submit(self,path,p):self.wait();self.future=self.pool.submit(write,path,p)

def restore(path,device,rank,arm,ids=None):
    manifest=read_json(str(path)+'.manifest.json');assert sha256(path)==manifest['sha256'] and manifest['audit']['passed']
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert p['arm']==arm
    if ids is not None:assert p['identities']==ids
    m=from_state(p['model'],arm,device);o=optimizer_for(m,device)
    assert optimizer_names(m,o)==p['optimizer_names'];o.load_state_dict(p['optimizer']);assert_optimizer(o,p['completed_updates'])
    restore_rng(p['rng_by_rank'][rank],device)
    return m,o,p

def reset_original(path,arm,device):
    import random,numpy as np
    assert sha256(path)==INIT_SHA
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==0 and not p['optimizer']['state']
    m=from_state(p['model'],arm,device);o=optimizer_for(m,device);assert not o.state
    assert tensor_identity(m.named_parameters())==FULL_INIT_SHA and tensor_identity(m.base.named_parameters())==BASE_SHA
    torch.set_rng_state(p['cpu_rng']);random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
    if device.type=='cuda':torch.cuda.manual_seed(p['cuda_seed'])
    return m,o,p

def prune(rolling,ack_dir):
    paths=sorted(Path(rolling).glob('u*.pt'))
    valid=[p for p in paths if Path(str(p)+'.manifest.json').exists()]
    removed=[]
    for p in valid[:-2]:
        newer=valid[-2:]
        if not all((Path(ack_dir)/(q.name+'.json')).exists() and read_json(Path(ack_dir)/(q.name+'.json'))['sha256']==read_json(str(q)+'.manifest.json')['sha256'] for q in [p,*newer]):continue
        assert sha256(p)==read_json(str(p)+'.manifest.json')['sha256']
        p.unlink();removed.append(p.name)
    return removed
