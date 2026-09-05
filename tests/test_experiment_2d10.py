"""Focused router semantics, causal placement, gradients, and recomputation."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import smoke_test as support
import experiment_2d7_core as prior
import experiment_2d10_core as core
S=support.load_training_symbols()
DEVICE=os.environ.get('EXP2D10_TEST_DEVICE','cpu')
torch.set_num_threads(1)

def model(arm=None):
    torch.manual_seed(210)
    foundation=S['GPT'](S['GPTConfig'](block_size=70,vocab_size=32,n_layer=12,n_head=2,n_embd=16,residual_mode='standard')).to(DEVICE)
    m=core.RetrievalGatingGPT(foundation) if arm else prior.BoundaryAlignmentGPT(foundation,'O')
    m=m.to(DEVICE)
    with torch.no_grad():
        for b,c in zip(core.BLOCKS,(.3,.01,.04)):m.gate_parameter(b).fill_(c)
    if arm:m.enable_router(arm)
    return m

def inputs():
    x=torch.arange(140,device=DEVICE).reshape(2,70)%32
    return x,(x+1)%32

def run(m,x,y=None,mode='parallel',checkpoint=False):
    if mode=='parallel':
        r=m.forward_multi_pass(x,targets=y,num_passes=2,activation_checkpointing=checkpoint)
        return r['logits'],r.get('loss')
    dtype=torch.bfloat16 if torch.is_autocast_enabled(DEVICE) else torch.float32
    cache=m.init_incremental_state(x.size(0),device=DEVICE,dtype=dtype)
    rows=[]
    for i in range(x.size(1)):
        z,cache=m.incremental_step(x[:,i],cache); rows.append(z)
    z=torch.cat(rows,1)
    return z,None if y is None else F.cross_entropy(z.float().flatten(0,1),y.flatten())

def close(a,b):torch.testing.assert_close(a.float(),b.float(),rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('mode',['parallel','incremental'])
def test_zero_t_parent_outputs_logits_ce(bf16,mode):
    p,t=model(),model('T')
    x,y=inputs(); results=[]
    with torch.no_grad(),torch.autocast(DEVICE,dtype=torch.bfloat16,enabled=bf16):
        for m in (p,t):
            projections=[]
            hooks=[m.base.transformer.h[b].attn.c_proj.register_forward_pre_hook(lambda mod,args:projections.append(args[0].clone())) for b in core.BLOCKS]
            z,ce=run(m,x,y,mode); results.append((z,ce,projections))
            for h in hooks:h.remove()
    close(results[0][0],results[1][0]); assert abs(float(results[0][1]-results[1][1]))<=1e-6
    for a,b in zip(results[0][2],results[1][2]):close(a,b)

@pytest.mark.parametrize('bf16',[False,True])
def test_formula_empty_memory_and_real_zero_values(bf16):
    t,h=model('T'),model('H')
    dtype=torch.bfloat16 if bf16 else torch.float32
    residual=torch.randn(2,70,16,device=DEVICE)
    q,local,rec=[torch.randn(2,2,70,8,device=DEVICE,dtype=dtype) for _ in range(3)]
    for b in core.BLOCKS:
        assert torch.equal(t.routers[str(b)].W1,h.routers[str(b)].W1)
        mask=h.recurrent_mask(b,70,70,DEVICE); bank=SimpleNamespace(valid_mask=mask)
        combined,coefficient,d=h.combine(residual,q,local,rec,b,bank)
        c=h.gate_parameter(b).detach().float().tanh()
        weights=torch.stack((1/(1+c),c/(1+c)))
        available=mask.any(-1).reshape(1,1,70,1)
        expected=torch.where(available,weights[0].to(dtype)*local+weights[1].to(dtype)*rec,local)
        close(combined,expected)
        close(d['lambda_L']+d['lambda_R'],torch.ones_like(d['lambda_L']))
        assert torch.equal(coefficient,d['lambda_R'].to(dtype).unsqueeze(1))
        empty=h.combine(residual,q,local,rec*0,b,None)[0]
        assert torch.equal(empty,local)
        zero_values=h.combine(residual,q,local,rec*0,b,bank)[0]
        close(zero_values,torch.where(available,weights[0].to(dtype)*local,local))
        tc,tg,td=t.combine(residual,q,local,rec,b,bank)
        assert torch.equal(tg,t.gate_parameter(b).tanh().to(dtype).expand_as(tg))

@pytest.mark.parametrize('arm',['T','H'])
@pytest.mark.parametrize('mode',['parallel','incremental'])
def test_nonzero_causality_row_isolation(arm,mode):
    m=model(arm)
    with torch.no_grad():
        for r in m.routers.values():r.W2.normal_(0,.02)
    x,y=inputs(); suffix=x.clone();suffix[:,66:]=(suffix[:,66:]+7)%32
    row=x.clone();row[1]=(row[1]+9)%32
    with torch.no_grad():
        a=run(m,x,mode=mode)[0]; b=run(m,suffix,mode=mode)[0]; c=run(m,row,mode=mode)[0]
    close(a[:,:66],b[:,:66]);close(a[0],c[0])

@pytest.mark.parametrize('arm',['T','H'])
def test_gradients_initial_hidden_then_output_update_and_checkpoint(arm):
    m=model(arm);x,y=inputs();opt=torch.optim.AdamW(m.parameters(),lr=3e-5)
    run(m,x,y,checkpoint=True)[1].backward()
    for r in m.routers.values():
        assert r.W2.grad.isfinite().all() and r.W2.grad.abs().sum()>0
        assert r.W1.grad is not None and r.W1.grad.eq(0).all()
        if arm=='H':assert r.b2.grad.isfinite().all() and r.b2.grad.abs().sum()>0
    assert m.g_rec_b6.grad is None
    if arm=='H':assert all(m.gate_parameter(b).grad is None for b in core.BLOCKS)
    opt.step();opt.zero_grad(set_to_none=True)
    clone=copy.deepcopy(m)
    run(m,x,y,checkpoint=True)[1].backward()
    run(clone,x,y,checkpoint=False)[1].backward()
    for n,p in m.named_parameters():
        other=dict(clone.named_parameters())[n]
        if p.grad is None:assert other.grad is None
        else:close(p.grad,other.grad)
    for r in m.routers.values():assert r.W1.grad.isfinite().all() and r.W1.grad.abs().sum()>0
    # Differentiable current projected q and both completed branch outputs.
    for b in core.BLOCKS:
        q,l,r=[torch.randn(2,2,70,8,device=DEVICE,requires_grad=True) for _ in range(3)]
        residual=torch.randn(2,70,16,device=DEVICE,requires_grad=True)
        bank=SimpleNamespace(valid_mask=m.recurrent_mask(b,70,70,DEVICE))
        m.combine(residual,q,l,r,b,bank)[0].square().mean().backward()
        for v in (q,l,r):assert v.grad.isfinite().all() and v.grad.abs().sum()>0
        source=torch.randn(2,70,16,device=DEVICE,requires_grad=True)
        output,_=m._parallel_special_block(residual,b,source,None,None,False)
        output.square().mean().backward()
        assert source.grad.isfinite().all() and source.grad[:,:6].abs().sum()>0
