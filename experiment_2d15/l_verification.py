"""Prespecified objective/gradient/AdamW and accumulation checks."""
import copy,time,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .model import LocalOnly,fresh_base,optimizer_for,frozen_identity,from_state
from .checkpoints import cpu_snapshot,payload,write,restore_rng,capture_rng,validate

def tensor_errors(a,b):
    a=a.double().flatten();b=b.double().flatten();d=a-b
    aa=float(a.dot(a));bb=float(b.dot(b));dd=float(d.dot(d));ab=float(a.dot(b))
    return dict(max_abs=float(d.abs().max()) if d.numel() else 0.,diff_sq=dd,ref_sq=bb,actual_sq=aa,dot=ab,relative_l2=(dd/max(bb,1e-300))**.5,zero_actual=int((a==0).sum()),zero_ref=int((b==0).sum()),sign_disagreements=int(((a*b)<0).sum()))

def objective_result(m,opt,x,y,passes,checkpointing,bf16):
    opt.zero_grad(set_to_none=True);m.train();device=x.device
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
        loss=m(x,y,activation_checkpointing=checkpointing)[0] if passes==1 else m.forward_multi_pass(x,targets=y,num_passes=passes,activation_checkpointing=checkpointing)['loss']
    loss.backward();grads={}
    for n,p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
            grads[n]=p.grad.detach().cpu().clone()
        else:assert p.grad is None,n
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
    opt.step()
    return dict(loss=float(loss.detach()),gradients=grads,clip_norm=norm,model=cpu_snapshot(m.state_dict()),optimizer=cpu_snapshot(opt.state_dict()))

def compare_results(a,b,bf16):
    tol=read_json(FROZEN/'TOLERANCES.json')
    np.testing.assert_allclose(a['loss'],b['loss'],atol=tol['cuda_loss_atol'] if bf16 else tol['cpu_atol'],rtol=tol['cuda_loss_rtol'] if bf16 else tol['cpu_rtol'])
    per={};dd=aa=bb=ab=0
    for n,g in a['gradients'].items():
        z=tensor_errors(g,b['gradients'][n]);per[n]=z
        dd+=z['diff_sq'];aa+=z['actual_sq'];bb+=z['ref_sq'];ab+=z['dot']
        if not bf16:torch.testing.assert_close(g,b['gradients'][n],atol=tol['cpu_atol'],rtol=tol['cpu_rtol'])
    rel=(dd/bb)**.5;cos=ab/(aa*bb)**.5
    assert rel<=tol['cuda_gradient_relative_l2'] and cos>=tol['cuda_gradient_cosine'],(rel,cos)
    # Absolute diagnostics retained for near-zero gradients; no unstable per-tensor ratio gate.
    updates={n:tensor_errors(v,b['model'][n]) for n,v in a['model'].items()}
    if bf16:
        assert max(v['max_abs'] for v in updates.values())<=2e-6
        assert abs(a['clip_norm']-b['clip_norm'])/max(b['clip_norm'],1e-30)<=.01
    moments={}
    for i,s in a['optimizer']['state'].items():
        moments[str(i)]={k:tensor_errors(v,b['optimizer']['state'][i][k]) for k,v in s.items()}
    if not bf16:
        for n,v in a['model'].items():torch.testing.assert_close(v,b['model'][n],atol=tol['cpu_atol'],rtol=tol['cpu_rtol'])
        for i,s in a['optimizer']['state'].items():
            for k,v in s.items():torch.testing.assert_close(v,b['optimizer']['state'][i][k],atol=tol['cpu_atol'],rtol=tol['cpu_rtol'])
    return dict(passed=True,loss_single=a['loss'],loss_repeated=b['loss'],loss_max_abs=abs(a['loss']-b['loss']),gradient_relative_l2=rel,gradient_cosine=cos,gradient_max_abs=max(v['max_abs'] for v in per.values()),clip_norm_single=a['clip_norm'],clip_norm_repeated=b['clip_norm'],per_tensor_gradients=per,per_tensor_parameters=updates,optimizer_state= moments,parameter_max_abs=max(v['max_abs'] for v in updates.values()),inactive_gradients_absent=True)

def objective_suite(m,x,y,bf16=False,progress=None):
    t=time.time();original=cpu_snapshot(m.state_dict());opt=optimizer_for(m,x.device)
    initial_frozen=frozen_identity(m);results=[]
    for noninitial in [False,True]:
        m.load_state_dict(original);opt=optimizer_for(m,x.device)
        if noninitial:objective_result(m,opt,x,y,1,True,bf16)
        state=cpu_snapshot(m.state_dict());osnap=cpu_snapshot(opt.state_dict())
        for checkpointing in [False,True]:
            for passes in [2,3]:
                if progress:progress(noninitial,checkpointing,passes)
                m.load_state_dict(state);opt=optimizer_for(m,x.device);opt.load_state_dict(copy.deepcopy(osnap))
                a=objective_result(m,opt,x,y,1,checkpointing,bf16)
                m.load_state_dict(state);opt=optimizer_for(m,x.device);opt.load_state_dict(copy.deepcopy(osnap))
                b=objective_result(m,opt,x,y,passes,checkpointing,bf16)
                r=compare_results(a,b,bf16);r.update(noninitial=noninitial,checkpointing=checkpointing,passes=passes)
                assert frozen_identity(m)==initial_frozen
                results.append(r);del a,b
    m.load_state_dict(original);m.zero_grad(set_to_none=True)
    return dict(passed=True,device=str(x.device),bf16=bf16,probes=results,seconds=time.time()-t)

