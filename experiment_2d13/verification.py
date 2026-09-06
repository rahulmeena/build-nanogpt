"""Prespecified objective/gradient/AdamW and accumulation checks."""
import copy,time,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .model import LocalOnly,fresh_base,optimizer_for,frozen_identity,from_state
from .checkpoints import cpu_snapshot,payload,write,restore_rng,capture_rng,validate
from .train import physical_batches,step

def tensor_errors(a,b):
    a=a.double().flatten();b=b.double().flatten();d=a-b
    aa=float(a.dot(a));bb=float(b.dot(b));dd=float(d.dot(d));ab=float(a.dot(b))
    return dict(max_abs=float(d.abs().max()) if d.numel() else 0.,diff_sq=dd,ref_sq=bb,actual_sq=aa,dot=ab,relative_l2=(dd/max(bb,1e-300))**.5,zero_actual=int((a==0).sum()),zero_ref=int((b==0).sum()),sign_disagreements=int(((a*b)<0).sum()))

def objective_result(m,opt,x,y,passes,checkpointing,bf16):
    opt.zero_grad(set_to_none=True);m.train();device=x.device
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
        loss=m(x,y,activation_checkpointing=checkpointing) if passes==1 else m.forward_multi_pass(x,targets=y,num_passes=passes,activation_checkpointing=checkpointing)['loss']
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

def accumulation(m,x,y,bf16=False):
    original=cpu_snapshot(m.state_dict());results=[]
    # Global single-GPU order versus emulated original four-rank averaging.
    for grouping in ['single','original_four_rank']:
        total={n:torch.zeros_like(p,device='cpu') for n,p in m.named_parameters() if p.requires_grad}
        if grouping=='single':owners=[list(range(16))];divisor=16
        else:owners=[[4*i+r for i in range(4)] for r in range(4)];divisor=4
        for indices in owners:
            m.zero_grad(set_to_none=True)
            for index in indices:
                xx=x[index*32:(index+1)*32];yy=y[index*32:(index+1)*32]
                with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=bf16):loss=m(xx,yy,activation_checkpointing=True)
                (loss/divisor).backward()
            for n,p in m.named_parameters():
                if p.requires_grad:total[n]+=p.grad.detach().cpu()/len(owners)
                else:assert p.grad is None
        results.append(total)
    dd=aa=bb=ab=0;per={}
    for n,a in results[0].items():
        z=tensor_errors(a,results[1][n]);per[n]=z;dd+=z['diff_sq'];aa+=z['actual_sq'];bb+=z['ref_sq'];ab+=z['dot']
        torch.testing.assert_close(a,results[1][n],atol=2e-6,rtol=2e-5)
    m.zero_grad(set_to_none=True)
    return dict(passed=True,per_tensor=per,relative_l2=(dd/bb)**.5,cosine=ab/(aa*bb)**.5,rows=list(range(512)),normalization='16 means /16 versus four ranks each four means /4, followed by one rank mean')

def cpu_suite():
    torch.set_num_threads(1);m=LocalOnly(fresh_base(True));x=(torch.arange(140).reshape(2,70)*17+3)%32;y=(x+1)%32
    frozen=frozen_identity(m)
    with patch.object(m,'combine',side_effect=AssertionError('router invoked')),patch.object(m,'project_recurrent_kv',side_effect=AssertionError('recurrent read invoked')):
        objective=objective_suite(m,x,y)
    xx=(torch.arange(512*70).reshape(512,70)*17+3)%32;yy=(xx+1)%32
    norm=accumulation(m,xx,yy)
    assert frozen_identity(m)==frozen
    from experiment_2d12.verification import focused
    # Sealed independent references and causality/reset checks, tiny fixture.
    m.requires_grad_(False);structural=focused(m)
    m.set_condition('H_ALL_OFF');m.base.requires_grad_(True)
    # Check actual new state serializer/reload and optimizer boundary progression.
    o=optimizer_for(m,'cpu');step(m,o,xx,yy,1,'cpu')
    for s in o.state.values():s['step'].fill_(714)
    for g in o.param_groups:g['lr']=learning_rate(713)
    step(m,o,xx,yy,715,'cpu')
    p=payload(m,o,dict(shard=0,position=0,wraps=0,logical_batches=715*8),715,{}, {},{},'cpu')
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'u00715.pt';write(path,p,tiny=True)
        saved=torch.load(path,map_location='cpu',weights_only=False);validate(saved,True)
        step(m,o,xx,yy,716,'cpu');expect=cpu_snapshot(m.state_dict());eo=cpu_snapshot(o.state_dict())
        b=from_state(saved['model'],tiny=True);bo=optimizer_for(b,'cpu');bo.load_state_dict(saved['optimizer']);restore_rng(saved['rng_by_rank'][0])
        step(b,bo,xx,yy,716,'cpu')
        for n,v in b.state_dict().items():torch.testing.assert_close(v,expect[n],atol=2e-6,rtol=2e-5)
        for i,s in bo.state_dict()['state'].items():
            for k,v in s.items():torch.testing.assert_close(v,eo['state'][i][k],atol=2e-6,rtol=2e-5)
    lr={u:learning_rate(u-1) for u in [1,715,716,1000]};assert lr[1]==6e-4/715 and abs(lr[715]-6e-4)<1e-18 and abs(lr[716]-6e-4)<1e-18 and lr[1000]<6e-4
    r=dict(passed=True,objective=objective,accumulation=norm,structural=structural,resume_boundary_passed=True,lr_boundaries=lr)
    atomic_json(PACKAGE/'results/CPU_AUDIT.json',r);print(dict(passed=True,objective_probes=len(objective['probes']),resume_boundary=True),flush=True)
if __name__=='__main__':cpu_suite()
