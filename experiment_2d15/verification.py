"""Independent loss assembly, writer autograd, accumulation, and recovery probes."""
import copy,tempfile,time
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .model import CE1FreeH,fresh_base,from_state,optimizer_for,tensor_identity,trainability
from experiment_2d11.checkpoints import cpu_snapshot,restore_rng
from .train import step

def reference(m,x,y,passes,checkpointing):
    rows=[m.forward_pass(x,targets=y,activation_checkpointing=checkpointing)]
    for _ in range(1,passes):
        previous=rows[-1]
        rows.append(m.forward_pass(x,targets=y,b1_recurrent_source=previous['h12'],
            b3_recurrent_source=previous['h10'],b5_recurrent_source=previous['h8'],
            activation_checkpointing=checkpointing))
    return rows[1]['loss'] if passes==2 else (rows[1]['loss']+rows[2]['loss'])/2

def objective_suite(m,x,y,bf16=False,progress=None):
    original=cpu_snapshot(m.state_dict());records=[]
    for noninitial in (False,True):
        m.load_state_dict(original)
        if noninitial:
            with torch.no_grad():
                for r in m.routers.values():r.W2.add_(.031);r.b2.add_(torch.tensor([.03,-.02],device=x.device))
                m.base.transformer.h[4].attn.c_proj.bias.add_(.017)
        for checkpointing in (False,True):
            for passes in (2,3):
                if progress:progress(noninitial,checkpointing,passes)
                m.train();m.zero_grad(set_to_none=True);captured=[]
                base_forward=m.forward_pass
                def capture(*args,**kwargs):
                    result=base_forward(*args,**kwargs);captured.append(result);return result
                with patch.object(m,'forward_pass',side_effect=capture):
                    with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=bf16):
                        actual,diagnostics=m(x,y,num_passes=passes,activation_checkpointing=checkpointing)
                same=captured[1]['loss'] if passes==2 else .5*captured[1]['loss']+.5*captured[2]['loss']
                assert torch.equal(actual,same)
                edge=torch.autograd.grad(actual,captured[0]['loss'],allow_unused=True,retain_graph=True)[0]
                assert edge is None,'CE1 has a direct objective path'
                assert not diagnostics.requires_grad
                actual.backward()
                ga={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
                assert all(torch.isfinite(g).all() for g in ga.values())
                value=float(actual.detach());del captured,actual,same
                m.zero_grad(set_to_none=True)
                with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=bf16):
                    expected=reference(m,x,y,passes,checkpointing)
                expected.backward();maximum=0.
                for n,p in m.named_parameters():
                    if p.requires_grad:
                        g=p.grad.detach().cpu();maximum=max(maximum,float((ga[n]-g).abs().max()))
                        torch.testing.assert_close(ga[n],g,atol=2e-6,rtol=2e-5)
                        if bf16:assert torch.equal(ga[n],g),n
                    else:assert p.grad is None
                assert value==float(expected.detach())
                records.append(dict(passed=True,noninitial=noninitial,checkpointing=checkpointing,
                    passes=passes,loss_abs_error=abs(value-float(expected.detach())),gradient_max_abs=maximum,
                    direct_ce1_edge_absent=True,all_active_parameter_gradients_present=True))
                del ga,expected
    m.load_state_dict(original);m.zero_grad(set_to_none=True)
    return dict(passed=True,device=str(x.device),bf16=bf16,probes=records)

def temporal_suite(m,x,y,bf16=False):
    records=[];m.train()
    for checkpointing in (False,True):
        for detached in (False,True):
            m.zero_grad(set_to_none=True);rows=[];proxies=[];original=m.forward_pass
            def track(*args,**kwargs):
                result=original(*args,**kwargs);rows.append(result)
                outputs={}
                for name in ('h12','h10','h8'):
                    proxy=result[name].clone()
                    outputs[name]=proxy
                    result[name]=proxy.detach() if detached else proxy
                proxies.append(outputs);return result
            with patch.object(m,'forward_pass',side_effect=track):
                with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=bf16):
                    loss,_=m(x,y,num_passes=3,activation_checkpointing=checkpointing)
            for ce_index,source_passes in ((1,(0,)),(2,(0,1))):
                tensors=[proxies[p][n] for p in source_passes for n in ('h12','h10','h8')]
                gradients=torch.autograd.grad(rows[ce_index]['loss'],tensors,allow_unused=True,retain_graph=True)
                writer_rows=[]
                for (p,n),g in zip([(p,n) for p in source_passes for n in ('h12','h10','h8')],gradients):
                    if detached:
                        assert g is None;norm=0.
                    else:
                        assert g is not None and torch.isfinite(g).all()
                        norm=float(g.float().norm());assert norm>0,(ce_index,p,n)
                        # No future writers can affect earlier predictions; eligible prefix has signal.
                        lag={'h12':1,'h10':31,'h8':63}[n]
                        assert float(g[:,:max(1,x.shape[1]-lag)].float().abs().sum())>0
                    writer_rows.append(dict(source_pass=p+1,writer=n,gradient_norm=norm,absent=g is None))
                records.append(dict(ce_pass=ce_index+1,checkpointing=checkpointing,detached_negative_control=detached,writers=writer_rows))
            del rows,proxies,loss
    m.zero_grad(set_to_none=True)
    return dict(passed=True,probes=records,isolated_writer_proxy_gradients=True)

