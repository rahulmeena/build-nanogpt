"""One GPU, sixteen B32 means, one global clip and one AdamW step."""
import time
import numpy as np
import torch
from .common import learning_rate,accounting

def physical_batches(x,y):
    assert x.shape==y.shape and len(x)==512
    for start in range(0,512,32):
        yield torch.as_tensor(x[start:start+32]),torch.as_tensor(y[start:start+32])

def step(m,opt,x,y,u,device='cuda',checkpointing=True,progress=None):
    assert 1<=u<=1000
    device=torch.device(device)
    if device.type=='cuda':assert all(g['fused'] is True for g in opt.param_groups), 'CUDA optimizer fused flag lost'
    m.train();opt.zero_grad(set_to_none=True)
    total=torch.zeros((),device=device)
    for i,(xx,yy) in enumerate(physical_batches(x,y)):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            loss=m(xx.to(device),yy.to(device),activation_checkpointing=checkpointing)
        (loss/16).backward();total+=loss.detach()/16
        if progress:progress(i)
    for n,p in m.named_parameters():
        assert (p.grad is not None)==p.requires_grad,n
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    lr=learning_rate(u-1)
    for g in opt.param_groups:g['lr']=lr
    opt.step()
    return dict(single_pass_training_loss=float(total),grad_norm=float(norm),lr=lr,lr_index=u-1,**accounting(u))
