"""One real B32 x 16 accumulated update, with CE1-free attached loss."""
import torch
from .common import learning_rate,accounting,pass_count

def physical_batches(x,y):
    assert x.shape==y.shape and len(x)==512
    for start in range(0,512,32):
        yield torch.as_tensor(x[start:start+32]),torch.as_tensor(y[start:start+32])

def step(model,opt,x,y,u,device='cuda',checkpointing=True,progress=None):
    device=torch.device(device);passes=pass_count(u)
    if device.type=='cuda':assert all(g['fused'] is True for g in opt.param_groups)
    model.train();opt.zero_grad(set_to_none=True)
    total=torch.zeros(passes+1,device=device)
    for i,(xx,yy) in enumerate(physical_batches(x,y)):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            loss,diagnostics=model(xx.to(device),yy.to(device),num_passes=passes,
                                   activation_checkpointing=checkpointing)
        (loss/16).backward()
        total[0]+=loss.detach()/16;total[1:]+=diagnostics/16
        if progress:progress(i)
    for name,p in model.named_parameters():
        assert (p.grad is not None)==p.requires_grad,name
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    lr=learning_rate(u-1)
    for group in opt.param_groups:group['lr']=lr
    opt.step()
    return dict(weighted_multipass_training_loss=float(total[0]),pass_losses=total[1:].tolist(),
                pass_count=passes,loss_weights=[0,1] if passes==2 else [0,.5,.5],
                microbatches=16,loss_divisor=16,grad_norm=float(norm),lr=lr,lr_index=u-1,**accounting(u))
