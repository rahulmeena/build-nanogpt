"""Historical H's physical partition, full DDP forward, and one averaged clip/step."""
from contextlib import nullcontext
import torch
import torch.distributed as dist
from .common import learning_rate,accounting,pass_count
from .model import assert_optimizer
from experiment_2d11.data import physical_batches

def step(wrapped,model,opt,x,y,u,rank,device,checkpointing=True):
    assert_optimizer(opt)
    passes=1 if model.arm=='L_nf4' else pass_count(u)
    model.train();opt.zero_grad(set_to_none=True)
    total=torch.zeros(passes+1,device=device)
    for micro,(xx,yy) in enumerate(physical_batches(x,y,rank,4,32)):
        with wrapped.no_sync() if micro<3 else nullcontext():
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss,diagnostics=wrapped(xx.to(device),yy.to(device),num_passes=passes,activation_checkpointing=checkpointing)
            (loss/4).backward()
        total[0]+=loss.detach()/4;total[1:]+=diagnostics/4
    dist.all_reduce(total);total/=4
    for n,p in model.named_parameters():assert (p.grad is not None)==p.requires_grad,n
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    for g in opt.param_groups:g['lr']=learning_rate(u-1)
    opt.step();assert_optimizer(opt,u)
    return dict(weighted_multipass_training_loss=float(total[0]),pass_losses=total[1:].tolist(),pass_count=passes,
                grad_norm=float(norm),lr=learning_rate(u-1),lr_index=u-1,world_size=4,microbatch=32,accumulation=4,
                fused=False,foreach=None,**accounting(u,model.arm))
