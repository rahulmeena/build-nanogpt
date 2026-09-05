"""Real four-process Gloo rehearsal of the complete H forward/backward graph."""
import copy
from datetime import timedelta
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import atomic_json
from .model import FreshH,fresh_base,optimizer_for
from .train import training_step


def main():
    rank=int(os.environ['RANK']);torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=90))
    model=FreshH(fresh_base(True));wrapped=DDP(model,broadcast_buffers=False,static_graph=False)
    optimizer=optimizer_for(model,'cpu');rows=[]
    x=(np.arange(32*70).reshape(32,70)%32).astype(np.int64);y=(x+1)%32
    for u in [31,32,33]:
        ref=copy.deepcopy(model);opt=optimizer_for(ref,'cpu');opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        stats=training_step(wrapped,model,optimizer,x,y,rank,u,torch.device('cpu'),micro_b=2)
        opt.zero_grad(set_to_none=True)
        for start in range(0,32,2):
            loss,_=ref(torch.from_numpy(x[start:start+2]),torch.from_numpy(y[start:start+2]),num_passes=3 if u%32==0 else 2)
            (loss/16).backward()
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.requires_grad],1.)
        for group in opt.param_groups:group['lr']=stats['lr']
        opt.step()
        error=max((p-q).abs().max().item() for p,q in zip(model.parameters(),ref.parameters()))
        assert error<2e-6,error
        rows.append(dict(update=u,max_parameter_error=error))
    if rank==0:atomic_json(sys.argv[1],dict(passed=True,world_size=4,backend='gloo',rows=rows,cuda_established=False))
    dist.destroy_process_group()


if __name__=='__main__':main()
