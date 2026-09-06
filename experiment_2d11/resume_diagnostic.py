"""Disposable GPU diagnosis of strict optimizer-boundary resume; no tolerance changes."""
import argparse,os,time
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import *
from .model import FreshH,fresh_base,optimizer_for,tensor_identity
from .data import LogicalLoader
from .checkpoints import restore_checkpoint,cpu_snapshot
from .train import training_step
p=argparse.ArgumentParser()
for n in ['run','checkpoint','data']:p.add_argument('--'+n,required=True)
a=p.parse_args();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);rank=int(os.environ['RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
dist.init_process_group('nccl',timeout=timedelta(seconds=180),device_id=device)
model=FreshH(fresh_base()).to(device);opt=optimizer_for(model,device)
loader=LogicalLoader(a.data,read_json(FROZEN/'shards.json')['train'])
wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
rows=[];expected=None
for i in range(3):
 restore_checkpoint(model,opt,a.checkpoint,loader,device,rank,{'preflight':True})
 before=tensor_identity(model.named_parameters());state=loader.state_dict();x,y,_=loader.next_global()
 stats=training_step(wrapped,model,opt,x,y,rank,32,device)
 current=cpu_snapshot(model.state_dict())
 errors=sorted([(float((v-expected[n]).abs().max()),n) for n,v in current.items()],reverse=True)[:12] if expected else None
 rows.append(dict(iteration=i,before_identity=before,loader_before=state,loader_after=loader.state_dict(),stats=stats,top_errors=errors))
 expected=current
 atomic_json(Path(a.run)/f'DETERMINISTIC_RESUME_RANK{rank}.json',dict(rows=rows,time=time.time(),scientific_updates=0))
dist.destroy_process_group()
