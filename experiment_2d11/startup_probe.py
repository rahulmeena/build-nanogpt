"""Bounded disposable full-size DDP probe; never creates scientific progress."""
import argparse
import copy
from datetime import timedelta
from contextlib import nullcontext
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import FreshH, fresh_base, optimizer_for, tensor_identity
from .common import atomic_json, heartbeat, learning_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 4
    torch.set_float32_matmul_precision('high')
    start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    heartbeat(out, rank, 'DISPOSABLE_CUDA_PROBE', 0, start+480)
    model = FreshH(fresh_base()).to(device)
    assert sum(p.numel() for p in model.parameters()) == 124697386
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 124697382
    initial = tensor_identity(model.named_parameters())
    identities = [None]*4
    dist.all_gather_object(identities, initial)
    assert len(set(identities)) == 1
    optimizer = optimizer_for(model, device)
    wrapped = DDP(model, device_ids=[rank], broadcast_buffers=False,
                  find_unused_parameters=False, static_graph=False)
    generator = torch.Generator(device='cpu').manual_seed(88700+rank)
    xs = [torch.randint(0, 50257, (32, 1025), generator=generator).to(device) for _ in range(4)]
    records = []
    for ordinal, passes in enumerate((2, 2, 3, 2, 3)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.time()
        objectives = torch.zeros(passes, device=device)
        for micro in range(4):
            with wrapped.no_sync() if micro < 3 else nullcontext():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, losses = wrapped(xs[micro][:, :-1], xs[micro][:, 1:],
                                           num_passes=passes, activation_checkpointing=True)
                (loss/4).backward()
            objectives += losses/4
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        hidden = [r.W1.grad.abs().sum() for r in model.routers.values()]
        if ordinal == 0:
            assert all(v.item() == 0 for v in hidden)
        else:
            assert all(v.item() > 0 for v in hidden)
        assert all(p.grad is None and p not in optimizer.state for n,p in model.named_parameters() if n.startswith('g_rec'))
        # Compact all-rank gradient agreement; full numerical reference remains separate.
        compact = torch.stack([p.grad.float().sum() for p in model.parameters() if p.requires_grad])
        minimum, maximum = compact.clone(), compact.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert torch.equal(minimum, maximum)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(ordinal)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time()-started
        records.append(dict(pass_count=passes, seconds=elapsed, loss=loss.item(),
                            pass_losses=objectives.tolist(), grad_norm=norm.item(),
                            rank_gradient_summary_max_error=(maximum-minimum).abs().max().item(),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        heartbeat(out, rank, 'DISPOSABLE_CUDA_PROBE', ordinal+1, start+480)
        atomic_json(out/f'probe-rank{rank}.json', dict(disposable=True, records=records,
                    initial_identity=initial, initial_shared_across_ranks=True,
                    completed_scientific_updates=0, gpu=torch.cuda.get_device_name(),
                    torch_version=torch.__version__, elapsed_seconds=time.time()-start))
    dist.barrier()
    if rank == 0:
        atomic_json(out/'PROBE_COMPLETE.json', dict(passed=True, disposable=True,
                    scientific_updates=0, remaining_preflight_required=True,
                    seconds=time.time()-start))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
