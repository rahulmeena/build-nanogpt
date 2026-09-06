"""One fresh arm through u5000, including every prescribed GPU evaluation."""
import argparse,gc,json,os,time,traceback
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import *
from .model import tensor_identity,frozen_identity,load_h
from .checkpoints import reset_original,restore,payload,Writer,preserve_rng,capture_rng,restore_rng,prune
from .evaluate import evaluate,progress
from .train import step
from experiment_2d11.data import LogicalLoader

def run(a):
    configure_cuda_determinism();rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    assert torch.__version__=='2.8.0+cu128' and torch.cuda.device_count()==4
    assert torch.cuda.get_device_name(rank)=='NVIDIA A100-SXM4-80GB'
    dist.init_process_group('nccl',timeout=timedelta(seconds=1800),device_id=device)
    root=a.root/a.arm;root.mkdir(parents=True,exist_ok=True);rolling=a.scratch/a.arm/'rolling';rolling.mkdir(parents=True,exist_ok=True)
    b=read_json(a.root/'controller/binding.json')
    pod=[x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0') if x.startswith(b'RUNPOD_POD_ID=')]
    assert pod==[b['pod_id']]
    pre=read_json(a.root/'controller/BOTH_ARMS_PREFLIGHT_PASSED.json');assert pre['passed']
    code=read_json(FROZEN/'CODE_IDENTITY.json')
    for n,h in code['files'].items():assert sha256(REPO/n)==h,n
    ids=dict(code_identity=identity(code),configuration=identity(read_json(FROZEN/'EXECUTION.json')),panel_identity=read_json(FROZEN/'PANEL.json')['identity'],original_initial_sha256=INIT_SHA,stream_plan=sha256(FROZEN/'stream_plan.jsonl'),expected_batches=sha256(FROZEN/'expected_batches.jsonl'),arm=a.arm)
    candidates=[]
    for d in [root/'checkpoints',rolling]:
        for f in d.glob('u*.pt'):
            mf=Path(str(f)+'.manifest.json')
            if mf.exists() and read_json(mf)['audit']['passed']:candidates.append(f)
    if candidates:
        path=max(candidates,key=lambda x:int(x.stem[1:]));m,o,saved=restore(path,device,rank,a.arm,ids)
        u=saved['completed_updates'];cursor=saved['loader'];ledger=saved['evaluation_ledger'];resources=saved['resources'];rng=saved['rng_by_rank'][rank]
    else:
        m,o,saved=reset_original(a.original,a.arm,device);u=0;cursor=saved['loader'];ledger={};resources=dict(training_seconds=0.,evaluation_seconds=0.,binding=b);rng=capture_rng(device)
    loader=LogicalLoader(a.data,read_json(REPO/'experiment_2d11/frozen/shards.json')['train'],cursor=cursor)
    wrapped=DDP(m,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,static_graph=False)
    restore_rng(rng,device);initial_frozen=frozen_identity(m);del saved
    writer=Writer() if rank==0 else None
    plan=[json.loads(x) for x in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    expected=[json.loads(x) for x in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    assert all(learning_rate(i)==r['lr'] for i,r in enumerate(expected)), 'Linux LR differs from actual H'
    if rank==0 and (root/'metrics.jsonl').exists():
        old=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
        discarded=[r for r in old if r['completed_updates']>u]
        if discarded:
            atomic_bytes(root/f'discarded-{time.time_ns()}.jsonl',b''.join(canonical(r)+b'\n' for r in discarded))
            atomic_bytes(root/'metrics.jsonl',b''.join(canonical(r)+b'\n' for r in old if r['completed_updates']<=u))
    def wait_writer():
        if rank==0:writer.wait()
        dist.barrier()
    def save():
        path=(root/'checkpoints' if u in (0,*MILESTONES) else rolling)/f'u{u:05d}.pt'
        with preserve_rng(device):
            wait_writer();rngs=[None]*4;dist.all_gather_object(rngs,capture_rng(device))
            if rank==0:
                if path.exists():
                    mf=read_json(str(path)+'.manifest.json');assert mf['audit']['model_tensor_identity']==tensor_identity(m.named_parameters())
                else:
                    resources['billed_seconds_at_save']=time.time()-b['billing_start']
                    writer.submit(path,payload(m,o,loader.state_dict(),u,ledger,ids,resources,rngs))
            dist.barrier()
        return path
    def score(candidate,panel,label,path):
        with preserve_rng(device):
            v=evaluate(candidate,panel,a.validation,root/'evaluations',label,path,128,ids['code_identity'],rank,u,a.arm)
            ledger[label]=dict(ce=v['ce'],binding=v['binding'],path=label+'_COMPLETE.json')
            resources['evaluation_seconds']+=v['seconds']
        return v
    mon=read_json(FROZEN/'MONITOR.json');panel=read_json(FROZEN/'PANEL.json')
    while True:
        path=None
        if u%500==0 or u in SCHEDULE:path=save()
        if u in SCHEDULE:
            wait_writer();score(m,mon,f'{a.arm}_monitor_u{u:05d}',path)
        if u in MILESTONES:
            wait_writer()
            score(m,panel,('L_LOCAL' if a.arm=='L_nf4' else 'R_ON')+f'_u{u:05d}',path)
            if a.arm=='R_nf4':
                m.set_condition('H_ALL_OFF');score(m,panel,f'R_ALL_OFF_u{u:05d}',path);m.set_condition('H_ON')
            else:
                # H is a frozen comparator, with no optimizer and no training calls.
                with preserve_rng(device):h=load_h(a.h_dir/f'u{u:05d}.pt',u,device)
                score(h,panel,f'H_ON_u{u:05d}',a.h_dir/f'u{u:05d}.pt')
                h.set_condition('H_ALL_OFF');score(h,panel,f'H_ALL_OFF_u{u:05d}',a.h_dir/f'u{u:05d}.pt')
                del h;gc.collect();torch.cuda.empty_cache()
        if u==5000:break
        started=time.time();u+=1
        x,y,data=loader.next_global(plan[u-1]);assert data==expected[u-1]['data'] and expected[u-1]['lr']==learning_rate(u-1)
        stats=step(wrapped,m,o,x,y,u,rank,device);torch.cuda.synchronize()
        elapsed=time.time()-started;elapsed_tensor=torch.tensor(elapsed,device=device);dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX);elapsed=float(elapsed_tensor)
        resources['training_seconds']+=elapsed
        if rank==0:
            stats.update(data=data,training_seconds=elapsed,cumulative_training_seconds=resources['training_seconds'],timestamp=time.time(),arm=a.arm)
            append_json(root/'metrics.jsonl',stats);writer.poll();progress(root,'UPDATE_COMPLETE',update=u)
            if u%25==0:
                deleted=prune(rolling,root/'export_acks')
                if deleted:append_json(root/'ROLLING_RETENTION.jsonl',dict(time=time.time(),deleted=deleted))
    wait_writer();assert frozen_identity(m)==initial_frozen and loader.state_dict()==plan[-1]['after']
    if rank==0:
        atomic_json(root/'GPU_COMPLETE.json',dict(passed=True,arm=a.arm,completed_updates=5000,logical_targets=2621440000,evaluation_ledger=ledger,resources=resources,terminal_cursor=loader.state_dict(),all_5000_batches_verified=True))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','scratch','original','data','validation','h-dir']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        atomic_json(a.root/a.arm/f'FAILURE-rank{os.environ.get("RANK","unknown")}-{time.time_ns()}.json',dict(time=time.time(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),pod_stop_requested=False));raise
