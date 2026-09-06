"""One bounded CUDA lifecycle; all reporting and resampling remain local."""
import argparse,gc,os,time,traceback,signal
import numpy as np
import torch
from .common import *
from .model import tensor_identity,frozen_identity,load_h
from .checkpoints import restore,payload,Writer,write,preserve_rng
from .train import step
from .evaluate import evaluate,progress
from .preflight import run as preflight
from experiment_2d11.data import LogicalLoader

def run(args):
    root=args.run;root.mkdir(parents=True,exist_ok=True);configure_cuda_determinism()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    binding=read_json(args.binding);assert time.time()<binding['hard_deadline']-1800
    code=read_json(FROZEN/'CODE_IDENTITY.json')
    for n,h in code['files'].items():assert sha256(REPO/n)==h,n
    code_id=identity(code)
    assert sha256(args.validation)==VAL_SHA and sha256(args.h)==H_SHA
    shards=read_json(REPO/'experiment_2d11/frozen/shards.json')
    for s in shards['train'][:6]:assert sha256(args.data/s['filename'])==s['sha256']
    original=torch.load(args.original,map_location='cpu',weights_only=False,mmap=True)
    assert sha256(args.original)==INIT_SHA and original['completed_updates']==0
    del original
    progress(root,'INPUTS_VERIFIED')
    if args.resume:
        pre=read_json(root/'PREFLIGHT.json');assert pre['passed']
    else:pre=preflight(args.initial,args.h,root,binding)
    # Restore all scientific state after disposable checks. u0 freezes CUDA RNG.
    m,o,saved=restore(args.resume or args.initial,'cuda')
    assert tensor_identity(m.base.named_parameters())==BASE_SHA if not args.resume else True
    if args.resume:assert saved['identities']['code_identity']==code_id
    identities=dict(saved['identities']);identities['code_identity']=code_id
    loader=LogicalLoader(args.data,shards['train'],cursor=saved['loader'])
    ledger=saved['evaluation_ledger'];resources=saved['resources'];resources['preflight_seconds']=pre['seconds']
    u=saved['completed_updates'];initial_frozen=saved['frozen_identity'];del saved
    if args.resume and (root/'metrics.jsonl').exists():
        rows=[__import__('json').loads(z) for z in (root/'metrics.jsonl').read_text().splitlines()]
        discarded=[r for r in rows if r['completed_updates']>u]
        if discarded:
            name=f'metrics-discarded-{int(time.time())}.jsonl'
            atomic_bytes(root/name,b''.join(canonical(r)+b'\n' for r in discarded))
            atomic_bytes(root/'metrics.jsonl',b''.join(canonical(r)+b'\n' for r in rows if r['completed_updates']<=u))
            append_json(root/'REPLAY_LOG.jsonl',dict(time=time.time(),resume_update=u,discarded_updates=len(discarded),discarded_metrics=name))
    batch=pre['batch_size'];mon=read_json(FROZEN/'MONITOR.json');panel=read_json(FROZEN/'PANEL.json')
    plan=[__import__('json').loads(l) for l in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    expected=[__import__('json').loads(l) for l in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    writer=Writer();checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
    cancelled=False
    def terminate(signum,frame):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def save():
        with preserve_rng('cuda'):
            writer.wait();path=checkpoints/f'u{u:05d}.pt'
            resources['cumulative_billed_seconds_at_save']=time.time()-binding['billing_start']
            if path.exists():
                v=read_json(str(path)+'.manifest.json');assert v['audit']['completed_updates']==u
                assert v['audit']['model_tensor_identity']==tensor_identity(m.named_parameters())
            else:
                progress(root,'CHECKPOINT_SNAPSHOT',update=u)
                writer.submit(path,payload(m,o,loader.state_dict(),u,ledger,identities,resources,'cuda'))
            return path
    while True:
        if u in SCHEDULE:
            path=save();writer.wait()
            key=f'L_monitor_u{u:05d}'
            result=evaluate(m,mon,args.validation,root/'evaluations',key,path,batch,binding['hard_deadline']-300,code_id)
            ledger[key]=dict(ce=result['ce'],path=key+'_COMPLETE.json',binding=result['binding'])
            resources['evaluation_seconds']+=result['seconds']
        if u==1000:break
        assert not cancelled and not (root/'STOP_REQUEST').exists(),'controller requested termination'
        if time.time()>=binding['hard_deadline']-1800:
            save();writer.wait();raise TimeoutError('training cutoff preserves evaluation/export reserve')
        before=time.time();u+=1
        x,y,data=loader.next_global(plan[u-1]);assert data==expected[u-1]['data'],f'H batch mismatch before update {u}'
        stats=step(m,o,x,y,u,progress=lambda i:progress(root,'TRAIN',update=u,micro=i))
        torch.cuda.synchronize();elapsed=time.time()-before;resources['training_seconds']+=elapsed
        stats.update(data=data,training_seconds=elapsed,cumulative_training_seconds=resources['training_seconds'],timestamp=time.time())
        assert stats['lr']==expected[u-1]['lr'];append_json(root/'metrics.jsonl',stats)
        writer.poll();progress(root,'UPDATE_COMPLETE',update=u)
    writer.wait();path=checkpoints/'u01000.pt'
    assert loader.state_dict()==plan[-1]['after'] and frozen_identity(m)==initial_frozen
    l=evaluate(m,panel,args.validation,root/'evaluations','L524M',path,batch,binding['hard_deadline']-300,code_id)
    resources['evaluation_seconds']+=l['seconds'];del m,o;gc.collect();torch.cuda.empty_cache()
    h=load_h(args.h,'cuda')
    r=evaluate(h,panel,args.validation,root/'evaluations','H524M',args.h,batch,binding['hard_deadline']-300,code_id)
    resources['evaluation_seconds']+=r['seconds']
    assert sha256(args.original)==INIT_SHA and sha256(args.h)==H_SHA
    atomic_json(root/'GPU_WORK_FINISHED.json',dict(passed=True,**accounting(u),final_cursor=loader.state_dict(),frozen_tensors_unchanged=True,resources=resources,final_sequence_condition_rows=8192,final_target_predictions=8388608,monitor_target_predictions=6553600,total_new_scientific_target_predictions=14942208,batch_size=batch,scientific_training_arms=1,initialization_only_untrained=True,original_H_evaluation_only=True,initial_original_sha256=INIT_SHA,H_source_sha256=H_SHA))
    progress(root,'FINISHED')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['run','binding','initial','original','h','data','validation']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        atomic_json(a.run/'FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
