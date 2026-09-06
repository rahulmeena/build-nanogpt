"""One R trajectory and five fixed evaluations; no CPU report work on the pod."""
import argparse,gc,json,os,signal,time,traceback
import torch
from .common import *
from .model import tensor_identity,frozen_identity,load_h
from .checkpoints import restore,payload,Writer,preserve_rng,validate
from .train import step
from .evaluate import evaluate,progress
from .preflight import run as preflight
from .resources import capacity
from . import comparator
from experiment_2d11.data import LogicalLoader

def run(args):
    root=args.run;root.mkdir(parents=True,exist_ok=True);configure_cuda_determinism()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    binding=read_json(args.binding);assert time.time()<binding['hard_deadline']-1800
    # SSH child environments omit Runpod metadata; PID 1 retains the provider value.
    pod_ids=[entry.split(b'=',1)[1].decode() for entry in Path('/proc/1/environ').read_bytes().split(b'\0')
             if entry.startswith(b'RUNPOD_POD_ID=')]
    assert pod_ids==[binding['pod_id']]
    code=read_json(FROZEN/'CODE_IDENTITY.json')
    for name,digest in code['files'].items():assert sha256(REPO/name)==digest,name
    code_id=identity(code);progress(root,'VERIFYING_INPUTS')
    assert sha256(args.validation)==VAL_SHA and sha256(args.h)==H_SHA and sha256(args.original)==INIT_SHA
    shards=read_json(REPO/'experiment_2d11/frozen/shards.json')
    for row in shards['train'][:6]:assert sha256(args.data/row['filename'])==row['sha256']
    space=capacity(root.parent,args.l_run);atomic_json(root/'RESERVATION.json',space)
    assert space['passed'],'shared quota/reservation shortfall'
    if args.resume:pre=read_json(root/'PREFLIGHT.json');assert pre['passed']
    else:pre=preflight(args.initial,args.h,root,binding)
    m,o,saved=restore(args.resume or args.initial,'cuda')
    if not args.resume:assert tensor_identity(m.named_parameters())==FULL_INIT_SHA
    identities=dict(saved['identities']);identities['code_identity']=code_id
    if args.resume:assert saved['identities']==identities
    loader=LogicalLoader(args.data,shards['train'],cursor=saved['loader'])
    ledger=saved['evaluation_ledger'];resources=saved['resources'];resources['preflight_seconds']=pre['seconds']
    resources['binding']=binding;resources['reservation']=space
    u=saved['completed_updates'];initial_frozen=saved['frozen_identity'];del saved
    mon=read_json(FROZEN/'MONITOR.json');panel=read_json(FROZEN/'PANEL.json')
    plan=[json.loads(z) for z in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    expected=[json.loads(z) for z in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    writer=Writer();checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
    if args.resume and (root/'metrics.jsonl').exists():
        old=[json.loads(z) for z in (root/'metrics.jsonl').read_text().splitlines()]
        discarded=[r for r in old if r['completed_updates']>u]
        if discarded:
            atomic_bytes(root/f'discarded-{int(time.time())}.jsonl',b''.join(canonical(r)+b'\n' for r in discarded))
            atomic_bytes(root/'metrics.jsonl',b''.join(canonical(r)+b'\n' for r in old if r['completed_updates']<=u))
    cancelled=False
    def terminate(*args):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def save():
        with preserve_rng('cuda'):
            writer.wait();path=checkpoints/f'u{u:05d}.pt'
            resources['cumulative_billed_seconds_at_save']=time.time()-binding['billing_start']
            if path.exists():
                v=read_json(str(path)+'.manifest.json')
                assert v['audit']['completed_updates']==u and v['audit']['model_tensor_identity']==tensor_identity(m.named_parameters())
            else:
                space=capacity(root.parent,args.l_run);atomic_json(root/'RESERVATION.json',space)
                assert space['passed'],'shared-volume reservation exhausted'
                resources['reservation']=space
                progress(root,'CHECKPOINT_SNAPSHOT',update=u)
                writer.submit(path,payload(m,o,loader.state_dict(),u,ledger,identities,resources,'cuda'))
            return path
    def score(model,panel,label,path):
        value=evaluate(model,panel,args.validation,root/'evaluations',label,path,128,binding['hard_deadline']-300,code_id)
        ledger[label]=dict(ce=value['ce'],path=label+'_COMPLETE.json',binding=value['binding'])
        resources['evaluation_seconds']+=value['seconds'];return value
    while True:
        if u in SCHEDULE:
            path=save();writer.wait();score(m,mon,f'R_monitor_u{u:05d}',path)
        if u==1000:break
        assert not cancelled and not (root/'STOP_REQUEST').exists(),'controller requested termination'
        if time.time()>=binding['hard_deadline']-3600:
            save();writer.wait();raise TimeoutError('training cutoff preserves required endpoint/export work')
        t=time.time();u+=1
        x,y,data=loader.next_global(plan[u-1]);assert data==expected[u-1]['data'],f'H batch mismatch BEFORE update {u}'
        stats=step(m,o,x,y,u,progress=lambda i:progress(root,'TRAIN',update=u,micro=i))
        torch.cuda.synchronize();seconds=time.time()-t;resources['training_seconds']+=seconds
        stats.update(data=data,training_seconds=seconds,cumulative_training_seconds=resources['training_seconds'],timestamp=time.time())
        assert stats['lr']==expected[u-1]['lr'];append_json(root/'metrics.jsonl',stats)
        writer.poll();progress(root,'UPDATE_COMPLETE',update=u)
        if u%100==0:
            # Descriptive rates affect only technical budget integrity, never quality decisions.
            rates=[json.loads(z)['training_seconds'] for z in (root/'metrics.jsonl').read_text().splitlines()][-100:]
            projection=(sum(rates)/len(rates))*(1000-u)*1.2+pre['projection']['evaluation_seconds']+1800
            atomic_json(root/'LATEST_PROJECTION.json',dict(update=u,remaining_seconds=projection,remaining_budget=binding['hard_deadline']-time.time()))
            if time.time()+projection>binding['hard_deadline']-300:
                save();writer.wait();raise TimeoutError('measured remaining workload exceeds cumulative ceiling')
    writer.wait();path=checkpoints/'u01000.pt'
    assert loader.state_dict()==plan[-1]['after'] and frozen_identity(m)==initial_frozen
    endpoint=validate(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
    atomic_json(root/'R_TRAINING_COMPLETE.json',dict(passed=True,**accounting(u),checkpoint=read_json(str(path)+'.manifest.json'),
        audit=endpoint,terminal_cursor=loader.state_dict(),full_stream_verified=True,loss_ce1_excluded=True))
    score(m,panel,'R_ON',path);m.set_condition('H_ALL_OFF');score(m,panel,'R_ALL_OFF',path)
    del m,o;gc.collect();torch.cuda.empty_cache()
    h=load_h(args.h,'cuda');score(h,panel,'H_ON',args.h)
    h.set_condition('H_ALL_OFF');score(h,panel,'H_ALL_OFF',args.h);del h;gc.collect();torch.cuda.empty_cache()
    progress(root,'CHECKING_L_COMPLETION')
    audit=comparator.ready(args.l_run)
    if audit:
        atomic_json(root/'L_PROVENANCE.json',audit);l=comparator.load(audit);score(l,panel,'L_LOCAL',Path(audit['checkpoint']))
        assert all(sha256(p)==digest for p,digest in audit['source_hashes'].items())
        del l;gc.collect();torch.cuda.empty_cache()
    assert sha256(args.original)==INIT_SHA and sha256(args.h)==H_SHA
    atomic_json(root/'GPU_WORK_FINISHED.json',dict(passed=True,R_complete=True,joint_complete=audit is not None,
        status='JOINT_GPU_COMPLETE' if audit else 'R complete; joint comparison awaiting L',**accounting(u),
        final_cursor=loader.state_dict(),resources=resources,evaluation_ledger=ledger,scientific_training_arms=1,
        final_sequence_condition_rows=20480 if audit else 16384,final_target_predictions=20971520 if audit else 16777216,
        monitor_target_predictions=6553600,initial_original_sha256=INIT_SHA,H_source_sha256=H_SHA,
        full_H_initial_identity=FULL_INIT_SHA,other_experiment_files_modified=False))
    progress(root,'FINISHED')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['run','binding','initial','original','h','data','validation','l-run']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',type=Path);args=p.parse_args()
    try:run(args)
    except BaseException as e:
        atomic_json(args.run/'FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
