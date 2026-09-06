"""Single-GPU, inference-only queue; no bootstrap/report/Git operations."""
import argparse,os,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .model import load_model,tensor_identity
from .verification import focused
from .artifacts import existing,validate_batch,collect
from experiment_2d11.data import array_hash
from experiment_2d11.common import configure_cuda_determinism

def forbid(*args,**kwargs):raise RuntimeError('2D12 forbids optimizer construction and backward')

def progress(root,stage,**data):atomic_json(root/'PROGRESS.json',dict(time=time.time(),stage=stage,**data))

@torch.inference_mode()
def ce(model,x,y,root,stage,**meta):
    state=model.init_incremental_state(len(x),device=x.device,dtype=torch.bfloat16)
    sums=torch.zeros(len(x),device=x.device,dtype=torch.float64)
    for t in range(x.shape[1]):
        with torch.autocast('cuda',dtype=torch.bfloat16):z,state=model.incremental_step(x[:,t],state)
        sums+=F.cross_entropy(z[:,0].float(),y[:,t],reduction='none').double()
        if t%64==0:progress(root,stage,position=t,**meta)
    return sums,state

def preflight(model,root,binding):
    started=time.time();signal.alarm(600)
    progress(root,'CUDA_PREFLIGHT',part='focused checks')
    audit=focused(model,device='cuda',bf16=True)
    model.set_condition('H_ON')
    perf=[]
    for batch in [128,64,32]:
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            x=((torch.arange(batch*1024,device='cuda').reshape(batch,1024)*17+3)%50304).long();y=(x+1)%50304
            t=time.time();nll,state=ce(model,x,y,root,'CUDA_PREFLIGHT',part='full length speed and memory',batch=batch)
            torch.cuda.synchronize();elapsed=time.time()-t
            cache=model.incremental_cache_audit(state);assert cache['passed']
            assert cache['actual_unique_storage_bytes']//batch==33289728
            assert torch.isfinite(nll).all()
            perf.append(dict(batch_size=batch,seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),cache_audit=cache))
            del x,y,nll,state;break
        except torch.cuda.OutOfMemoryError:
            perf.append(dict(batch_size=batch,oom=True));torch.cuda.empty_cache()
    else:raise RuntimeError('No permitted physical batch fits')
    projected=elapsed*(4096/batch)*5
    assert time.time()+projected+600<binding['hard_deadline'],'full-H-equivalent projection exceeds remaining budget'
    audit.update(performance=perf,batch_size=batch,seconds=time.time()-started,projected_scoring_seconds=projected,
        device_name=torch.cuda.get_device_name(),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu_count=torch.cuda.device_count(),scientific_panel_used=False,
        cuda_tolerance_basis='Direct ON and independent fully local incremental references exact; block reference atol .15/rtol .02; BF16 parallel-vs-incremental logit atol 1.0 and per-sequence CE atol .05 frozen after disposable diagnosis, before scientific scoring. Trained CPU FP32 parallel max logit error 2.575e-5 and max mean CE error 5.041e-7; shape-dependent BF16 rounding, not intervention semantics. Actual maxima recorded.')
    atomic_json(root/'PREFLIGHT_AUDIT.json',audit)
    signal.alarm(0)
    return batch

def run(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    configure_cuda_determinism();torch.set_float32_matmul_precision('high');torch.set_num_threads(4)
    torch.optim.Optimizer.__init__=forbid;torch.Tensor.backward=forbid;torch.autograd.backward=forbid
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    binding=read_json(args.binding)
    assert sha256(args.checkpoint)==CHECKPOINT_SHA and sha256(args.validation)==VAL_SHA
    sources=read_json(FROZEN/'SOURCE_IDENTITIES.json');panel=read_json(FROZEN/'PANEL.json')
    for name,digest in read_json(FROZEN/'CODE_IDENTITY.json')['files'].items():assert sha256(REPO/name)==digest
    model=load_model(args.checkpoint,device='cuda')
    original=tensor_identity(model.named_parameters());assert original==sources['model_tensor_identity']
    if (root/'RUN_CONFIG.json').exists():
        config=read_json(root/'RUN_CONFIG.json');batch=config['batch_size']
        assert read_json(root/'PREFLIGHT_AUDIT.json')['passed']
        frozen_code=read_json(FROZEN/'CODE_IDENTITY.json')
        assert config['binding']['code_commit']==frozen_code['commit']
        assert config['binding']['code_files_identity']==identity(frozen_code['files'])
        assert config['binding']['interventions_sha256']==sha256(FROZEN/'INTERVENTIONS.json')
        assert config['binding']['panel_identity']==panel['identity']
        assert config['binding']['checkpoint_sha256']==CHECKPOINT_SHA
    else:
        batch=preflight(model,root,binding)
        code=read_json(FROZEN/'CODE_IDENTITY.json')
        scientific_binding=dict(panel_identity=panel['identity'],checkpoint_sha256=CHECKPOINT_SHA,
            code_commit=code['commit'],code_files_identity=identity(code['files']),
            interventions_sha256=sha256(FROZEN/'INTERVENTIONS.json'),batch_size=batch,
            mode='eval inference_mode BF16 CUDA, FP32 token CE, FP64 NLL sum',
            matmul_precision='high',deterministic_algorithms=True,cublas_workspace_config=':4096:8')
        config=dict(batch_size=batch,binding=scientific_binding,frozen_at=time.time(),
            conditions=list(CONDITIONS),training_updates=0,preflight_sha256=sha256(root/'PREFLIGHT_AUDIT.json'))
        atomic_json(root/'RUN_CONFIG.json',config)
    arr=np.load(args.validation,mmap_mode='r',allow_pickle=False)
    for condition in CONDITIONS:
        model.set_condition(condition);started=time.time();skipped=0;done=0
        for start in range(0,4096,batch):
            selected=panel['sequences'][start:start+batch];path=root/condition/f'batch_{start:04d}.json'
            if existing(path,condition,selected,config['binding']) is not None:skipped+=1;continue
            assert time.time()<binding['hard_deadline']-600,'scientific cutoff reached'
            x=np.stack([arr[r['input_start']:r['input_start']+1024] for r in selected]).astype(np.int64)
            y=np.stack([arr[r['input_start']+1:r['input_start']+1025] for r in selected]).astype(np.int64)
            for r,xx,yy in zip(selected,x,y):assert array_hash(xx,yy)==r['token_sha256']
            t=time.time();sums,state=ce(model,torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),root,
                'SCORING',condition=condition,batch_start=start)
            sums=sums.cpu().tolist();cache=model.incremental_cache_audit(state)
            assert cache['passed'] and cache['actual_unique_storage_bytes']//batch==33289728
            value=dict(condition=condition,binding=config['binding'],disabled_destinations=list(CONDITIONS[condition]),
                started_at=t,finished_at=time.time(),seconds=time.time()-t,cache_audit=cache,
                rows=[dict(id=r['id'],input_start=r['input_start'],token_sha256=r['token_sha256'],nll=nll,count=1024,ce=nll/1024) for r,nll in zip(selected,sums)])
            validate_batch(value,condition,selected,config['binding']);atomic_json(path,value);done+=1
            progress(root,'BATCH_COMPLETE',condition=condition,batch_start=start,completed=start+batch)
            del state
        after=tensor_identity(model.named_parameters());assert after==original
        atomic_json(root/(condition+'_COMPLETE.json'),dict(condition=condition,completed_at=time.time(),
            seconds=time.time()-started,new_batches=done,resumed_batches=skipped,tensor_identity_before=original,
            tensor_identity_after=after,training_updates=0))
    collect(root,panel)
    atomic_json(root/'GPU_WORK_FINISHED.json',dict(passed=True,time=time.time(),conditions=list(CONDITIONS),
        sequence_condition_results=20480,target_predictions=20971520,training_updates=0,
        tensor_identity_before=original,tensor_identity_after=tensor_identity(model.named_parameters())))
    progress(root,'FINISHED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        atomic_json(a.output/'FAILURE.json',dict(time=time.time(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise
