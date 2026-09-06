"""Read-only acquisition of atomically completed L and its machine-readable proof."""
import json
import torch
from .common import *
from .model import from_state,tensor_identity,optimizer_for,optimizer_names

def ready(source_run):
    source_run=Path(source_run);checkpoint=source_run/'checkpoints/u01000.pt'
    manifest=Path(str(checkpoint)+'.manifest.json');metrics=source_run/'metrics.jsonl'
    if not all(p.exists() for p in (checkpoint,manifest,metrics)):return None
    before={str(p):sha256(p) for p in (checkpoint,manifest,metrics)}
    v=read_json(manifest)
    assert before[str(checkpoint)]==v['sha256'] and v['audit']['passed'] and v['completed_updates']==1000
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert payload['schema']=='2d13-L-complete-v1' and payload['world_size']==1
    assert payload['completed_updates']==1000 and payload['accounting']['logical_targets']==524288000
    assert payload['next_lr_index']==1000 and payload['identities']['initial_sha']==INIT_SHA
    expected=[json.loads(z) for z in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    rows=[json.loads(z) for z in metrics.read_text().splitlines()]
    assert len(rows)==1000
    for u,(r,e) in enumerate(zip(rows,expected),1):
        assert r['completed_updates']==u and r['data']==e['data'] and r['lr']==e['lr']
        assert r['actual_ce_passes']==1 and r['logical_targets']==524288*u
    assert payload['loader']==expected[-1]['data']['after']
    model=from_state(payload['model']);model.requires_grad_(False);model.base.requires_grad_(True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==124475904
    frozen=tensor_identity((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    assert frozen==payload['frozen_identity']
    initial=source_run/'checkpoints/u00000.pt';iv=read_json(str(initial)+'.manifest.json')
    assert sha256(initial)==iv['sha256'] and iv['audit']['passed']
    ip=torch.load(initial,map_location='cpu',mmap=True,weights_only=False)
    im=from_state(ip['model']);assert tensor_identity(im.base.named_parameters())==BASE_SHA
    assert tensor_identity(im.named_parameters())==FULL_INIT_SHA
    for n,p in model.named_parameters():
        assert torch.isfinite(p).all()
        if not p.requires_grad:assert torch.equal(p,dict(im.named_parameters())[n])
    opt=optimizer_for(model,'cpu');assert optimizer_names(model,opt)==payload['optimizer_names']
    opt.load_state_dict(payload['optimizer'])
    for n,p in model.named_parameters():
        if not p.requires_grad:assert p not in opt.state
        else:
            s=opt.state[p];assert int(s['step'])==1000 and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
    assert all(sha256(p)==digest for p,digest in before.items()),'L source changed during acquisition'
    audit=dict(passed=True,checkpoint=str(checkpoint),sha256=v['sha256'],bytes=checkpoint.stat().st_size,
        model_tensor_identity=tensor_identity(model.named_parameters()),source_run=str(source_run),source_initial_sha256=iv['sha256'],
        source_backbone_identity=BASE_SHA,full_initial_identity=FULL_INIT_SHA,completed_updates=1000,logical_targets=524288000,
        terminal_cursor=payload['loader'],full_stream_verified=True,local_only_training=True,routers_frozen=True,
        source_hashes=before,source_code_identity=payload['identities']['code_identity'],completion_manifest=v,
        final_source_commit_or_tag='pending external archival; machine-readable completed checkpoint verified')
    return audit

def load(audit,device='cuda'):
    assert audit['passed'] and sha256(audit['checkpoint'])==audit['sha256']
    p=torch.load(audit['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    model=from_state(p['model'],device).requires_grad_(False).eval();model.set_condition('H_ALL_OFF')
    assert tensor_identity(model.named_parameters())==audit['model_tensor_identity']
    return model
