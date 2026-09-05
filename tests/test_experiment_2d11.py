import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_2d11.common import *
from experiment_2d11.data import *
from experiment_2d11.model import *
from experiment_2d11.checkpoints import *
from experiment_2d11.evaluate import *
torch.set_num_threads(1)


def tiny():return FreshH(fresh_base(True))
def inputs():
    x=torch.arange(140).reshape(2,70)%32
    return x,(x+1)%32


def test_original_ast_fresh_init_and_isolated_router_rng():
    b=fresh_base(True);rng=torch.get_rng_state().clone();h=FreshH(b)
    assert torch.equal(rng,torch.get_rng_state())
    assert tensor_identity(h.base.named_parameters())==tensor_identity(fresh_base(True).named_parameters())
    assert h.base.lm_head.weight is h.base.transformer.wte.weight
    for i,r in h.routers.items():
        g=torch.Generator().manual_seed(20260916+int(i))
        assert torch.equal(r.W1,torch.randn(r.W1.shape,generator=g)*.02)
        assert r.W2.eq(0).all() and r.b1.eq(0).all() and r.b2.eq(0).all()
    opt=optimizer_for(h,'cpu')
    assert all(p.requires_grad for g in opt.param_groups for p in g['params'])
    assert not opt.state


@pytest.mark.parametrize('passes',[2,3])
def test_loss_checkpoint_recompute_and_attached_temporal_gradient(passes):
    h=tiny();other=copy.deepcopy(h);x,y=inputs()
    loss,ls=h(x,y,num_passes=passes,activation_checkpointing=True)
    direct=other.forward_multi_pass(x,targets=y,num_passes=passes,activation_checkpointing=False)
    weights=torch.tensor([.25,.75] if passes==2 else [.2,.4,.4])
    torch.testing.assert_close(loss,(weights*ls).sum())
    loss.backward();direct['loss'].backward()
    for (n,p),(nn,q) in zip(h.named_parameters(),other.named_parameters()):
        assert n==nn
        if not p.requires_grad:assert p.grad is q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    opt=optimizer_for(h,'cpu');opt.step();opt.zero_grad(set_to_none=True)
    h(x,y,num_passes=passes)[0].backward()
    for r in h.routers.values():
        assert r.W1.grad.abs().sum()>0 and r.b1.grad.abs().sum()>0
    for b in (0,2,4):
        source=torch.randn(2,70,16,requires_grad=True)
        residual=torch.randn(2,70,16,requires_grad=True)
        h._parallel_special_block(residual,b,source,None,None,False)[0].square().mean().backward()
        assert source.grad[:,:6].abs().sum()>0


@pytest.mark.parametrize('bf16',[False,True])
def test_router_simplex_availability_and_zero_valued_memory(bf16):
    h=tiny();dtype=torch.bfloat16 if bf16 else torch.float32
    q,l,r=[torch.randn(2,2,70,8,dtype=dtype) for _ in range(3)]
    residual=torch.randn(2,70,16)
    for b in (0,2,4):
        mask=h.recurrent_mask(b,70,70,'cpu');bank=SimpleNamespace(valid_mask=mask)
        combined,coefficient,diag=h.combine(residual,q,l,r,b,bank)
        available=mask.any(-1).view(1,1,70,1)
        torch.testing.assert_close(combined,torch.where(available,.5*l+.5*r,l),atol=0,rtol=0)
        assert torch.equal(diag['lambda_L']+diag['lambda_R'],torch.ones_like(diag['lambda_L']))
        assert torch.equal(h.combine(residual,q,l,r,b,None)[0],l)
        z=h.combine(residual,q,l,r*0,b,bank)[0]
        torch.testing.assert_close(z,torch.where(available,.5*l,l),atol=0,rtol=0)


@pytest.mark.parametrize('incremental',[False,True])
def test_causal_suffix_and_row_isolation_past_lag63(incremental):
    h=tiny().eval();x,y=inputs()
    with torch.no_grad():
        for r in h.routers.values():r.W2.normal_(0,.02)
    def f(x):
        if not incremental:return h.forward_multi_pass(x,num_passes=3)['logits']
        state=h.init_incremental_state(2);zs=[]
        for i in range(70):z,state=h.incremental_step(x[:,i],state);zs.append(z)
        return torch.cat(zs,1)
    suffix=x.clone();suffix[:,66:]=(suffix[:,66:]+7)%32
    other=x.clone();other[1]=(other[1]+9)%32
    with torch.no_grad():a,b,c=f(x),f(suffix),f(other)
    torch.testing.assert_close(a[:,:66],b[:,:66],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(a[0],c[0],atol=2e-6,rtol=2e-5)


def test_schedule_and_accounting_all_indices():
    assert len(CE_SCHEDULE)==24 and len(HELLA_SCHEDULE)==7
    original=original_symbols()['get_lr']
    assert all(learning_rate(i)==original(i) for i in range(19073))
    assert accounting(0)['mean_ce_passes_per_logical_target'] is None
    assert accounting(1000)['ce_pass_targets']==1064828928
    assert accounting(2000)['ce_pass_targets']==2129657856
    assert accounting(19072)['ce_pass_targets']==20310917120
    assert sum(pass_count(u)==3 for u in range(1,19073))==596
    with pytest.raises(ValueError):pass_count(19073)


def test_original_loader_every_transition_wrap_and_partition(tmp_path,monkeypatch):
    folder=tmp_path/'edu_fineweb10B';folder.mkdir()
    shards=[]
    for i,n in enumerate([53,64,85]):
        a=(np.arange(n)+i*100).astype(np.uint16);name=f'train_{i}.npy';np.save(folder/name,a)
        shards.append(dict(filename=name,tokens=n,dtype='uint16'))
    monkeypatch.chdir(tmp_path)
    original=original_symbols()['DataLoaderLite'](4,3,0,1,'train')
    loader=LogicalLoader(folder,shards,b=4,t=3)
    for _ in range(60):
        x,y=loader.next_batch();xx,yy=original.next_batch()
        assert np.array_equal(x,xx.numpy()) and np.array_equal(y,yy.numpy())
        assert loader.cursor.shard==original.current_shard and loader.cursor.position==original.current_position
    x=np.arange(512*3).reshape(512,3);y=x+1
    recovered={}
    for rank in range(4):
        for micro,(a,b) in enumerate(physical_batches(x,y,rank)):
            recovered[micro*4+rank]=(a,b)
    assert np.array_equal(np.concatenate([recovered[i][0] for i in range(16)]),x)
    assert np.array_equal(np.concatenate([recovered[i][1] for i in range(16)]),y)


def test_full_metadata_plan_against_separate_original_rule():
    shards=read_json(FROZEN/'shards.json')['train'];plan=full_plan(shards)
    shard=position=wraps=0
    for row in plan:
        for source,offset in row['spans']:
            assert (source,offset)==(shard,position)
            position+=65536
            if position+65537>shards[shard]['tokens']:
                shard=(shard+1)%len(shards);position=0;wraps+=int(shard==0)
        assert row['after']['position']==position and row['after']['shard']==shard
    assert wraps==1 and row['after']==read_json(FROZEN/'stream_summary.json')['last']['after']


def test_stop_boundaries_and_missing_inputs():
    ref=REFERENCES[1000]['ce']
    assert quality_decision(1000,ref+.5)['stop']
    assert not quality_decision(1000,ref+.5-1e-8)['stop']
    # Supply exactly representable gaps/advantages using direct boundary values away from cancellation.
    ref1=REFERENCES[2000]
    assert quality_decision(2000,ref1['ce']+.2,ref1['hella'],.2)['stop']
    assert not quality_decision(2000,ref1['ce']+.149,ref1['hella'],.2)['stop']
    assert not quality_decision(2000,ref1['ce']+.151,ref1['hella'],.21)['stop']
    assert not quality_decision(2000,ref1['ce']+.2,ref1['hella']+.005001,.2)['stop']
    with pytest.raises(ValueError):quality_decision(2000,4.)


def test_g_incremental_against_original_parallel():
    g=IncrementalG(fresh_base(True)).eval();x,y=inputs();state=g.init_incremental_state(2)
    with torch.no_grad():
        zs=[]
        for i in range(70):z,state=g.incremental_step(x[:,i],state);zs.append(z)
        z=torch.cat(zs,1);parallel=g.base(x)[0]
    torch.testing.assert_close(z,parallel,atol=2e-6,rtol=2e-5)
    assert all(c.length==69 for c in state.caches)


def hella_fixture():
    rows=[]
    for i in range(3):
        choices=[dict(tokens=([i+1,2,3]+[(i+c+4)%32]*(c+2)),prompt_length=3) for c in range(4)]
        rows.append(dict(id=i,original_id=i,label=i%4,choices=choices,length=8))
    return rows


@pytest.mark.parametrize('bf16',[False,True])
def test_hella_masks_first_completion_and_batched_reference(bf16):
    g=IncrementalG(fresh_base(True)).eval();rows=hella_fixture()
    scores=hella_batch(g,rows,'cpu',bf16)
    for i,row in enumerate(rows):
        tokens,mask=collate_hella([row],'cpu');state=g.init_incremental_state(4);zs=[]
        with torch.no_grad():
            for pos in range(tokens.size(1)):
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):z,state=g.incremental_step(tokens[:,pos],state)
                zs.append(z)
        z=torch.cat(zs,1)
        losses=F.cross_entropy(z[:,:-1].contiguous().view(-1,32),tokens[:,1:].contiguous().view(-1),reduction='none').reshape(4,-1)
        original=(losses*mask[:,1:]).sum(1)/mask[:,1:].sum(1)
        torch.testing.assert_close(scores[i],original,atol=0,rtol=0)
        assert mask[0,2]==0 and mask[0,3]==1
        assert scores[i].argmin().item()==original_symbols()['get_most_likely_row'](tokens,mask,z)


def test_partition_and_merge_coverage():
    rows=[dict(id=i,length=(i%7+1)*17) for i in range(10042)]
    plan=partition_examples(rows)
    assert sorted(x for owner in plan['owners'] for x in owner)==list(range(10042))
    parts=[dict(rows=[dict(id=i,correct=True) for i in owner],seconds=1.) for owner in plan['owners']]
    assert merge(parts,range(10042),'hella')['score']==1.
    parts[0]['rows'].append(parts[0]['rows'][0])
    with pytest.raises(AssertionError):merge(parts,range(10042),'hella')


def test_checkpoint_immutable_resume_rng_and_cadence(tmp_path):
    h=tiny();opt=optimizer_for(h,'cpu');x,y=inputs()
    # Tiny fixtures exercise the real optimizer counter and 31->32 cadence boundary.
    for u in range(1,32):
        opt.zero_grad(set_to_none=True);h(x,y,num_passes=pass_count(u))[0].backward();opt.step()
    loader=SimpleNamespace(state_dict=lambda:dict(shard=0,position=0,wraps=0,logical_batches=248))
    rng=capture_rng();payload=make_payload(h,opt,loader,31,{'ce:0':{'score':4.}},[rng],{}, {})
    before=payload['model']['routers.0.b2'].clone()
    with torch.no_grad():h.routers['0'].b2.add_(1.)
    assert torch.equal(payload['model']['routers.0.b2'],before)
    p=tmp_path/'u00031.pt';writer=CheckpointWriter();writer.submit(p,payload,tiny=True);m=writer.wait()
    assert sha256(p)==m['sha256'];assert m['audit']['passed']
    restored=torch.load(p,weights_only=False)
    a,b=tiny(),tiny();oa,ob=optimizer_for(a,'cpu'),optimizer_for(b,'cpu')
    for mod,o in [(a,oa),(b,ob)]:
        mod.load_state_dict(restored['model']);o.load_state_dict(copy.deepcopy(restored['optimizer']))
        restore_rng(restored['rng_by_rank'][0]);o.zero_grad(set_to_none=True)
        mod(x,y,num_passes=pass_count(32))[0].backward();o.step()
    assert tensor_identity(a.named_parameters())==tensor_identity(b.named_parameters())
    state=torch.get_rng_state().clone();template=uninitialized_h(True)
    assert torch.equal(state,torch.get_rng_state())


def test_frozen_initial_restores_exactly_and_platform_probe_is_disposable():
    initial=read_json(FROZEN/'initial_identity.json')
    probe=read_json(ROOT/'results/experiment_2d11/initial_probe/output/probe-rank0.json')
    # Seeded normal draws can differ between macOS ARM and Linux x86. Scientific ranks
    # must load the frozen tensors; regenerating from just the seed is forbidden.
    assert initial['base_identity']==initial['mapped_base_identity']
    assert probe['disposable'] and probe['completed_scientific_updates']==0
    panel=read_json(FROZEN/'fresh_panel.json');audit=read_json(FROZEN/'panel_disjointness.json')
    assert len(panel['sequences'])==4096 and panel['targets']==4194304 and audit['passed']
    assert all(any(term in p['path'] for p in audit['sources']) for term in ['experiment_2d9','experiment_2d10_h_250m','experiment_2d10_retrieval'])
