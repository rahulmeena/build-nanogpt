"""Independent local-only references and focused inference-only fixtures."""
import copy,math,time
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from .common import *
from .model import AblatedH,FreshH,fresh_base,tensor_identity

@torch.inference_mode()
def reference_block(model,r,b,history=None,manual=True):
    block=model.base.transformer.h[b];batch,length,width=r.shape;heads=model.config.n_head
    q,k,v=block.attn.c_attn(block.ln_1(r)).chunk(3,-1)
    q,k,v=[a.reshape(batch,length,heads,width//heads).transpose(1,2) for a in (q,k,v)]
    if history is None:
        t=torch.arange(length,device=r.device);lag=t[:,None]-t[None,:]
        mask=(lag>=0)&(lag<WINDOWS.get(b,model.config.block_size))
    else:
        k=torch.cat([history.key,k],2);v=torch.cat([history.value,v],2);mask=None
    if manual:
        scores=q@k.transpose(-1,-2)/math.sqrt(width//heads)
        if mask is not None:scores=scores.masked_fill(~mask,float('-inf'))
        local=scores.softmax(-1)@v
    else:local=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    a=r+F.linear(local.transpose(1,2).reshape(batch,length,width),block.attn.c_proj.weight,block.attn.c_proj.bias)
    return a+block.mlp(block.ln_2(a))

@torch.inference_mode()
def local_backbone(model,x,manual=True):
    r=model.base.transformer.wte(x)+model.base.transformer.wpe(torch.arange(x.shape[1],device=x.device))
    for b in range(12):r=reference_block(model,r,b,manual=manual)
    return model.base.lm_head(model.base.transformer.ln_f(r))

@torch.inference_mode()
def logits(model,x,bf16=False):
    state=model.init_incremental_state(len(x),device=x.device,dtype=torch.bfloat16 if bf16 else torch.float32)
    zs=[]
    for t in range(x.shape[1]):
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=bf16):z,state=model.incremental_step(x[:,t],state)
        zs.append(z)
    return torch.cat(zs,1),state

@torch.inference_mode()
def focused(model,device='cpu',bf16=False,full_rings=False):
    start=time.time();device=torch.device(device);width=model.config.n_embd;vocab=model.config.vocab_size
    before=tensor_identity(model.named_parameters());maxima={};checks=[]
    atol,rtol=(.15,.02) if bf16 else (2e-6,2e-5)
    x=(torch.arange(2*70,device=device).reshape(2,70)*17+3)%vocab
    model.set_condition('H_ON')
    sealed=FreshH(copy.deepcopy(model.base));sealed.load_state_dict(model.state_dict(),strict=True);sealed=sealed.to(device).eval().requires_grad_(False)
    actual,_=logits(model,x,bf16);expected,_=logits(sealed,x,bf16)
    assert torch.equal(actual,expected)
    y=(x+1)%vocab
    ca=F.cross_entropy(actual.float().reshape(-1,vocab),y.reshape(-1),reduction='none').reshape(2,70).double().sum(1)
    cb=F.cross_entropy(expected.float().reshape(-1,vocab),y.reshape(-1),reduction='none').reshape(2,70).double().sum(1)
    assert torch.equal(ca,cb);maxima['on_sealed_logits']=0.;maxima['on_sealed_nll']=0.
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
        pa=model.forward_multi_pass(x,num_passes=3,activation_checkpointing=False)['logits']
        pb=sealed.forward_multi_pass(x,num_passes=3,activation_checkpointing=False)['logits']
    assert torch.equal(pa,pb);maxima['on_sealed_parallel_logits']=0.;del sealed
    # Independently written references, bank independence, router bypass, one projection.
    for b in WINDOWS:
        model.set_condition({0:'H_B1_OFF',2:'H_B3_OFF',4:'H_B5_OFF'}[b])
        gen=torch.Generator(device=device).manual_seed(842+b)
        r=torch.randn(2,70,width,device=device,generator=gen)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
            with patch.object(model,'combine',side_effect=AssertionError('disabled router called')),patch.object(model,'project_recurrent_kv',side_effect=AssertionError('disabled read called')):
                a,diag=model._parallel_special_block(r,b,None,None,None,True)
                aa,_=model._parallel_special_block(r,b,torch.randn_like(r)*100,None,None,False)
                assert torch.equal(a,aa) and torch.equal(diag['mixture'],diag['local'])
                ref=reference_block(model,r,b,manual=not bf16)
                torch.testing.assert_close(a,ref,atol=atol,rtol=rtol)
                maxima[f'B{b+1}_parallel_reference']=float((a-ref).abs().max())
                # Nonempty native local cache, empty/nonempty and invalid recurrent banks.
                n=WINDOWS[b]-1;h=model.config.n_head;d=width//h
                dtype=torch.bfloat16 if bf16 else torch.float32
                cache=SimpleNamespace(key=torch.randn(2,h,n,d,device=device,dtype=dtype),value=torch.randn(2,h,n,d,device=device,dtype=dtype))
                cache.length=n
                values=[]
                for bank in [None,SimpleNamespace(values=r[:,:0],valid_mask=torch.zeros(1,0,device=device,dtype=torch.bool)),
                        SimpleNamespace(values=r,valid_mask=torch.ones(1,70,device=device,dtype=torch.bool)),
                        SimpleNamespace(values=r*100,valid_mask=torch.zeros(1,70,device=device,dtype=torch.bool))]:
                    z,_,_=model._incremental_special_block(r[:,:1],b,cache,bank,None,None,n,False,False);values.append(z)
                assert all(torch.equal(z,values[0]) for z in values)
                ref=reference_block(model,r[:,:1],b,cache,manual=not bf16)
                torch.testing.assert_close(values[0],ref,atol=atol,rtol=rtol)
                maxima[f'B{b+1}_incremental_reference']=float((values[0]-ref).abs().max())
        checks.append(f'B{b+1}: exact local mixture, one projection, bank/mask independence and router/read traps passed')
    for condition in CONDITIONS:
        model.set_condition(condition)
        a,state=logits(model,x,bf16)
        changed=x.clone();changed[:,66:]=(changed[:,66:]+7)%vocab
        b,_=logits(model,changed,bf16);assert torch.equal(a[:,:66],b[:,:66])
        changed=x.clone();changed[1]=(changed[1]+11)%vocab
        b,_=logits(model,changed,bf16);assert torch.equal(a[0],b[0])
        again,_=logits(model,x,bf16);assert torch.equal(a,again)
        if condition=='H_ALL_OFF':
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):ref=local_backbone(model,x,manual=not bf16)
            torch.testing.assert_close(a,ref,atol=atol,rtol=rtol)
            maxima['all_off_incremental_parallel']=float((a-ref).abs().max())
        if full_rings:
            long=(torch.arange(1024,device=device).reshape(1,-1)*19+5)%vocab
            # Capture writers at final step and prove they belong to this trajectory.
            state=model.init_incremental_state(1,device=device,dtype=torch.float32)
            for t in range(1024):
                z,state,d=model.incremental_step(long[:,t],state,return_block_states=True)
            for writer,block in [('h8',7),('h10',9),('h12',11)]:
                assert torch.equal(getattr(state,writer+'_ring')[:,-1:],d['block_states'][block])
                assert getattr(state,writer+'_positions')==tuple(range(1,1024))
            audit=model.incremental_cache_audit(state);assert audit['passed']
            assert audit['cache_lengths']==[1,1023,31,1023,63]+[1023]*7
            checks.append(dict(condition=condition,full_ring_audit=audit))
        checks.append(condition+': suffix causality, row isolation, fresh-state reset passed')
    # Masks include boundary exactly, reject future and lag1024.
    for b,lag in [(0,1),(2,31),(4,63)]:
        m=model.recurrent_mask(b,1,1025,device,query_offset=1024)
        assert m[0,1024-lag] and not m[0,1024-lag+1] and m[0,1] and not m[0,0] and not m[0,1024]
        l=model.local_mask(b,70,device);assert l[-1].sum()==WINDOWS[b]
    after=tensor_identity(model.named_parameters());assert before==after
    return dict(passed=True,device=str(device),bf16=bf16,atol=atol,rtol=rtol,maxima=maxima,checks=checks,
        before_tensor_sha256=before,after_tensor_sha256=after,seconds=time.time()-start,scientific_updates=0)


def cpu():
    torch.set_num_threads(1);base=fresh_base(True)
    # Expand only the positional fixture so all 1024 real ring transitions execute.
    base.config.block_size=1024;base.transformer.wpe=torch.nn.Embedding(1024,16)
    model=AblatedH(base).eval().requires_grad_(False)
    with torch.no_grad():
        for r in model.routers.values():r.W2.normal_(0,.1);r.b2.normal_(0,.1)
        for b in WINDOWS:model.base.transformer.h[b].attn.c_proj.bias.normal_(0,.1)
    audit=focused(model,full_rings=True)
    atomic_json(PACKAGE/'results/CPU_MODEL_AUDIT.json',audit)
    print({k:v for k,v in audit.items() if k!='checks'})

if __name__=='__main__':cpu()
