"""Explicit branch removal; original ON operations and writer bookkeeping retained."""
import torch
from torch.nn import functional as F
from experiment_2d11.model import FreshH, fresh_base, tensor_identity
from .common import CONDITIONS, WINDOWS

class AblatedH(FreshH):
    def __init__(self,base,condition='H_ON'):
        super().__init__(base)
        self.set_condition(condition)

    def set_condition(self,condition):
        if condition not in CONDITIONS:raise ValueError(condition)
        self.condition=condition
        self.disabled_destinations=frozenset(CONDITIONS[condition])

    def _incremental_bank_from_ring(self,ring,positions,state_position,minimum_lag,mode):
        # Parent step still maintains all three writer rings. Skip materializing a
        # recurrent read at disabled destinations; minimum lags are unique here.
        destination={1:0,31:2,63:4}[minimum_lag]
        if destination in self.disabled_destinations:return None
        return super()._incremental_bank_from_ring(ring,positions,state_position,minimum_lag,mode)

    def _incremental_special_block(self,residual,block_index,cache,recurrent_bank,
            permutation,gate_override,local_capacity,return_diagnostics,diagnostic_attention_weights):
        if block_index not in self.disabled_destinations:
            return super()._incremental_special_block(residual,block_index,cache,recurrent_bank,
                permutation,gate_override,local_capacity,return_diagnostics,diagnostic_attention_weights)
        assert local_capacity==WINDOWS[block_index]-1
        # The sealed ordinary block is exactly local attention, one c_proj,
        # attention residual, then LN/MLP and its residual. It never calls combine.
        output,next_cache=self._incremental_ordinary_block(residual,block_index,cache,local_capacity)
        diag={'effective_weights':[1.,0.],'router_invoked':False,'recurrent_read':False} if return_diagnostics else None
        return output,next_cache,diag

    def _parallel_special_block(self,residual,block_index,recurrent_source,
            recurrent_permutation,gate_override,return_diagnostics):
        if block_index not in self.disabled_destinations:
            return super()._parallel_special_block(residual,block_index,recurrent_source,
                recurrent_permutation,gate_override,return_diagnostics)
        block=self.base.transformer.h[block_index]
        batch,length,channels=residual.shape;heads=self.config.n_head
        q,k,v=block.attn.c_attn(block.ln_1(residual)).split(channels,dim=-1)
        q,k,v=[z.view(batch,length,heads,channels//heads).transpose(1,2) for z in (q,k,v)]
        local=F.scaled_dot_product_attention(q,k,v,attn_mask=self.local_mask(block_index,length,residual.device))
        mixture=local
        a=residual+block.attn.c_proj(mixture.transpose(1,2).contiguous().view(batch,length,channels))
        output=a+block.mlp(block.ln_2(a))
        return output,({'local':local,'mixture':mixture,'effective_weights':[1.,0.],
                        'router_invoked':False} if return_diagnostics else None)

@torch.inference_mode()
def load_model(path,condition='H_ON',device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert payload['completed_updates']==19072
    assert payload['accounting']['logical_targets']==9999220736
    model=AblatedH(fresh_base(),condition)
    model.load_state_dict(payload['model'],strict=True)
    assert model.base.lm_head.weight is model.base.transformer.wte.weight
    assert sum(p.numel() for p in model.parameters())==124697386
    model.requires_grad_(False);model.eval()
    return model.to(device)
