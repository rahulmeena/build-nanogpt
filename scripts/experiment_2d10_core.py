"""Retrieval-aware routers over unchanged O1 local/recurrent attention outputs.

The two special-block methods retain the sealed 2D3a attention/cache operations;
only branch combination changes. Inputs are explicit current-invocation tensors.
"""
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
import experiment_2d7_core as prior

BLOCKS = (0, 2, 4)
PARAMETER_COUNTS = {"T": 124699588, "H": 124697386}
CONDITIONS = {"T": "T_REAL", "H": "H_REAL"}
W_NAMES = {0: "w_B1", 2: "w_B3", 4: "w_B5"}

def architecture_manifest(arm):
    value = prior.architecture_manifest("O")
    value.update(experiment="2D10", arm=arm, condition=CONDITIONS[arm],
        parameter_count=PARAMETER_COUNTS[arm], new_parameters=PARAMETER_COUNTS[arm]-124475908,
        router_inputs=["q_projected", "local_output", "recurrent_output"],
        normalization="FP32 affine-free LayerNorm per vector, population variance, epsilon 1e-5",
        hidden_width=32, activation="SiLU", final_coefficient_cast="attention dtype",
        formula="A_L+tanh(g0+w^T RMS(h)+W2 SiLU(W1 phi+b1)) A_R" if arm=="T" else "softmax(W2 SiLU(W1 phi+b1)+b2) dot [A_L,A_R]",
        initialization_seed="20260908 + zero-based destination block",
        initialization="W1 Normal(0,.02); b1/W2 zero; T w zero; H b2=[0,log(tanh(parent g0))]",
        compatibility_states=["g_rec_b6"] if arm=="T" else ["g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"],
        empty_memory="exact local-only", persistent_router_state=False)
    return value

def architecture_fingerprint(arm):
    return hashlib.sha256(json.dumps(architecture_manifest(arm),sort_keys=True,separators=(",",":")).encode()).hexdigest()

class Router(nn.Module):
    def __init__(self, width, block, arm, g0):
        super().__init__()
        generator=torch.Generator(device="cpu").manual_seed(20260908+block)
        self.W1=nn.Parameter(torch.randn(32,3*width,generator=generator)*.02)
        self.b1=nn.Parameter(torch.zeros(32))
        self.W2=nn.Parameter(torch.zeros(1 if arm=="T" else 2,32))
        if arm=="H":
            c=g0.detach().cpu().float().tanh()
            if not bool(torch.isfinite(c) & (c>0)):
                raise ValueError("H requires a strictly positive finite source coefficient")
            self.b2=nn.Parameter(torch.stack((torch.zeros_like(c),c.log())))

class RetrievalGatingGPT(prior.BoundaryAlignmentGPT):
    def __init__(self, base):
        super().__init__(base,"O")
        self.arm=None
        self.gate_collector=None

    def enable_router(self, arm):
        if self.arm is not None or arm not in PARAMETER_COUNTS:
            raise ValueError("invalid/repeated router initialization")
        self.arm=arm
        self.routers=nn.ModuleDict({str(b):Router(self.config.n_embd,b,arm,self.gate_parameter(b)) for b in BLOCKS})
        self.routers.to(device=self.g_rec.device,dtype=torch.float32)
        if arm=="T":
            for name in W_NAMES.values():
                self.register_parameter(name,nn.Parameter(torch.zeros(self.config.n_embd,device=self.g_rec.device)))

    def architecture_fingerprint(self):
        return architecture_fingerprint(self.arm)

    def combine(self, residual, query, local, recurrent, block, bank):
        batch,_,length,_=query.shape
        def flatten(x):
            return x.transpose(1,2).contiguous().view(batch,length,-1).float()
        with torch.autocast(device_type=residual.device.type,enabled=False):
            phi=torch.cat([F.layer_norm(flatten(x),(self.config.n_embd,),eps=1e-5) for x in (query,local,recurrent)],dim=-1)
            router=self.routers[str(block)]
            hidden=F.silu(F.linear(phi,router.W1,router.b1))
            if bank is None:
                available=torch.zeros((1,length,1),device=query.device,dtype=torch.bool)
            else:
                available=bank.valid_mask.any(-1).reshape(1,length,1)
            if self.arm=="T":
                h=residual.float()
                u=self.gate_parameter(block).float()+(h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-5)*getattr(self,W_NAMES[block])).sum(-1,keepdim=True)
                delta=F.linear(hidden,router.W2)
                g=torch.tanh(u+delta)
                coefficient=g.to(local.dtype).unsqueeze(1)
                combined=local+coefficient*recurrent
                diagnostics={"u":u,"delta":delta,"g":g,"cast_g":coefficient.squeeze(1).float()}
            else:
                z=F.linear(hidden,router.W2,router.b2)
                weights=z.softmax(-1)
                forced=torch.cat((torch.ones_like(weights[...,:1]),torch.zeros_like(weights[...,:1])),dim=-1)
                weights=torch.where(available,weights,forced)
                cast=weights.to(local.dtype)
                combined=cast[...,:1].unsqueeze(1)*local+cast[...,1:].unsqueeze(1)*recurrent
                coefficient=cast[...,1:].unsqueeze(1)
                diagnostics={"logit_difference":z[...,1:]-z[...,:1],"lambda_L":weights[...,:1],"lambda_R":weights[...,1:],"cast_lambda_L":cast[...,:1].float(),"cast_lambda_R":cast[...,1:].float(),"entropy":-(weights*weights.clamp_min(1e-38).log()).sum(-1,keepdim=True)}
            diagnostics["available"]=available.expand(batch,-1,-1)
        if self.gate_collector is not None:
            self.gate_collector.record(block,diagnostics)
        return combined,coefficient,diagnostics

    def _parallel_special_block(
        self,
        residual,
        block_index,
        recurrent_source,
        recurrent_permutation,
        gate_override,
        return_diagnostics,
    ):
        block = self.base.transformer.h[block_index]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        normalized = block.ln_1(residual)
        query, local_key, local_value = block.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.local_mask(block_index, length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query, local_key, local_value, attn_mask=local_mask, is_causal=False
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )

        bank = None
        recurrent_weights = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
        else:
            if recurrent_permutation is not None:
                recurrent_permutation = self._validate_permutation(
                    recurrent_permutation, batch, residual.device
                )
                recurrent_source = recurrent_source[recurrent_permutation]
            bank = self.build_recurrent_bank(block_index, recurrent_source)
            rec_key, rec_value = self.project_recurrent_kv(block_index, bank.values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                rec_key,
                rec_value,
                bank.valid_mask,
                return_weights=return_diagnostics,
            )
        combined, coefficient, routing = self.combine(
            residual, query, local_pre, recurrent_pre, block_index, bank)
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        after_attention = residual + block.attn.c_proj(combined)
        output = after_attention + block.mlp(block.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        return output, {
            "block": block_index + 1,
            "source_block": {0: 12, 2: 10, 4: 8, 5: 7}[block_index],
            "local_attention_weights": local_weights,
            "recurrent_attention_weights": recurrent_weights,
            "local_valid_mask": local_mask,
            "recurrent_valid_mask": None if bank is None else bank.valid_mask,
            "recurrent_positions": None if bank is None else bank.positions,
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
            "gate_raw": self.gate_parameter(block_index),
            "gate_coefficient": coefficient,
                "routing": routing,
        }

    def _incremental_special_block(
        self,
        residual,
        block_index,
        cache,
        recurrent_bank,
        permutation,
        gate_override,
        local_capacity,
        return_diagnostics,
        diagnostic_attention_weights,
    ):
        block = self.base.transformer.h[block_index]
        batch = residual.size(0)
        channels = int(self.config.n_embd)
        heads = int(self.config.n_head)
        head_size = channels // heads
        normalized = block.ln_1(residual)
        query, current_key, current_value = block.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, 1, heads, head_size).transpose(1, 2)
        current_key = current_key.view(batch, 1, heads, head_size).transpose(1, 2)
        current_value = current_value.view(batch, 1, heads, head_size).transpose(1, 2)
        local_keys = current_key if cache is None else torch.cat((cache.key, current_key), dim=2)
        local_values = current_value if cache is None else torch.cat((cache.value, current_value), dim=2)
        if local_keys.size(2) > local_capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess local KV")
        local_pre = F.scaled_dot_product_attention(
            query, local_keys, local_values, is_causal=False
        )
        next_cache = self._append_cache(current_key, current_value, cache, local_capacity)

        recurrent_weights = None
        if recurrent_bank is None:
            recurrent_pre = torch.zeros_like(local_pre)
        elif recurrent_bank.values.size(1) == 0:
            recurrent_pre = torch.zeros_like(local_pre)
            recurrent_weights = local_pre.new_empty((batch, heads, 1, 0))
        else:
            bank_values = recurrent_bank.values
            if permutation is not None:
                bank_values = bank_values[permutation]
            rec_key, rec_value = self.project_recurrent_kv(block_index, bank_values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                rec_key,
                rec_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics and diagnostic_attention_weights,
            )
        combined, coefficient, routing = self.combine(
            residual, query, local_pre, recurrent_pre, block_index, recurrent_bank)
        combined = combined.transpose(1, 2).contiguous().view(batch, 1, channels)
        residual = residual + block.attn.c_proj(combined)
        residual = residual + block.mlp(block.ln_2(residual))
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "recurrent_attention_weights": recurrent_weights,
                "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
                "gate_coefficient": coefficient,
                "routing": routing,
                "recurrent_positions": (
                    None if recurrent_bank is None else recurrent_bank.positions
                ),
            }
        return residual, next_cache, diagnostics

