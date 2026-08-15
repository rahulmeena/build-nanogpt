import os
import math
import time
import inspect
import json
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, self_only=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        if self_only:
            # A diagonal-only attention row has one allowed key, so its softmax
            # is exactly one and the result is the current position's value.
            # Slice the trained fused QKV projection so no historical K/V—or
            # unnecessary Q/K activation—is materialized in this mode.
            y = F.linear(
                x,
                self.c_attn.weight[2 * self.n_embd:],
                self.c_attn.bias[2 * self.n_embd:],
            )
        else:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    def forward_step(self, x, cache=None, self_only=False):
        """Attend one token with an explicit cache and return its updated cache."""
        B, T, C = x.size()
        if T != 1:
            raise ValueError("incremental attention requires exactly one token")
        if self_only:
            if cache is not None:
                raise ValueError("self-only Block 1 must not have a historical KV cache")
            y = F.linear(
                x,
                self.c_attn.weight[2 * self.n_embd:],
                self.c_attn.bias[2 * self.n_embd:],
            )
            return self.c_proj(y), None
        if not isinstance(cache, AttentionKVCache):
            raise ValueError("incremental historical attention requires an explicit KV cache")
        if cache.length >= cache.key.size(2):
            raise ValueError("incremental KV cache capacity exceeded")
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        q = q.view(B, 1, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, 1, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, 1, self.n_head, head_size).transpose(1, 2)
        expected = (B, self.n_head, cache.key.size(2), head_size)
        if tuple(cache.key.shape) != expected or tuple(cache.value.shape) != expected:
            raise ValueError("incremental KV cache shape mismatch")
        if cache.key.device != x.device or cache.value.device != x.device:
            raise ValueError("incremental KV cache device mismatch")
        cache.key[:, :, cache.length : cache.length + 1].copy_(k.detach())
        cache.value[:, :, cache.length : cache.length + 1].copy_(v.detach())
        new_length = cache.length + 1
        keys = cache.key[:, :, :new_length]
        values = cache.value[:, :, :new_length]
        # Detached historical caches are mutated as later tokens arrive. During
        # reader-only training, preserve the exact prefix consumed by this
        # token so several independent token graphs can be backwarded together
        # without an in-place version change. Inference keeps the zero-copy
        # Experiment-2B0 path.
        if torch.is_grad_enabled() and q.requires_grad:
            keys = keys.clone()
            values = values.clone()
        y = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, 1, C)
        updated = AttentionKVCache(cache.key, cache.value, new_length)
        return self.c_proj(y), updated

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class RMSNorm(nn.Module):

    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)

class FullAttnRes(nn.Module):
    """Paper-exact, single-query softmax attention over residual depth."""

    def __init__(self, n_embd, eps=1e-5, destination=""):
        super().__init__()
        # Attention Residuals, Section 5: all pseudo-queries must start at zero.
        self.query = nn.Parameter(torch.zeros(n_embd))
        self.norm = RMSNorm(n_embd, eps=eps)
        self.destination = destination
        self.instrumentation_enabled = False
        self.last_stats = None
        self.masked_source = None

    def _score(self, value):
        key = self.norm(value)
        # Equation (2) has no 1/sqrt(d) factor. F.linear avoids materializing
        # a B x T x C float32 product while autocast uses the training dtype.
        return F.linear(key, self.query.unsqueeze(0)).squeeze(-1)

    def forward(self, values, return_weights=False):
        if not values:
            raise ValueError("FullAttnRes requires at least one residual value")

        score_rows = []
        use_checkpoint = self.training and torch.is_grad_enabled()
        for value in values:
            if use_checkpoint:
                score = checkpoint(
                    self._score,
                    value,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                score = self._score(value)
            score_rows.append(score.float())
        logits = torch.stack(score_rows, dim=0)  # depth x B x T

        if self.masked_source is not None and self.masked_source < len(values):
            if len(values) == 1:
                raise ValueError("cannot mask the only FullAttnRes source")
            logits = logits.clone()
            logits[self.masked_source] = -torch.inf

        weights = F.softmax(logits, dim=0)
        output = torch.zeros_like(values[0])
        for source_weight, value in zip(weights.unbind(dim=0), values):
            # The GPT-2 residual stream is FP32 under BF16 autocast. Preserve
            # each source dtype for the multiply, then accumulate into FP32.
            contribution = source_weight.to(value.dtype).unsqueeze(-1) * value
            output = output + contribution.to(output.dtype)

        if self.instrumentation_enabled:
            with torch.no_grad():
                safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
                entropy = -(weights * safe_weights.log()).sum(dim=0)
                self.last_stats = {
                    "destination": self.destination,
                    "source_depths": list(range(len(values))),
                    "mean_weights": weights.mean(dim=(1, 2)).detach().cpu().tolist(),
                    "mean_entropy": entropy.mean().detach().cpu().item(),
                }

        if return_weights:
            return output, weights
        return output

class TopDownAttnRes(nn.Module):
    """Independent depth router for detached previous-token teacher states."""

    def __init__(self, n_embd, source_depths, eps=1e-5):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_embd))
        self.norm = RMSNorm(n_embd, eps=eps)
        self.gate = nn.Parameter(torch.zeros(()))
        self.source_depths = tuple(source_depths)
        self.instrumentation_enabled = False
        self.last_stats = None
        self.masked_source = None

    def _score(self, value):
        key = self.norm(value)
        return F.linear(key, self.query.unsqueeze(0)).squeeze(-1)

    def forward(self, values, return_weights=False):
        if len(values) != len(self.source_depths):
            raise ValueError(
                f"expected {len(self.source_depths)} top-down sources, got {len(values)}"
            )
        score_rows = [self._score(value) for value in values]
        logits = torch.stack([score.float() for score in score_rows], dim=0)
        if self.masked_source is not None:
            logits = logits.clone()
            logits[self.masked_source] = -torch.inf
        weights = F.softmax(logits, dim=0)
        output = torch.zeros_like(values[0])
        for source_weight, value in zip(weights.unbind(dim=0), values):
            contribution = source_weight.to(value.dtype).unsqueeze(-1) * value
            output = output + contribution.to(output.dtype)

        if self.instrumentation_enabled:
            with torch.no_grad():
                safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
                entropy = -(weights * safe_weights.log()).sum(dim=0)
                self.last_stats = {
                    "source_depths": list(self.source_depths),
                    "mean_weights": weights.mean(dim=(1, 2)).detach().cpu().tolist(),
                    "mean_entropy": entropy.mean().detach().cpu().item(),
                    "query_norm": self.query.detach().float().norm().cpu().item(),
                    "gate": self.gate.detach().float().cpu().item(),
                    "gate_coefficient": self.gate.detach().float().tanh().cpu().item(),
                }

        if return_weights:
            return output, weights
        return output


class MemoryWriterAdapter(nn.Module):
    """Rank-limited residual writer used only for recurrent high-to-low memory."""

    def __init__(self, n_embd, rank=8, eps=1e-5):
        super().__init__()
        self.W_down = nn.Linear(n_embd, rank, bias=False)
        self.W_up = nn.Linear(rank, n_embd, bias=False)
        self.eps = float(eps)

    def forward(self, source):
        # The detach is deliberately inside the adapter so no caller can
        # accidentally grant temporal credit to the ordinary Transformer path.
        source = source.detach()
        normalized = source * torch.rsqrt(
            source.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        ).to(source.dtype)
        delta = self.W_up(F.silu(self.W_down(normalized)))
        return source + delta, delta


EXPERIMENT_2A0_SOURCE_DEPTHS = (16, 17, 20, 24)
EXPERIMENT_2A0_MODES = {
    "full_context",
    "masked_l1_no_feedback",
    "masked_l1_topdown_teacher",
    "masked_l1_shuffled_feedback",
}

EXPERIMENT_2B0_INCREMENTAL_MODES = {
    "full_context",
    "masked_l1_no_feedback",
    "masked_l1_topdown_teacher",
    "masked_l1_topdown_self",
    "masked_l1_shuffled_self_feedback",
}


@dataclass
class AttentionKVCache:
    """Explicit preallocated incremental-attention cache for one block."""

    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def prefix(self):
        return self.key[:, :, : self.length], self.value[:, :, : self.length]


@dataclass
class RecurrentState:
    """Serializable Experiment-2B0 state; modules retain no hidden history."""

    position: int
    mode: str
    kv_caches: tuple
    feedback_memory: torch.Tensor

    def state_dict(self):
        caches = []
        for cache in self.kv_caches:
            if cache is None:
                caches.append(None)
            else:
                key, value = cache.prefix()
                caches.append(
                    {
                        "key": key.detach().clone(),
                        "value": value.detach().clone(),
                        "length": cache.length,
                    }
                )
        return {
            "schema": "full_attnres_recurrent_state_v1",
            "position": self.position,
            "mode": self.mode,
            "kv_caches": caches,
            "feedback_memory": self.feedback_memory.detach().clone(),
        }


def shift_teacher_sources(source_values):
    """Detach and shift a [source, batch, time, channel] teacher bank by one."""
    if not isinstance(source_values, torch.Tensor) or source_values.ndim != 4:
        raise ValueError("teacher source bank must have shape [source, batch, time, channel]")
    shifted = torch.zeros_like(source_values)
    shifted[:, :, 1:, :] = source_values[:, :, :-1, :]
    return shifted.detach()


def fixed_derangement(batch_size, device=None):
    """Deterministic fixed-point-free batch permutation for shuffled controls."""
    if batch_size < 2:
        raise ValueError("shuffled feedback requires batch_size > 1")
    return torch.arange(batch_size, device=device).roll(1)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    residual_mode: str = "standard"
    attnres_rms_eps: float = 1e-5
    enable_topdown_feedback: bool = False
    enable_memory_writers: bool = False
    memory_writer_rank: int = 8
    memory_writer_init_seed: int = 20260202

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.residual_mode not in {"standard", "full_attnres"}:
            raise ValueError(f"unknown residual mode: {config.residual_mode}")
        if config.enable_topdown_feedback and config.residual_mode != "full_attnres":
            raise ValueError("top-down feedback requires residual_mode='full_attnres'")
        if config.enable_memory_writers and not config.enable_topdown_feedback:
            raise ValueError("memory writers require the top-down feedback reader")
        if config.enable_memory_writers and config.memory_writer_rank != 8:
            raise ValueError("Experiment 2B2 memory writers require rank 8")
        self.config = config

        transformer = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )
        if config.residual_mode == "full_attnres":
            destinations = []
            for block_index in range(config.n_layer):
                destinations.extend([
                    f"block_{block_index + 1:02d}_attention",
                    f"block_{block_index + 1:02d}_mlp",
                ])
            destinations.append("ln_f_input")
            transformer["attnres"] = nn.ModuleList([
                FullAttnRes(
                    config.n_embd,
                    eps=config.attnres_rms_eps,
                    destination=destination,
                )
                for destination in destinations
            ])
            if config.enable_topdown_feedback:
                transformer["topdown_attnres"] = TopDownAttnRes(
                    config.n_embd,
                    EXPERIMENT_2A0_SOURCE_DEPTHS,
                    eps=config.attnres_rms_eps,
                )
            if config.enable_memory_writers:
                transformer["memory_writers"] = nn.ModuleDict({
                    f"writer_v{depth}": MemoryWriterAdapter(
                        config.n_embd,
                        rank=config.memory_writer_rank,
                        eps=config.attnres_rms_eps,
                    )
                    for depth in EXPERIMENT_2A0_SOURCE_DEPTHS
                })
        self.transformer = nn.ModuleDict(transformer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
        if config.enable_memory_writers:
            self.initialize_memory_writers(config.memory_writer_init_seed)

    def initialize_memory_writers(self, seed):
        """Deterministic down projection and exact zero-effect up projection."""
        if not self.config.enable_memory_writers:
            raise ValueError("this model has no memory writers")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for depth in EXPERIMENT_2A0_SOURCE_DEPTHS:
                writer = self.transformer.memory_writers[f"writer_v{depth}"]
                nn.init.normal_(writer.W_down.weight, mean=0.0, std=0.02)
                nn.init.zeros_(writer.W_up.weight)

    def write_recurrent_memory(self, raw_sources, disabled_writer_depths=()):
        """Adapt detached raw states without changing their same-token consumers."""
        if not self.config.enable_memory_writers:
            raise ValueError("this model has no memory writers")
        if not isinstance(raw_sources, torch.Tensor) or raw_sources.ndim != 4:
            raise ValueError("raw writer sources must have [source,batch,time,channel]")
        disabled = set(disabled_writer_depths or ())
        invalid = disabled - set(EXPERIMENT_2A0_SOURCE_DEPTHS)
        if invalid:
            raise ValueError(f"invalid disabled writer depths: {sorted(invalid)}")
        memories = []
        deltas = []
        for index, depth in enumerate(EXPERIMENT_2A0_SOURCE_DEPTHS):
            source = raw_sources[index].detach()
            if depth in disabled:
                memory = source
                delta = torch.zeros_like(source)
            else:
                memory, delta = self.transformer.memory_writers[
                    f"writer_v{depth}"
                ](source)
            memories.append(memory)
            deltas.append(delta)
        return torch.stack(memories, dim=0), torch.stack(deltas, dim=0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        mode="full_context",
        feedback_sources=None,
        feedback_permutation=None,
        feedback_gate_override=None,
        return_source_depths=None,
    ):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        if mode not in EXPERIMENT_2A0_MODES:
            raise ValueError(f"unknown Experiment 2A0 mode: {mode}")
        if self.config.residual_mode == "standard" and mode != "full_context":
            raise ValueError("masked/top-down modes require residual_mode='full_attnres'")
        if return_source_depths is not None and self.config.residual_mode != "full_attnres":
            raise ValueError("residual source capture requires residual_mode='full_attnres'")
        masked_l1 = mode != "full_context"
        uses_feedback = mode in {
            "masked_l1_topdown_teacher",
            "masked_l1_shuffled_feedback",
        }
        if uses_feedback and not self.config.enable_topdown_feedback:
            raise ValueError("this model was constructed without top-down feedback")
        if uses_feedback and feedback_sources is None:
            raise ValueError(f"{mode} requires shifted teacher feedback sources")
        if not uses_feedback and feedback_sources is not None:
            raise ValueError(f"{mode} does not accept feedback sources")
        if mode != "masked_l1_shuffled_feedback" and feedback_permutation is not None:
            raise ValueError("a feedback permutation is valid only in shuffled-feedback mode")
        if not uses_feedback and feedback_gate_override is not None:
            raise ValueError("a gate override is valid only in a feedback mode")
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        if self.config.residual_mode == "standard":
            # Frozen baseline path: deliberately unchanged.
            for block in self.transformer.h:
                x = block(x)
        else:
            # v0 is the combined token + learned-position embedding. Every
            # Attention and MLP output is then a distinct residual value.
            values = [x]
            destination_index = 0
            for block_index, block in enumerate(self.transformer.h):
                h = self.transformer.attnres[destination_index](values)
                if block_index == 0 and masked_l1:
                    if uses_feedback:
                        if not isinstance(feedback_sources, torch.Tensor) or feedback_sources.ndim != 4:
                            raise ValueError(
                                "feedback sources must have shape [source, batch, time, channel]"
                            )
                        expected_shape = (
                            len(EXPERIMENT_2A0_SOURCE_DEPTHS), B, T, self.config.n_embd
                        )
                        if tuple(feedback_sources.shape) != expected_shape:
                            raise ValueError(
                                f"feedback source shape {tuple(feedback_sources.shape)} != {expected_shape}"
                            )
                        memory_bank = feedback_sources.detach()
                        if mode == "masked_l1_shuffled_feedback":
                            if feedback_permutation is None:
                                feedback_permutation = fixed_derangement(B, idx.device)
                            if tuple(feedback_permutation.shape) != (B,):
                                raise ValueError("feedback permutation must have shape [batch]")
                            expected_indices = torch.arange(B, device=feedback_permutation.device)
                            if torch.any(feedback_permutation == expected_indices):
                                raise ValueError("feedback permutation must be fixed-point-free")
                            if not torch.equal(
                                torch.sort(feedback_permutation).values, expected_indices
                            ):
                                raise ValueError("feedback permutation must contain every batch index once")
                            memory_bank = memory_bank[:, feedback_permutation]
                        topdown = self.transformer.topdown_attnres(
                            list(memory_bank.unbind(dim=0))
                        )
                        if feedback_gate_override is None:
                            gate = self.transformer.topdown_attnres.gate.tanh()
                        else:
                            gate = h.new_tensor(float(feedback_gate_override))
                        h = h + gate * topdown
                    values.append(block.attn(block.ln_1(h), self_only=True))
                else:
                    values.append(block.attn(block.ln_1(h)))
                destination_index += 1
                h = self.transformer.attnres[destination_index](values)
                values.append(block.mlp(block.ln_2(h)))
                destination_index += 1
            # The paper's output layer performs one final depth aggregation.
            x = self.transformer.attnres[destination_index](values)
            captured_sources = None
            if return_source_depths is not None:
                source_depths = tuple(return_source_depths)
                invalid = [depth for depth in source_depths if depth < 0 or depth >= len(values)]
                if invalid:
                    raise ValueError(f"invalid residual source depths: {invalid}")
                captured_sources = torch.stack([values[depth] for depth in source_depths], dim=0)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_source_depths is not None:
            return logits, loss, captured_sources
        return logits, loss

    def init_recurrent_state(self, batch_size, mode, device=None, dtype=None):
        """Create explicit, fixed-capacity state for token-by-token inference."""
        if self.config.residual_mode != "full_attnres":
            raise ValueError("incremental Full-AttnRes requires residual_mode='full_attnres'")
        if mode not in EXPERIMENT_2B0_INCREMENTAL_MODES:
            raise ValueError(f"unknown incremental mode: {mode}")
        if "topdown" in mode or "self_feedback" in mode:
            if not self.config.enable_topdown_feedback:
                raise ValueError("incremental feedback requires a top-down router")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        reference = self.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        head_size = self.config.n_embd // self.config.n_head
        cache_shape = (
            batch_size,
            self.config.n_head,
            self.config.block_size,
            head_size,
        )
        masked_l1 = mode != "full_context"
        caches = []
        for block_index in range(self.config.n_layer):
            if block_index == 0 and masked_l1:
                caches.append(None)
                continue
            caches.append(
                AttentionKVCache(
                    key=torch.empty(cache_shape, device=device, dtype=dtype),
                    value=torch.empty(cache_shape, device=device, dtype=dtype),
                    length=0,
                )
            )
        memory = torch.zeros(
            len(EXPERIMENT_2A0_SOURCE_DEPTHS),
            batch_size,
            1,
            self.config.n_embd,
            device=device,
            dtype=dtype,
        )
        return RecurrentState(0, mode, tuple(caches), memory)

    def load_recurrent_state(self, payload, device=None, dtype=None):
        """Restore a compact recurrent-state payload into fresh KV buffers."""
        if not isinstance(payload, dict) or payload.get("schema") != "full_attnres_recurrent_state_v1":
            raise ValueError("invalid recurrent-state payload")
        memory = payload.get("feedback_memory")
        if not isinstance(memory, torch.Tensor) or memory.ndim != 4:
            raise ValueError("invalid recurrent feedback memory")
        position = payload.get("position")
        if not isinstance(position, int) or not 0 <= position <= self.config.block_size:
            raise ValueError("invalid recurrent position")
        device = memory.device if device is None else torch.device(device)
        dtype = memory.dtype if dtype is None else dtype
        state = self.init_recurrent_state(
            memory.size(1), payload.get("mode"), device=device, dtype=dtype
        )
        caches = payload.get("kv_caches")
        if not isinstance(caches, (list, tuple)) or len(caches) != self.config.n_layer:
            raise ValueError("invalid recurrent cache collection")
        restored = []
        for fresh, saved in zip(state.kv_caches, caches):
            if fresh is None or saved is None:
                if fresh is not None or saved is not None:
                    raise ValueError("recurrent Block-1 cache policy mismatch")
                restored.append(None)
                continue
            if set(saved) != {"key", "value", "length"} or saved["length"] != position:
                raise ValueError("invalid serialized KV cache")
            expected_prefix = fresh.key[:, :, :position]
            if (
                tuple(saved["key"].shape) != tuple(expected_prefix.shape)
                or tuple(saved["value"].shape) != tuple(expected_prefix.shape)
            ):
                raise ValueError("serialized KV cache shape mismatch")
            fresh.key[:, :, :position].copy_(saved["key"].to(device=device, dtype=dtype))
            fresh.value[:, :, :position].copy_(saved["value"].to(device=device, dtype=dtype))
            restored.append(AttentionKVCache(fresh.key, fresh.value, position))
        expected_memory = (
            len(EXPERIMENT_2A0_SOURCE_DEPTHS),
            memory.size(1),
            1,
            self.config.n_embd,
        )
        if tuple(memory.shape) != expected_memory:
            raise ValueError("serialized recurrent memory shape mismatch")
        return RecurrentState(
            position,
            state.mode,
            tuple(restored),
            memory.detach().to(device=device, dtype=dtype).clone(),
        )

    def reset_recurrent_memory(self, state):
        """Reset only high-to-low memory; preserve position and Blocks 2–12 caches."""
        self._validate_recurrent_state(state, state.feedback_memory.size(1))
        return RecurrentState(
            state.position,
            state.mode,
            state.kv_caches,
            torch.zeros_like(state.feedback_memory),
        )

    def _validate_recurrent_state(self, state, batch_size):
        if not isinstance(state, RecurrentState):
            raise ValueError("forward_step requires an explicit RecurrentState")
        if state.mode not in EXPERIMENT_2B0_INCREMENTAL_MODES:
            raise ValueError("recurrent state has an unknown mode")
        if not isinstance(state.position, int) or not 0 <= state.position < self.config.block_size:
            raise ValueError("recurrent position is outside the configured context")
        expected_memory = (
            len(EXPERIMENT_2A0_SOURCE_DEPTHS),
            batch_size,
            1,
            self.config.n_embd,
        )
        if tuple(state.feedback_memory.shape) != expected_memory:
            raise ValueError("recurrent feedback-memory shape mismatch")
        if len(state.kv_caches) != self.config.n_layer:
            raise ValueError("recurrent state has the wrong number of KV caches")
        masked_l1 = state.mode != "full_context"
        for block_index, cache in enumerate(state.kv_caches):
            if block_index == 0 and masked_l1:
                if cache is not None:
                    raise ValueError("masked Block 1 must not retain a KV cache")
            elif not isinstance(cache, AttentionKVCache) or cache.length != state.position:
                raise ValueError("recurrent KV cache length mismatch")

    def forward_step(
        self,
        idx,
        state,
        feedback_sources=None,
        feedback_permutation=None,
        feedback_gate_override=None,
        reset_feedback=False,
        use_memory_writers=False,
        disabled_writer_depths=(),
        return_diagnostics=False,
    ):
        """Process one token with explicit Full-AttnRes KV and feedback state."""
        if idx.ndim == 1:
            idx = idx.unsqueeze(1)
        if idx.ndim != 2 or idx.size(1) != 1:
            raise ValueError("forward_step token input must have shape [batch] or [batch, 1]")
        B = idx.size(0)
        self._validate_recurrent_state(state, B)
        mode = state.mode
        teacher_mode = mode == "masked_l1_topdown_teacher"
        self_mode = mode in {
            "masked_l1_topdown_self",
            "masked_l1_shuffled_self_feedback",
        }
        uses_feedback = teacher_mode or self_mode
        if teacher_mode != (feedback_sources is not None):
            raise ValueError("teacher mode requires exactly one current-token feedback bank")
        shuffled = mode == "masked_l1_shuffled_self_feedback"
        if shuffled != (feedback_permutation is not None):
            raise ValueError("shuffled self-feedback requires exactly one permutation")
        if not uses_feedback and feedback_gate_override is not None:
            raise ValueError("gate override is valid only for a feedback mode")
        reset_mask = None
        if isinstance(reset_feedback, torch.Tensor):
            if (
                reset_feedback.dtype != torch.bool
                or tuple(reset_feedback.shape) != (B,)
                or reset_feedback.device != idx.device
            ):
                raise ValueError("per-example feedback reset mask must be boolean [batch]")
            reset_mask = reset_feedback
        elif not isinstance(reset_feedback, bool):
            raise ValueError("reset_feedback must be a boolean or boolean [batch] tensor")
        if (reset_feedback is True or reset_mask is not None) and not self_mode:
            raise ValueError("memory-only reset is valid only for self-feedback modes")
        if use_memory_writers and not self.config.enable_memory_writers:
            raise ValueError("writer recurrence requires configured memory writers")
        if use_memory_writers and not self_mode:
            raise ValueError("writer recurrence is valid only for self-feedback modes")

        position = state.position
        pos = torch.tensor([position], dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        values = [x]
        destination_index = 0
        updated_caches = []
        topdown = None
        feedback_contribution = None
        topdown_weights = None
        for block_index, block in enumerate(self.transformer.h):
            h = self.transformer.attnres[destination_index](values)
            if block_index == 0 and mode != "full_context":
                if uses_feedback:
                    if teacher_mode:
                        memory_bank = feedback_sources.detach()
                    elif reset_mask is not None:
                        memory_bank = torch.where(
                            reset_mask.view(1, B, 1, 1),
                            torch.zeros_like(state.feedback_memory),
                            state.feedback_memory
                            if use_memory_writers
                            else state.feedback_memory.detach(),
                        )
                    elif reset_feedback:
                        memory_bank = torch.zeros_like(state.feedback_memory)
                    else:
                        memory_bank = (
                            state.feedback_memory
                            if use_memory_writers
                            else state.feedback_memory.detach()
                        )
                    expected_shape = (
                        len(EXPERIMENT_2A0_SOURCE_DEPTHS),
                        B,
                        1,
                        self.config.n_embd,
                    )
                    if tuple(memory_bank.shape) != expected_shape:
                        raise ValueError("incremental feedback source shape mismatch")
                    if memory_bank.device != idx.device:
                        raise ValueError("incremental feedback source device mismatch")
                    if shuffled:
                        permutation = feedback_permutation
                        if tuple(permutation.shape) != (B,):
                            raise ValueError("feedback permutation must have shape [batch]")
                        expected_indices = torch.arange(B, device=permutation.device)
                        if (
                            torch.any(permutation == expected_indices)
                            or not torch.equal(torch.sort(permutation).values, expected_indices)
                        ):
                            raise ValueError("feedback permutation must be fixed-point-free")
                        memory_bank = memory_bank[:, permutation]
                    if return_diagnostics:
                        topdown, topdown_weights = self.transformer.topdown_attnres(
                            list(memory_bank.unbind(dim=0)), return_weights=True
                        )
                    else:
                        topdown = self.transformer.topdown_attnres(
                            list(memory_bank.unbind(dim=0))
                        )
                    if feedback_gate_override is None:
                        gate = self.transformer.topdown_attnres.gate.tanh()
                    else:
                        gate = h.new_tensor(float(feedback_gate_override))
                    feedback_contribution = gate * topdown
                    h = h + feedback_contribution
                attention_output, cache = block.attn.forward_step(
                    block.ln_1(h), cache=None, self_only=True
                )
            else:
                attention_output, cache = block.attn.forward_step(
                    block.ln_1(h), cache=state.kv_caches[block_index]
                )
            values.append(attention_output)
            updated_caches.append(cache)
            destination_index += 1
            h = self.transformer.attnres[destination_index](values)
            values.append(block.mlp(block.ln_2(h)))
            destination_index += 1

        x = self.transformer.attnres[destination_index](values)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        raw_memory = torch.stack(
            [values[depth] for depth in EXPERIMENT_2A0_SOURCE_DEPTHS], dim=0
        )
        writer_delta = None
        if use_memory_writers:
            memory, writer_delta = self.write_recurrent_memory(
                raw_memory, disabled_writer_depths=disabled_writer_depths
            )
        else:
            memory = raw_memory.detach()
        next_state = RecurrentState(
            position + 1,
            mode,
            tuple(updated_caches),
            memory,
        )
        if not return_diagnostics:
            return logits, next_state
        raw_diagnostic = raw_memory.detach()
        adapted_diagnostic = memory.detach()
        source_rms = raw_diagnostic.float().pow(2).mean(dim=(2, 3)).sqrt()
        adapted_rms = adapted_diagnostic.float().pow(2).mean(dim=(2, 3)).sqrt()
        diagnostics = {
            "position": position,
            "source_memory": raw_diagnostic,
            "adapted_memory": adapted_diagnostic,
            "writer_delta": None if writer_delta is None else writer_delta.detach(),
            "source_rms": source_rms,
            "adapted_rms": adapted_rms,
            "topdown_rms": None,
            "feedback_rms": None,
            "routing_weights": None,
            "routing_entropy": None,
        }
        if topdown is not None:
            safe_weights = topdown_weights.float().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            diagnostics.update(
                {
                    "topdown_rms": topdown.float().pow(2).mean(dim=(1, 2)).sqrt(),
                    "feedback_rms": feedback_contribution.float()
                    .pow(2)
                    .mean(dim=(1, 2))
                    .sqrt(),
                    "routing_weights": topdown_weights.detach(),
                    "routing_entropy": -(
                        safe_weights * safe_weights.log()
                    ).sum(dim=0).squeeze(-1),
                }
            )
        return logits, next_state, diagnostics

    def capture_residual_sources(
        self,
        idx,
        source_depths=EXPERIMENT_2A0_SOURCE_DEPTHS,
    ):
        """Run the original full-context stack and return selected raw values."""
        if self.config.residual_mode != "full_attnres":
            raise ValueError("residual source capture requires residual_mode='full_attnres'")
        _, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        source_depths = tuple(source_depths)
        invalid = [depth for depth in source_depths if depth < 0 or depth > 2 * self.config.n_layer]
        if invalid:
            raise ValueError(f"invalid residual source depths: {invalid}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        values = [x]
        destination_index = 0
        for block in self.transformer.h:
            h = self.transformer.attnres[destination_index](values)
            values.append(block.attn(block.ln_1(h)))
            destination_index += 1
            h = self.transformer.attnres[destination_index](values)
            values.append(block.mlp(block.ln_2(h)))
            destination_index += 1
        return torch.stack([values[depth] for depth in source_depths], dim=0)

    def set_attnres_instrumentation(self, enabled=True):
        if self.config.residual_mode != "full_attnres":
            return
        for router in self.transformer.attnres:
            router.instrumentation_enabled = enabled
            if enabled:
                router.last_stats = None

    def get_attnres_stats(self):
        if self.config.residual_mode != "full_attnres":
            return []
        return [
            router.last_stats
            for router in self.transformer.attnres
            if router.last_stats is not None
        ]

    def set_attnres_source_mask(self, source_depth=None):
        if self.config.residual_mode != "full_attnres":
            raise ValueError("source ablation requires residual_mode='full_attnres'")
        if source_depth is not None and source_depth < 0:
            raise ValueError("source depth must be non-negative")
        for router in self.transformer.attnres:
            # Source 0 is the only input to the first sublayer and cannot be
            # removed there. Mask it from every later destination instead.
            router.masked_source = source_depth if source_depth != 0 or router is not self.transformer.attnres[0] else None

    def set_topdown_instrumentation(self, enabled=True):
        if not self.config.enable_topdown_feedback:
            raise ValueError("this model has no top-down router")
        router = self.transformer.topdown_attnres
        router.instrumentation_enabled = enabled
        if enabled:
            router.last_stats = None

    def get_topdown_stats(self):
        if not self.config.enable_topdown_feedback:
            return None
        return self.transformer.topdown_attnres.last_stats

    def set_topdown_source_mask(self, source_depth=None):
        if not self.config.enable_topdown_feedback:
            raise ValueError("this model has no top-down router")
        router = self.transformer.topdown_attnres
        if source_depth is None:
            router.masked_source = None
            return
        if source_depth not in router.source_depths:
            raise ValueError(
                f"top-down source must be one of {router.source_depths}, got {source_depth}"
            )
        router.masked_source = router.source_depths.index(source_depth)

    def load_experiment1_full_attnres_state(self, experiment1_state):
        """Load an Experiment 1 state while preserving only new 2A0 tensors."""
        if not self.config.enable_topdown_feedback:
            raise ValueError("use strict load_state_dict for a model without top-down feedback")
        missing, unexpected = self.load_state_dict(experiment1_state, strict=False)
        expected_missing = {
            "transformer.topdown_attnres.query",
            "transformer.topdown_attnres.norm.weight",
            "transformer.topdown_attnres.gate",
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                f"Experiment 1 state mismatch: missing={missing}, unexpected={unexpected}"
            )
        router = self.transformer.topdown_attnres
        if (
            torch.count_nonzero(router.query).item() != 0
            or not torch.equal(router.norm.weight, torch.ones_like(router.norm.weight))
            or torch.count_nonzero(router.gate).item() != 0
        ):
            raise RuntimeError("top-down parameters lost their exact zero/one initialization")

    def freeze_for_topdown_training(self):
        if not self.config.enable_topdown_feedback:
            raise ValueError("this model has no top-down feedback parameters")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.transformer.topdown_attnres.parameters():
            parameter.requires_grad_(True)

    def freeze_for_memory_writer_training(self):
        if not self.config.enable_memory_writers:
            raise ValueError("this model has no memory writers")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.transformer.memory_writers.parameters():
            parameter.requires_grad_(True)

    def load_shared_baseline_state(self, baseline_state):
        """Load every GPT-2 tensor and leave only AttnRes tensors initialized."""
        current = self.state_dict()
        unexpected = sorted(set(baseline_state) - set(current))
        shape_mismatches = sorted(
            key for key in baseline_state
            if key in current and current[key].shape != baseline_state[key].shape
        )
        if unexpected or shape_mismatches:
            raise ValueError(
                f"baseline state mismatch: unexpected={unexpected}, shape_mismatches={shape_mismatches}"
            )
        merged = dict(current)
        for key, value in baseline_state.items():
            merged[key] = value
        missing, unexpected_after = self.load_state_dict(merged, strict=True)
        if missing or unexpected_after:
            raise RuntimeError(f"failed to load shared baseline state: missing={missing}, unexpected={unexpected_after}")

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def render_example(*args, **kwargs):
    from hellaswag import render_example as implementation
    return implementation(*args, **kwargs)


def iterate_examples(*args, **kwargs):
    from hellaswag import iterate_examples as implementation
    return implementation(*args, **kwargs)

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

if master_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    param_dtypes = sorted({str(p.dtype) for p in raw_model.parameters()})
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"parameter dtypes: {param_dtypes}")
    if device_type == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"model VRAM before first forward: allocated={allocated_mb:.2f}MB reserved={reserved_mb:.2f}MB peak={peak_mb:.2f}MB")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = os.environ.get("NANOGPT_LOG_DIR", "log")
checkpoint_dir = os.environ.get("NANOGPT_CHECKPOINT_DIR", log_dir)
metrics_file = os.environ.get("NANOGPT_METRICS_FILE")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
if metrics_file is not None and master_process:
    metrics_dir = os.path.dirname(metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_file, "w") as f:
        pass

def get_gpu_memory_mb():
    if device_type != "cuda":
        return None, None, None
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return allocated_mb, reserved_mb, peak_mb

def write_metrics(kind, step, tokens, train_loss=None, val_loss=None, hellaswag_accuracy=None, lr=None, grad_norm=None, step_time_ms=None, tokens_per_second=None):
    if metrics_file is None or not master_process:
        return
    allocated_mb, reserved_mb, peak_mb = get_gpu_memory_mb()
    row = {
        "kind": kind,
        "step": step,
        "tokens": tokens,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "hellaswag_accuracy": hellaswag_accuracy,
        "lr": lr,
        "grad_norm": grad_norm,
        "step_time_ms": step_time_ms,
        "tokens_per_second": tokens_per_second,
        "gpu_allocated_mb": allocated_mb,
        "gpu_reserved_mb": reserved_mb,
        "gpu_peak_mb": peak_mb,
    }
    with open(metrics_file, "a") as f:
        f.write(json.dumps(row) + "\n")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            write_metrics("val", step, step * total_batch_size, val_loss=val_loss_accum.item())
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            write_metrics("hellaswag", step, step * total_batch_size, hellaswag_accuracy=acc_norm)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        if device_type == "cuda" and step == 0:
            allocated_mb, reserved_mb, peak_mb = get_gpu_memory_mb()
            print(f"peak VRAM after first backward: allocated={allocated_mb:.2f}MB reserved={reserved_mb:.2f}MB peak={peak_mb:.2f}MB")
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        write_metrics(
            "train",
            step,
            (step + 1) * total_batch_size,
            train_loss=loss_accum.item(),
            lr=lr,
            grad_norm=float(norm),
            step_time_ms=dt * 1000,
            tokens_per_second=tokens_per_sec,
        )

if ddp:
    destroy_process_group()
