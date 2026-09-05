"""Original GPT-2 definitions, sealed H kernels, and an explicit fresh path."""
import ast
from dataclasses import dataclass
import hashlib
import inspect
import math
import os
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from .common import FROZEN, REPO, learning_rate

sys.path.insert(0, str(REPO/'scripts'))
from experiment_2d10_core import RetrievalGatingGPT, BLOCKS
from experiment_2d3a_core import AlternatingIntegrationRecurrentPyramidGPT
from experiment_2d2a_core import LayerKVCache


def original_symbols():
    """Execute only an explicit AST allowlist; never upstream top-level training."""
    name = 'experiment_2d11_original_symbols'
    if name in sys.modules:
        return sys.modules[name].__dict__
    source = (FROZEN/'original_train_gpt2.py.txt').read_text()
    allowed = {'CausalSelfAttention', 'MLP', 'Block', 'GPTConfig', 'GPT',
               'DataLoaderLite', 'load_tokens', 'get_most_likely_row', 'get_lr'}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in allowed]
    if {n.name for n in nodes} != allowed:
        raise RuntimeError('original definition allowlist mismatch')
    module = types.ModuleType(name)
    sys.modules[name] = module
    import numpy as np
    module.__dict__.update(torch=torch, nn=nn, F=F, dataclass=dataclass, math=math,
                           inspect=inspect, os=os, np=np, master_process=False,
                           max_lr=6e-4, min_lr=6e-4*.1, warmup_steps=715, max_steps=19073)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FROZEN/'original_train_gpt2.py.txt'), 'exec'), module.__dict__)
    return module.__dict__


def fresh_base(tiny=False):
    symbols = original_symbols()
    torch.manual_seed(1337)
    config = symbols['GPTConfig'](**(dict(block_size=70, vocab_size=32, n_embd=16,
                                         n_head=2, n_layer=12) if tiny else dict(vocab_size=50304)))
    base = symbols['GPT'](config)
    # This metadata satisfies the sealed wrapper's check; it changes no GPT tensor or computation.
    config.residual_mode = 'standard'
    return base


class FreshRouter(nn.Module):
    def __init__(self, width, block):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(20260916+block)
        self.W1 = nn.Parameter(torch.randn(32, 3*width, generator=generator)*.02)
        self.b1 = nn.Parameter(torch.zeros(32))
        self.W2 = nn.Parameter(torch.zeros(2, 32))
        self.b2 = nn.Parameter(torch.zeros(2))


class FreshH(RetrievalGatingGPT):
    def __init__(self, base):
        super().__init__(base)
        self.arm = 'H'
        self.routers = nn.ModuleDict({str(b): FreshRouter(base.config.n_embd, b) for b in BLOCKS})
        for name in ('g_rec', 'g_rec_b3', 'g_rec_b5', 'g_rec_b6'):
            getattr(self, name).requires_grad_(False)
            with torch.no_grad():
                getattr(self, name).zero_()

    def forward(self, tokens, targets, num_passes=2, activation_checkpointing=True):
        # DDP's entrypoint owns every pass and the attached temporal loss graph.
        result = self.forward_multi_pass(tokens, targets=targets, num_passes=num_passes,
                                        activation_checkpointing=activation_checkpointing)
        return result['loss'], torch.stack([x.detach() for x in result['pass_losses']])


def tensor_identity(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()+b'\0')
        h.update(str(tuple(value.shape)).encode()+str(value.dtype).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizer_for(model, device):
    active = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [dict(params=[p for _, p in active if p.ndim >= 2], weight_decay=.1),
              dict(params=[p for _, p in active if p.ndim < 2], weight_decay=0.)]
    return torch.optim.AdamW(groups, lr=learning_rate(0), betas=(.9, .95), eps=1e-8,
                             fused=torch.device(device).type == 'cuda')


def uninitialized_h(tiny=False):
    """Shape template without touching scientific global RNG (safe in CPU writer)."""
    symbols=original_symbols()
    config=symbols['GPTConfig'](**(dict(block_size=70,vocab_size=32,n_embd=16,n_head=2,n_layer=12)
                                  if tiny else dict(vocab_size=50304)))
    config.residual_mode='standard'
    with torch.device('meta'):
        return FreshH(symbols['GPT'](config))


def optimizer_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]


@dataclass(frozen=True)
class GState:
    position: int
    batch_size: int
    caches: tuple


class IncrementalG(nn.Module):
    """Original causal GPT-2 with compact W-1 historical KV; no recurrent rings."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config

    _append_cache = staticmethod(AlternatingIntegrationRecurrentPyramidGPT._append_cache)
    _incremental_ordinary_block = AlternatingIntegrationRecurrentPyramidGPT._incremental_ordinary_block

    def init_incremental_state(self, batch_size, **kwargs):
        return GState(0, batch_size, (None,)*self.config.n_layer)

    def incremental_step(self, token, state, **kwargs):
        if state.position >= self.config.block_size or token.numel() != state.batch_size:
            raise ValueError('invalid G incremental input')
        token = token.reshape(state.batch_size, 1)
        position = torch.tensor([state.position], device=token.device)
        residual = self.base.transformer.wte(token)+self.base.transformer.wpe(position)
        caches = []
        for index in range(self.config.n_layer):
            residual, cache = self._incremental_ordinary_block(
                residual, index, state.caches[index], self.config.block_size-1)
            caches.append(cache)
        logits = self.base.lm_head(self.base.transformer.ln_f(residual))
        return logits, GState(state.position+1, state.batch_size, tuple(caches))


def load_g(path, device='cpu'):
    # Legacy checkpoint config was pickled as __main__.GPTConfig.
    with torch.serialization.safe_globals([(original_symbols()['GPTConfig'], '__main__.GPTConfig')]):
        payload = torch.load(path, map_location='cpu', weights_only=True)
    model = fresh_base()
    model.load_state_dict(payload['model'], strict=True)
    if payload['step'] != 19072:
        raise ValueError('wrong historical endpoint')
    return IncrementalG(model).to(device)
