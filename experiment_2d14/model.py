"""The full H graph, with CE1 diagnostic-only; all recurrent writers attached."""
import torch
from experiment_2d12.model import AblatedH
from experiment_2d11.model import fresh_base, original_symbols, tensor_identity, optimizer_for, optimizer_names
from .common import *

class CE1FreeH(AblatedH):
    def __init__(self, base):
        super().__init__(base, 'H_ON')
        self.arm = 'R'

    def forward_multi_pass(self, *args, **kwargs):
        result = super().forward_multi_pass(*args, **kwargs)
        losses = result['pass_losses']
        result['pass_weights'] = (0., 1.) if len(losses) == 2 else (0., .5, .5)
        # CE1 is computed by the sealed kernel, but has no direct loss edge.
        if losses[0] is not None:
            result['loss'] = losses[1] if len(losses) == 2 else .5*losses[1] + .5*losses[2]
        return result

    def forward(self, tokens, targets, num_passes=2, activation_checkpointing=True):
        assert self.condition == 'H_ON'
        result = self.forward_multi_pass(tokens, targets=targets, num_passes=num_passes,
                                         activation_checkpointing=activation_checkpointing)
        return result['loss'], torch.stack([v.detach() for v in result['pass_losses']])

def template(tiny=False):
    symbols=original_symbols()
    config=symbols['GPTConfig'](**(dict(block_size=70,vocab_size=32,n_embd=16,n_head=2,n_layer=12)
                                   if tiny else dict(vocab_size=50304)))
    config.residual_mode='standard'
    with torch.device('meta'):
        return CE1FreeH(symbols['GPT'](config))

def from_state(state, device='cpu', tiny=False):
    m=template(tiny)
    m.load_state_dict(state,strict=True,assign=True)
    assert torch.equal(m.base.lm_head.weight,m.base.transformer.wte.weight)
    m.base.lm_head.weight=m.base.transformer.wte.weight
    return m.to(device)

def frozen_identity(model):
    return tensor_identity((n,p) for n,p in model.named_parameters() if not p.requires_grad)

def trainability(model):
    frozen=[n for n,p in model.named_parameters() if not p.requires_grad]
    assert set(frozen)=={'g_rec','g_rec_b3','g_rec_b5','g_rec_b6'}
    assert all(p.numel()==1 and p.item()==0 for n,p in model.named_parameters() if n in frozen)
    assert all(p.requires_grad for p in model.routers.parameters())
    assert model.base.lm_head.weight is model.base.transformer.wte.weight
    return dict(registered=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen=4,active_names=[n for n,p in model.named_parameters() if p.requires_grad],
                frozen_names=frozen,frozen_tensor_identity=frozen_identity(model),tied=True)

def load_h(path,device):
    assert sha256(path)==H_SHA
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==1000 and p['accounting']['logical_targets']==524288000
    m=from_state(p['model'],device).eval().requires_grad_(False)
    assert tensor_identity(m.named_parameters())==H_MODEL_SHA
    return m
