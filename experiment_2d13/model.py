"""Single-pass structural ALL_OFF with explicit backbone-only trainability."""
import torch
from experiment_2d12.model import AblatedH
from experiment_2d11.model import fresh_base,original_symbols,tensor_identity,optimizer_for,optimizer_names
from .common import *
class LocalOnly(AblatedH):
    def __init__(self,base):
        super().__init__(base,'H_ALL_OFF')
        self.requires_grad_(False)
        self.base.requires_grad_(True)
    def forward(self,x,y,activation_checkpointing=True):
        assert self.condition=='H_ALL_OFF' and self.disabled_destinations==frozenset((0,2,4))
        return self.forward_pass(x,targets=y,activation_checkpointing=activation_checkpointing)['loss']

def template(tiny=False):
    symbols=original_symbols()
    c=symbols['GPTConfig'](**(dict(block_size=70,vocab_size=32,n_embd=16,n_head=2,n_layer=12) if tiny else dict(vocab_size=50304)))
    c.residual_mode='standard'
    with torch.device('meta'):return LocalOnly(symbols['GPT'](c))

def from_state(state,device='cpu',tiny=False):
    m=template(tiny)
    m.load_state_dict(state,strict=True,assign=True)
    assert torch.equal(m.base.lm_head.weight,m.base.transformer.wte.weight)
    m.base.lm_head.weight=m.base.transformer.wte.weight
    m.requires_grad_(False);m.base.requires_grad_(True)
    return m.to(device)

def frozen_identity(model):return tensor_identity((n,p) for n,p in model.named_parameters() if not p.requires_grad)
def trainability(model):
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith('base.') for n in names)
    assert all(p.requires_grad for p in model.base.parameters())
    assert model.base.lm_head.weight is model.base.transformer.wte.weight
    return dict(registered=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),active_names=names,frozen_names=[n for n,p in model.named_parameters() if not p.requires_grad],frozen_tensor_identity=frozen_identity(model),tied=True)

def load_h(path,device):
    assert sha256(path)==H_SHA
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==1000 and p['accounting']['logical_targets']==524288000
    m=from_state(p['model'],device);m.set_condition('H_ON');m.requires_grad_(False);m.eval()
    assert tensor_identity(m.named_parameters())==H_MODEL_SHA
    return m
