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

class LocalOnly(AblatedH):
    def __init__(self,base):
        super().__init__(base,'H_ALL_OFF')
        self.requires_grad_(False)
        self.base.requires_grad_(True)
    def forward(self,x,y,num_passes=1,activation_checkpointing=True):
        assert self.condition=='H_ALL_OFF' and self.disabled_destinations==frozenset((0,2,4))
        loss=self.forward_pass(x,targets=y,activation_checkpointing=activation_checkpointing)['loss']
        return loss,loss.detach().reshape(1)


def template(arm,tiny=False):
    assert arm in ARMS
    symbols=original_symbols()
    config=symbols['GPTConfig'](**(dict(block_size=70,vocab_size=32,n_embd=16,n_head=2,n_layer=12) if tiny else dict(vocab_size=50304)))
    config.residual_mode='standard'
    with torch.device('meta'):
        return (LocalOnly if arm=='L_nf4' else CE1FreeH)(symbols['GPT'](config))

def from_state(state,arm,device='cpu',tiny=False):
    m=template(arm,tiny);m.load_state_dict(state,strict=True,assign=True)
    assert torch.equal(m.base.lm_head.weight,m.base.transformer.wte.weight)
    m.base.lm_head.weight=m.base.transformer.wte.weight
    if arm=='L_nf4':m.requires_grad_(False);m.base.requires_grad_(True)
    m.arm=arm
    return m.to(device)

def frozen_identity(model):return tensor_identity((n,p) for n,p in model.named_parameters() if not p.requires_grad)

def trainability(model):
    active=[n for n,p in model.named_parameters() if p.requires_grad]
    frozen=[n for n,p in model.named_parameters() if not p.requires_grad]
    assert model.base.lm_head.weight is model.base.transformer.wte.weight
    if model.arm=='L_nf4':assert all(n.startswith('base.') for n in active)
    else:assert set(frozen)=={'g_rec','g_rec_b3','g_rec_b5','g_rec_b6'}
    return dict(registered=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),active_names=active,frozen_names=frozen,frozen_tensor_identity=frozen_identity(model),tied=True)

def optimizer_for(model,device,options=None):
    if options is None:options=read_json(FROZEN/'OPTIMIZER.json')['initial_groups']
    assert len(options)==2
    active=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups=[]
    for i,opts in enumerate(options):
        opts=dict(opts);assert opts['fused'] is False and opts['foreach'] is None
        opts['betas']=tuple(opts['betas'])
        opts['params']=[p for _,p in active if (p.ndim>=2)==(i==0)]
        groups.append(opts)
    return torch.optim.AdamW(groups,lr=learning_rate(0),betas=(.9,.95),eps=1e-8,fused=False,foreach=None)

def assert_optimizer(opt,u=None):
    expected=read_json(FROZEN/'OPTIMIZER.json')['initial_groups']
    for g,e in zip(opt.param_groups,expected):
        assert set(g)-{'params'}==set(e)
        for k,v in e.items():
            if k=='lr':continue
            assert (list(g[k]) if isinstance(g[k],tuple) else g[k])==v,(k,g[k],v)
        if u is not None:assert g['lr']==historical_lrs()[max(u-1,0)]

def load_h(path,u,device):
    digest,model_id=H_CHECKPOINTS[u];assert sha256(path)==digest
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert p['completed_updates']==u
    m=from_state(p['model'],'R_nf4',device).eval().requires_grad_(False)
    assert tensor_identity(m.named_parameters())==model_id
    return m
