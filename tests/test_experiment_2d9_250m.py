"""Focused continuation accounting and exact learned-state resume tests."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import smoke_test
import experiment_2d9_250m as run
SYMBOLS = smoke_test.load_training_symbols()
torch.set_num_threads(1)


def foundation(device):
    return None, SYMBOLS['GPT'](SYMBOLS['GPTConfig'](block_size=70, vocab_size=32,
        n_layer=12, n_head=2, n_embd=16, residual_mode='standard')).to(device)


def update(model, optimizer, loader):
    x, y = loader.next_batch()
    optimizer.zero_grad(set_to_none=True)
    result = model.forward_multi_pass(x, targets=y, num_passes=2, activation_checkpointing=True)
    result['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


@pytest.mark.parametrize('arm', ['S', 'D'])
def test_checkpoint_resume_preserves_next_update_and_rng(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(run.base, 'instantiate_base', foundation)
    monkeypatch.setattr(run, 'loader_from_state', lambda s: run.base.d1.ExplicitShardLoader(s['shards'], s['batch_size'], s['sequence_length'], state=s))
    _, f = foundation(torch.device('cpu'))
    model = run.core.DynamicGatingGPT(f, 'S')
    optimizer = run.base.configure_optimizer(model, 'cpu')
    if arm == 'D': run.append_dynamic(model, optimizer)
    with torch.no_grad():
        for b in (0, 2, 4): model.gate_parameter(b).fill_(.05)
        if arm == 'D':
            for n in run.core.W_NAMES.values(): getattr(model, n).copy_(torch.linspace(-.01, .02, 16))
    tokens = tmp_path/'tokens.npy'; np.save(tokens, np.arange(2000).astype(np.int32)%32)
    loader = run.base.d1.ExplicitShardLoader([str(tokens)], 2, 70)
    update(model, optimizer, loader)
    # Deliberately unequal old Adam counters, plus the learned vector counters.
    for index, (name, p) in enumerate(model.named_parameters()):
        if name == 'g_rec_b6':
            optimizer.state[p] = {'step': torch.tensor(17.), 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}
        else:
            optimizer.state[p]['step'].fill_(191 if name in run.core.W_NAMES.values() else 51 + index%17)
    rng = run.base.capture_rng()
    payload = {'schema': run.SCHEMA, 'arm': arm, 'model': copy.deepcopy(model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()), 'optimizer_parameter_names': run.optimizer_names(model, optimizer),
        'loader_state': loader.state_dict(), 'rng_state': rng}
    path = tmp_path/'checkpoint.pt'; torch.save(payload, path)
    source_steps = run.d6.optimizer_steps_by_name(model, optimizer)
    dormant = run.d6.dormant_state(model, optimizer)
    expected_random = torch.rand(5)
    update(model, optimizer, loader)
    expected_model = copy.deepcopy(model.state_dict()); expected_opt = copy.deepcopy(optimizer.state_dict())
    restored, opt, stream, _ = run.load_final_checkpoint(path, torch.device('cpu'), restore=True)
    assert torch.equal(torch.rand(5), expected_random)
    assert run.state_equal(restored.state_dict(), payload['model'])
    assert run.state_equal(opt.state_dict(), payload['optimizer'])
    assert run.d6.optimizer_steps_by_name(restored, opt) == source_steps
    assert stream.state_dict() == payload['loader_state']
    update(restored, opt, stream)
    assert run.state_equal(restored.state_dict(), expected_model)
    assert run.state_equal(opt.state_dict(), expected_opt)
    assert stream.state_dict() == loader.state_dict()
    assert run.d6.dormant_state(restored, opt) == dormant
    if arm == 'D':
        assert all(run.d6.optimizer_steps_by_name(restored, opt)[n] == 192 for n in run.core.W_NAMES.values())


def test_exact_budget_global_cadence_and_distinct_sources():
    assert run.PARENT_GLOBAL_UPDATE + run.LOCAL_UPDATES == run.FINAL_GLOBAL_UPDATE == 2767
    assert run.LOCAL_UPDATES*run.TARGETS_PER_UPDATE == run.LOCAL_TARGETS == 149946368
    assert run.PRIOR_UPDATES+run.LOCAL_UPDATES == 477
    assert run.PRIOR_TARGETS+run.LOCAL_TARGETS == 250085376
    assert run.PARENT_TARGETS+run.LOCAL_TARGETS == run.FINAL_TARGETS == 1450704896
    globals_ = list(range(run.PARENT_GLOBAL_UPDATE+1, run.FINAL_GLOBAL_UPDATE+1))
    assert [g for g in globals_ if run.base.pass_count(g)==3] == [2496,2528,2560,2592,2624,2656,2688,2720,2752]
    assert sum(run.base.pass_count(g)==2 for g in globals_) == 277
    assert run.SOURCE_SHA256['S'] != run.SOURCE_SHA256['D']
    assert run.PANEL_SEED == 20260907


def test_exact_state_check_detects_lost_moment_and_counter():
    source={'state':{0:{'step':torch.tensor(191.),'exp_avg':torch.tensor([.1,.2]),'exp_avg_sq':torch.tensor([.01,.03])}},'param_groups':[{'params':[0],'lr':3e-5}]}
    candidate=copy.deepcopy(source)
    assert run.state_equal(source,candidate)
    candidate['state'][0]['exp_avg'].zero_()
    assert not run.state_equal(source,candidate)
    candidate=copy.deepcopy(source);candidate['state'][0]['step'].zero_()
    assert not run.state_equal(source,candidate)


def test_optimizer_name_mapping_rejects_reordered_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(run.base, 'instantiate_base', foundation)
    _, f=foundation(torch.device('cpu'));m=run.core.DynamicGatingGPT(f,'S')
    opt=run.base.configure_optimizer(m,'cpu');run.append_dynamic(m,opt)
    names=run.optimizer_names(m,opt);names[-1]=list(reversed(names[-1]))
    p=tmp_path/'wrong.pt';torch.save({'schema':run.SCHEMA,'arm':'D','model':m.state_dict(),'optimizer_parameter_names':names},p)
    with pytest.raises(SystemExit,match='optimizer name mapping mismatch'):
        run.load_final_checkpoint(p,torch.device('cpu'))


def test_recovery_save_preserves_rng_consumed_by_strict_reopen(tmp_path, monkeypatch):
    def payload(*args):
        rng=run.base.capture_rng()
        return {'rng_state':rng,'rng_digests':run.d5c.rng_digests(rng),
                'loader_state':{'test_cursor':144},'next_global_batch_sha256':'next','next_global_batch_stream_sha256':'stream'}
    def reopen(*args):
        torch.rand(1000)
        return {'passed':True,'checks':{'test_reconstruction':True}}
    monkeypatch.setattr(run,'checkpoint_payload',payload)
    monkeypatch.setattr(run,'strict_reopen',reopen)
    before=run.d5c.rng_digests(run.base.capture_rng())
    v=run.save_final(tmp_path/'recovery.pt',None,None,None,None,'D','ledger',torch.device('cpu'),144)
    assert run.d5c.rng_digests(run.base.capture_rng()) == before
    assert v['local_updates']==144 and v['global_update']==2625 and v['experiment_total_updates']==335
    with pytest.raises(SystemExit,match='refusing checkpoint overwrite'):
        run.save_final(tmp_path/'recovery.pt',None,None,None,None,'D','ledger',torch.device('cpu'),144)
