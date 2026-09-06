import copy,json,sys,time
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment_2d12.common import *
from experiment_2d12.analysis import bootstrap,flags
from experiment_2d12.artifacts import validate_batch,existing,collect
from experiment_2d12 import provider

def fixture_batch(condition,selected,binding):
    return dict(condition=condition,binding=binding,disabled_destinations=list(CONDITIONS[condition]),cache_audit={'passed':True},
        rows=[dict(id=r['id'],input_start=r['input_start'],token_sha256=r['token_sha256'],count=1024,nll=3072.,ce=3.) for r in selected])

def test_atomic_resume_coverage_alignment_and_corruption(tmp_path):
    panel=read_json(FROZEN/'PANEL.json');b=128
    binding={'panel_identity':panel['identity'],'checkpoint_sha256':CHECKPOINT_SHA,'code_commit':'fixture'}
    atomic_json(tmp_path/'RUN_CONFIG.json',dict(batch_size=b,binding=binding))
    for c in CONDITIONS:
        for start in range(0,4096,b):
            selected=panel['sequences'][start:start+b];p=tmp_path/c/f'batch_{start:04d}.json'
            assert existing(p,c,selected,binding) is None
            v=fixture_batch(c,selected,binding);atomic_json(p,v)
            assert existing(p,c,selected,binding)==v
    data=collect(tmp_path,panel);assert sum(len(v) for v in data.values())==20480
    p=tmp_path/'H_B3_OFF/batch_0000.json';value=read_json(p);value['rows'][0]['id']=1;atomic_json(p,value)
    with pytest.raises(AssertionError):collect(tmp_path,panel)
    value=fixture_batch('H_B3_OFF',panel['sequences'][:b],binding);value['binding']=dict(binding,code_commit='stale');atomic_json(p,value)
    with pytest.raises(AssertionError):collect(tmp_path,panel)
    atomic_bytes(p,b'{incomplete');
    with pytest.raises(ValueError):collect(tmp_path,panel)
    assert not list(tmp_path.rglob('*.tmp-*'))

def test_bootstrap_shared_indices_linear_percentiles_and_sign():
    d=np.column_stack([np.arange(5)*.001,np.arange(5)*-.002,np.ones(5)*.00005,np.zeros(5)])
    seed=20260922;n=300
    rng=np.random.default_rng(seed);out=np.array([d[rng.integers(0,5,5)].mean(0) for _ in range(n)])
    actual=bootstrap(d,seed,n,chunk=13)
    for i,row in enumerate(actual):
        assert np.allclose(row['ci95'],np.percentile(out[:,i],[2.5,97.5],method='linear'),rtol=0,atol=0)
        assert np.allclose(row['ci9875'],np.percentile(out[:,i],[.625,99.375],method='linear'),rtol=0,atol=0)
    assert actual[0]['flags']['removal_hurts'] and actual[1]['flags']['removal_helps']
    assert actual[2]['flags']['removal_hurts'] and actual[2]['flags']['practical_equivalence_at_reference']
    assert actual[3]['flags']['practical_equivalence_at_reference'] and not actual[3]['flags']['removal_hurts']
    assert not flags([-.001,.001])['practical_equivalence_at_reference']
    assert not flags([.0001,.001])['removal_hurts_beyond_reference']
    assert not flags([-.001,-.0001])['removal_helps_beyond_reference']
    # Identical 64-row group expansion reduces back to the original group means.
    expanded=np.repeat(d,64,axis=0).reshape(5,64,4).mean(1)
    np.testing.assert_allclose(expanded,d,atol=1e-16)

def test_stop_deadline_stalls_and_retry(monkeypatch,tmp_path):
    b={'hard_deadline':1000,'billing_start':0}
    assert provider.stop_reason(700,b,{'time':700,'last_progress':700})
    assert provider.stop_reason(190,b,None)
    assert provider.stop_reason(150,b,{'time':0,'last_progress':150})
    assert provider.stop_reason(650,b,{'time':650,'last_progress':0,'progress_timeout':600})
    assert provider.stop_reason(650,b,{'time':650,'last_progress':649}) is None
    responses=iter([{'desiredStatus':'RUNNING','runtimeStatus':'running'}, {'desiredStatus':'EXITED','runtimeStatus':'stopped'}])
    calls=[];monkeypatch.setattr(provider,'status',lambda:next(responses));monkeypatch.setattr(provider,'call',lambda a:calls.append(a));monkeypatch.setattr(provider.time,'sleep',lambda _:None)
    result=provider.stop_verified(tmp_path/'stop.json');assert result['passed'];assert calls==[['pod','stop',provider.POD,'-o','json']]

def test_panel_target_spans_tokens_and_exclusion_union():
    from experiment_2d11.data import array_hash
    p=read_json(FROZEN/'PANEL.json');a=read_json(FROZEN/'PANEL_DISJOINTNESS.json')
    source=read_json(FROZEN/'SOURCE_IDENTITIES.json');arr=np.load(source['validation_path'],mmap_mode='r')
    assert p['identity']==identity(p['sequences']) and len(p['sequences'])==4096
    for r in p['sequences']:
        s=r['input_start'];lo,hi=r['target_span_half_open']
        assert lo==s+1 and hi==s+1025 and array_hash(arr[s:s+1024],arr[s+1:s+1025])==r['token_sha256']
        assert not any(max(lo,c)<min(hi,d) for c,d in a['merged_union'])
    valid=[]
    for i in np.random.default_rng(20260921).permutation(a['total_complete_batches']):
        if not any(max(int(i)*65536+1,c)<min((int(i)+1)*65536+1,d) for c,d in a['merged_union']):valid.append(int(i))
    assert p['batch_indices_in_evaluation_order']==valid[:64]

def test_export_independent_hashes(tmp_path):
    from experiment_2d12.controller import export
    import shutil,hashlib
    remote_root=tmp_path/'persistent';remote_root.mkdir();atomic_json(remote_root/'batch.json',{'nll':3.})
    class RemoteFixture:
        def run(self,args,timeout):
            return json.dumps({p.name:sha256(p) for p in remote_root.glob('*.json')}).encode()
        def download(self,source,dest,timeout):
            Path(dest).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    local=tmp_path/'local';m=export(RemoteFixture(),str(remote_root),local)
    assert sha256(local/'raw/batch.json')==m['batch.json']==sha256(remote_root/'batch.json')

def test_no_training_entrypoints_in_new_scientific_modules():
    import ast
    for name in ('model.py','evaluate.py','verification.py'):
        tree=ast.parse((PACKAGE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in ('backward','step','zero_grad')
