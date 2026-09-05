"""Genuine trained-state continuation and JSON-roundtrip flag rendering."""
import copy,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import test_experiment_2d10 as fixture
import experiment_2d10_h_250m as run
import experiment_2d10_analysis as analysis

def test_trained_router_state_and_next_update_preserved_without_initializer():
    model=fixture.model('H');x,y=fixture.inputs()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);fixture.run(model,x,y,checkpoint=True)[1].backward();optimizer.step()
    payload={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
        'parameter_names':list(dict(model.named_parameters())),'optimizer_parameter_names':run.optimizer_names(model,optimizer)}
    with patch.object(run.core,'Router',side_effect=AssertionError('fresh initialization forbidden')),patch.object(run.core.RetrievalGatingGPT,'enable_router',side_effect=AssertionError('fresh initialization forbidden')):
        other,opt=run.restore_h_payload(payload,torch.device(fixture.DEVICE),copy.deepcopy(model.base))
    assert run.state_equal(model.state_dict(),other.state_dict())
    assert run.state_equal(optimizer.state_dict(),opt.state_dict())
    for mode in ('parallel','incremental'):
        with torch.no_grad():
            a,_=fixture.run(model,x,y,mode);b,_=fixture.run(other,x,y,mode)
        assert torch.equal(a,b)
    for m,o in ((model,optimizer),(other,opt)):
        o.zero_grad(set_to_none=True);fixture.run(m,x,y,checkpoint=True)[1].backward();o.step()
    assert run.state_equal(model.state_dict(),other.state_dict())
    assert run.state_equal(optimizer.state_dict(),opt.state_dict())
    assert all(model.gate_parameter(b).grad is None for b in (0,2,4,5))

def test_flag_json_roundtrip_and_old_erratum():
    saved=json.loads((run.SEALED_RESULT/'PAIRED_BOOTSTRAP.json').read_text())
    for contrast,expected in [('D-T',[False,False,False,False,True,True]),('D-H',[True,True,False,False,False,True]),('T-H',[True,True,False,False,False,True])]:
        flags=json.loads(json.dumps(saved['contrasts'][contrast]['adjusted_flags'],sort_keys=True))
        assert analysis.render_flag_values(flags)==' | '.join(map(str,expected))
    assert analysis.FLAG_COLUMNS==('positive','beyond_margin','negative','material_harm','practical_equivalence','second_condition_noninferiority')

def test_budget_cadence_and_frozen_ledger():
    path=run.HISTORICAL_RESULT/'MATCHED_BATCH_LEDGER.jsonl'
    assert run.sha256(path)==run.LEDGER_SHA256
    rows=run.load_rows(path)
    assert len(rows)==run.LOCAL_UPDATES==286
    assert rows[0]['global_update']==2482 and rows[-1]['global_update']==2767
    assert sum(x['target_count'] for x in rows)==149946368
    assert [x['global_update'] for x in rows if x['pass_count']==3]==[2496,2528,2560,2592,2624,2656,2688,2720,2752]
    assert run.PRIOR_UPDATES+run.LOCAL_UPDATES==477
    assert run.PRIOR_TARGETS+run.LOCAL_TARGETS==250085376
