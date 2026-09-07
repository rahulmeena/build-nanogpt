"""Independent Mac audit of a completed arm's entire frozen training prefix."""
import argparse
import hashlib
import json
import math
from .common import ARMS,FROZEN,Path,atomic_json,sha256


def run(archive,arm):
    assert arm in ARMS
    path=archive/arm/'metrics.jsonl';payload=path.read_bytes()
    rows=[json.loads(line) for line in payload.splitlines()]
    expected=[json.loads(line) for line in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    assert len(rows)==len(expected)==5000
    # Accumulate the protocol's actual work independently of the training
    # producer's closed-form accounting helper.
    totals=dict(completed_updates=0,logical_targets=0,backbone_pass_equivalent_updates=0,backbone_pass_targets=0,actual_ce_target_evaluations=0,nonzero_weight_ce_targets=0)
    if arm=='R_nf4':totals.update(two_pass_updates=0,three_pass_updates=0,diagnostic_only_ce1_targets=0,mean_backbone_passes=0)
    for u,(r,e) in enumerate(zip(rows,expected),1):
        assert r['arm']==arm and r['completed_updates']==u
        assert r['data']==e['data'] and r['lr']==e['lr'] and r['lr_index']==u-1
        assert r['fused'] is False and r['foreach'] is None
        assert r['world_size']==4 and r['microbatch']==32 and r['accumulation']==4
        passes=1 if arm=='L_nf4' else (3 if u%32==0 else 2)
        assert r['pass_count']==passes and e['data']['target_count']==524288
        totals['completed_updates']+=1;totals['logical_targets']+=524288
        totals['backbone_pass_equivalent_updates']+=passes
        totals['backbone_pass_targets']+=passes*524288
        totals['actual_ce_target_evaluations']+=passes*524288
        totals['nonzero_weight_ce_targets']+=(1 if arm=='L_nf4' else passes-1)*524288
        if arm=='R_nf4':
            totals['two_pass_updates']+=int(passes==2);totals['three_pass_updates']+=int(passes==3)
            totals['diagnostic_only_ce1_targets']+=524288
            totals['mean_backbone_passes']=totals['backbone_pass_equivalent_updates']/u
        for name,value in totals.items():assert r[name]==value
        assert len(r['pass_losses'])==r['pass_count']
        assert all(math.isfinite(x) for x in r['pass_losses'])
        assert math.isfinite(r['weighted_multipass_training_loss']) and math.isfinite(r['grad_norm'])
        assert r['training_seconds']>0 and math.isfinite(r['training_seconds'])
    seconds=math.fsum(r['training_seconds'] for r in rows)
    assert math.isclose(seconds,rows[-1]['cumulative_training_seconds'],rel_tol=1e-12,abs_tol=1e-9)
    digest=hashlib.sha256(payload).hexdigest();assert sha256(path)==digest
    result=dict(passed=True,arm=arm,updates=5000,all_updates_checked=True,metrics_sha256=digest,expected_batches_sha256=sha256(FROZEN/'expected_batches.jsonl'),exact_input_target_hashes_cursors_and_lr=True,optimizer_flags_match_H=True,four_rank_geometry=True,exact_pass_and_target_accounting=True,finite_losses_and_gradients=True,training_seconds=seconds,training_gpu_hours=4*seconds/3600,terminal_data=rows[-1]['data'],accounting=totals)
    atomic_json(archive/arm/'FULL_STREAM_AUDIT.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();print(json.dumps(run(a.archive,a.arm),indent=2))
