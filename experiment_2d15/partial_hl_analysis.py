"""Optional L-stage summaries; R remains missing and the family stays at 18."""
import argparse
import numpy as np
from .common import *
from .analysis import bootstrap
from .evaluate import collect


def write_report(archive, value):
    u=value['update']
    lines=[f'# Experiment 2D15: partial H/L comparison at update {u}', '',
           f"L_nf4 has completed {u:,} updates ({value['logical_targets']:,} logical targets). R_ON and R_ALL_OFF are pending. The main endpoint remains update 5000; this is an incomplete three-model comparison.", '',
           'All three conditions use the same fresh 4,096 sequences, true incremental BF16 forward, FP32 token CE, FP64 sums, and B128 groups on four GPUs. Raw batches were independently recollected on the Mac and checked against frozen panel, configuration, code, checkpoint and model identities.', '',
           '| Condition | CE | Perplexity |', '|---|---:|---:|']
    for c in ('L_LOCAL','H_ON','H_ALL_OFF'):
        v=value['means'][c];lines.append(f"| {c} | {v['ce']:.9f} | {v['ppl']:.6f} |")
    lines += ['', 'The sequence bootstrap uses 50,000 paired resamples with seed 20261001; the 64-group sensitivity uses 50,000 with seed 20261002. Pairing is preserved across available ages. Both retain the full 18-contrast family and adjusted 99.7222222222% intervals. Raw 95% intervals and separate classification flags are in the JSON.', '',
              '| Contrast | Mean CE difference | Sequence adjusted interval | Group adjusted interval |', '|---|---:|---|---|']
    for c,label in [('C','C = L_LOCAL − H_ON'),('G_H','G_H = H_ALL_OFF − H_ON')]:
        p=value['analysis']['primary'][c];g=value['analysis']['group_sensitivity'][c]
        lo,hi=p['adjusted_99_7222222222'];gl,gh=g['adjusted_99_7222222222']
        lines.append(f"| {label} | {p['mean']:+.9f} | [{lo:+.9f}, {hi:+.9f}] | [{gl:+.9f}, {gh:+.9f}] |")
    lines += ['', 'Negative C favors L quality; positive G_H favors H recurrence ON. R quality and the CE1-free joint goal remain unclassified until the full comparison is available. These are checkpoints from one trajectory per trained model, with intentional architecture and compute differences. No recipe adjustment or early stop follows these results.']
    atomic_bytes(archive/f'PARTIAL_HL_u{u:05d}.md', ('\n'.join(lines)+'\n').encode())


def run(archive):
    panel=read_json(FROZEN/'PANEL.json');ages=[];matrices=[];means={};inputs={}
    for u in MILESTONES:
        available=('H_ON','L_LOCAL','H_ALL_OFF')
        paths={c:archive/'L_nf4/evaluations'/f'{c}_u{u:05d}_COMPLETE.json' for c in available}
        if not all(p.exists() for p in paths.values()):continue
        matrix=np.full((4096,5),np.nan,dtype=np.float64);means[str(u)]={};inputs[str(u)]={}
        for c,path in paths.items():
            summary=read_json(path);b=summary['binding'];label=f'{c}_u{u:05d}'
            assert summary['passed'] and summary['checkpoint_unchanged'] and summary['tensors_unchanged']
            assert b['label']==label and b['arm']=='L_nf4' and b['milestone']==u
            assert b['panel_identity']==panel['identity'] and b['dataset_sha256']==VAL_SHA
            assert b['code_identity']==identity(read_json(FROZEN/'CODE_IDENTITY.json'))
            assert b['configuration_identity']==identity(read_json(FROZEN/'EXECUTION.json'))
            assert b['batch_size']==128 and b['world_size']==4 and b['partition']=='B128_group_index_modulo_4'
            assert b['mode']=='true_incremental_bf16_fp32_ce_fp64_nll'
            assert b['condition']==('H_ON' if c=='H_ON' else 'H_ALL_OFF')
            if c=='L_LOCAL':
                mf=read_json(archive/'L_nf4/checkpoints'/f'u{u:05d}.pt.manifest.json')
                expected_sha=mf['sha256'];model_id=mf['audit']['model_tensor_identity']
                assert read_json(archive/'VERIFIED_FILES.json')[f'L_nf4/u{u:05d}.pt']==expected_sha
            else:expected_sha,model_id=H_CHECKPOINTS[u]
            assert b['checkpoint_sha256']==expected_sha and b['model_tensor_identity']==model_id
            raw=collect(path.parent,label,panel,b,128)
            for key in ('rows','count','nll','ce','ppl'):assert raw[key]==summary[key]
            matrix[:,CONDITIONS.index(c)]=[r['nll']/1024 for r in raw['rows']]
            means[str(u)][c]=dict(ce=raw['ce'],ppl=raw['ppl'])
            inputs[str(u)][c]=dict(summary_sha256=sha256(path),binding=b,raw_batches_independently_recollected=True)
        matrices.append(matrix);ages.append(u)
    if not ages:return None
    matrix=np.stack(matrices,axis=1)
    # NaN explicitly marks unobserved R columns. Only the two R-independent
    # contrasts are retained; no R contrast or joint-goal flag is published.
    primary=bootstrap(matrix,20261001)
    groups=bootstrap(matrix.reshape(64,64,len(ages),5).mean(1),20261002)
    result=dict(updates=ages,complete_three_model_comparison=False,missing_conditions=['R_ON','R_ALL_OFF'],family_size=18,resamples=50000,sequence_seed=20261001,group_seed=20261002,means=means,inputs=inputs,ages={})
    for i,u in enumerate(ages):
        result['ages'][str(u)]={name:{k:part['ages'][i]['contrasts'][k] for k in ('C','G_H')} for name,part in [('primary',primary),('group_sensitivity',groups)]}
        value=dict(update=u,logical_targets=u*524288,means=means[str(u)],analysis=result['ages'][str(u)],family_size=18,resamples=50000,complete_three_model_comparison=False,missing_conditions=['R_ON','R_ALL_OFF'])
        atomic_json(archive/f'PARTIAL_HL_u{u:05d}.json',value)
        write_report(archive,value)
    atomic_json(archive/'PARTIAL_HL_ANALYSIS.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args();run(a.archive)
