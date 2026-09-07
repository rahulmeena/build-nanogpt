"""Three-age sequence/group pairing with a fixed 18-contrast family."""
import argparse,math
import numpy as np
from .common import *
NAMES=('A','B','C','G_H','G_R','I')

def contrasts(v):
    h,l,r,ho,ro=np.moveaxis(np.asarray(v,dtype=np.float64),-1,0)
    return np.stack((h-r,l-r,l-h,ho-h,ro-r,(ro-r)-(ho-h)),axis=-1)

def flags(lo,hi):return dict(positive=bool(lo>0),beyond_reference=bool(lo>.0001),negative=bool(hi<0),material_negative=bool(hi<-.0001),equivalence=bool(lo>=-.0001 and hi<=.0001))

def milestone_report(value):
    u=value['update'];primary=value['primary'];group=value['group_sensitivity']
    lines=[f'# Experiment 2D15: complete five-condition comparison at update {u}', '',
           f'Each trained trajectory has reached {u*524288:,} logical targets. Historical H is reused as the matched-age comparator. The prespecified main endpoint is update 5000; no recipe adjustment or early stop follows this result.', '',
           'All five conditions use the same fresh 4,096 validation sequences, true incremental BF16 forward, FP32 token CE and FP64 sums, with fixed B128 groups on four GPUs.', '',
           '| Condition | CE | Perplexity |', '|---|---:|---:|']
    for c in CONDITIONS:
        v=value['means'][c];lines.append(f"| {c} | {v['ce']:.9f} | {v['ppl']:.6f} |")
    lines += ['', 'Each bootstrap uses 50,000 paired resamples, preserving pairing across conditions and available ages. Sequence seed: 20261001. Canonical 64-group sensitivity seed: 20261002. The family remains all 18 contrasts, including at interim milestones; adjusted intervals are 99.7222222222%. The practical reference is delta_CE = 0.0001.', '',
              '| Contrast | Mean | Raw 95% interval | Adjusted interval | Separate adjusted-interval flags |', '|---|---:|---|---|---|']
    labels=dict(A='A = H_ON − R_ON',B='B = L_LOCAL − R_ON',C='C = L_LOCAL − H_ON',G_H='G_H = H_ALL_OFF − H_ON',G_R='G_R = R_ALL_OFF − R_ON',I='I = G_R − G_H')
    def interval(bounds):return f'[{bounds[0]:+.9f}, {bounds[1]:+.9f}]'
    def flag_text(v):
        names=dict(positive='positive',beyond_reference='above +delta',negative='negative',material_negative='below −delta',equivalence='practical equivalence')
        return '; '.join(names[k] for k,on in v['flags'].items() if on) or 'none'
    for name in NAMES:
        v=primary['contrasts'][name];lines.append(f"| {labels[name]} | {v['mean']:+.9f} | {interval(v['raw_95'])} | {interval(v['adjusted_99_7222222222'])} | {flag_text(v)} |")
    lines += ['', '| Group sensitivity contrast | Raw 95% interval | Adjusted interval | Separate flags |', '|---|---|---|---|']
    for name in NAMES:
        v=group['contrasts'][name];lines.append(f"| {name} | {interval(v['raw_95'])} | {interval(v['adjusted_99_7222222222'])} | {flag_text(v)} |")
    lines += ['', 'Positive A or B favors R quality; positive C favors H over L. Positive G values favor recurrence ON, and positive I means greater R OFF sensitivity. A flag of none does not establish practical equivalence.', '',
              f"CE1-free joint goal: **{'met' if primary['joint_goal_met'] else 'not met'}** under the primary sequence analysis; **{'met' if group['joint_goal_met'] else 'not met'}** in the group sensitivity. This requires the adjusted lower bounds A > −0.0001, G_R > +0.0001 and I > +0.0001 together.", '',
              'These checkpoints belong to one training trajectory per model. Intentional architecture, objective and pass-count differences remain. Larger OFF sensitivity alone does not establish better quality or isolated recurrent-content usefulness. No broad matched 10B/HellaSwag conclusion is made.']
    return '\n'.join(lines)+'\n'

def bootstrap(matrix,seed,resamples=50000):
    d=contrasts(matrix);rng=np.random.default_rng(seed);draws=np.empty((resamples,*d.shape[1:]),dtype=np.float64)
    for start in range(0,resamples,32):
        count=min(32,resamples-start);indices=rng.integers(0,len(d),size=(count,len(d)))
        draws[start:start+count]=d[indices].mean(axis=1,dtype=np.float64)
    raw=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    adjusted=np.percentile(draws,[2.5/18,100-2.5/18],axis=0,method='linear');results=[]
    for age in range(d.shape[1]):
        values={}
        for i,name in enumerate(NAMES):
            lo,hi=adjusted[:,age,i];values[name]=dict(mean=float(d[:,age,i].mean()),raw_95=raw[:,age,i].tolist(),adjusted_99_7222222222=[float(lo),float(hi)],flags=flags(lo,hi))
        results.append(dict(contrasts=values,joint_goal_met=bool(adjusted[0,age,0]>-.0001 and adjusted[0,age,4]>.0001 and adjusted[0,age,5]>.0001)))
    return dict(ages=results,resamples=resamples,seed=seed,family_size=18,adjusted_percentile_bounds=[2.5/18,100-2.5/18],paired_across_conditions_and_ages=True)

def run(archive):
    from .evaluate import collect
    panel=read_json(FROZEN/'PANEL.json');ages=[];matrices=[];means={}
    for u in MILESTONES:
        files=[archive/('L_nf4' if c in ('H_ON','H_ALL_OFF','L_LOCAL') else 'R_nf4')/'evaluations'/f'{c}_u{u:05d}_COMPLETE.json' for c in CONDITIONS]
        if not all(f.exists() for f in files):continue
        summaries=[read_json(f) for f in files];columns=[];bindings=[]
        for c,s in zip(CONDITIONS,summaries):
            b=s['binding'];bindings.append(b);assert b['panel_identity']==panel['identity'] and b['milestone']==u and b['batch_size']==128 and b['world_size']==4
            owner='L_nf4' if c in ('H_ON','H_ALL_OFF','L_LOCAL') else 'R_nf4'
            label=f'{c}_u{u:05d}'
            assert b['label']==label and b['arm']==owner and b['dataset_sha256']==VAL_SHA
            assert b['code_identity']==identity(read_json(FROZEN/'CODE_IDENTITY.json'))
            assert b['configuration_identity']==identity(read_json(FROZEN/'EXECUTION.json'))
            assert b['partition']=='B128_group_index_modulo_4' and b['mode']=='true_incremental_bf16_fp32_ce_fp64_nll'
            assert b['condition']==('H_ALL_OFF' if c=='L_LOCAL' or c.endswith('_ALL_OFF') else 'H_ON')
            if c.startswith('H_'):expected_sha,model_id=H_CHECKPOINTS[u]
            else:
                manifest=read_json(archive/owner/'checkpoints'/f'u{u:05d}.pt.manifest.json')
                expected_sha=manifest['sha256'];model_id=manifest['audit']['model_tensor_identity']
                assert read_json(archive/'VERIFIED_FILES.json')[f'{owner}/u{u:05d}.pt']==expected_sha
            assert b['checkpoint_sha256']==expected_sha and b['model_tensor_identity']==model_id
            assert len(s['rows'])==4096 and s['passed'] and s['checkpoint_unchanged'] and s['tensors_unchanged']
            raw=collect(archive/owner/'evaluations',label,panel,b,128)
            # Audit the underlying NLL exactly; descriptive exp(CE) is recomputed
            # locally and need not share Linux/macOS libm's last bit.
            for key in ('rows','count','nll','ce'):assert raw[key]==s[key]
            for row,ref in zip(s['rows'],panel['sequences']):
                assert row['id']==ref['id'] and row['token_sha256']==ref['token_sha256'] and row['input_start']==ref['input_start'] and row['count']==1024
            columns.append([r['nll']/1024 for r in s['rows']])
        assert bindings[0]['checkpoint_sha256']==bindings[3]['checkpoint_sha256']==H_CHECKPOINTS[u][0]
        assert bindings[2]['checkpoint_sha256']==bindings[4]['checkpoint_sha256']
        assert len({b['code_identity'] for b in bindings})==len({b['configuration_identity'] for b in bindings})==1
        matrix=np.array(columns,dtype=np.float64).T;matrices.append(matrix);ages.append(u)
        means[str(u)]={c:dict(ce=float(matrix[:,i].mean()),ppl=math.exp(float(matrix[:,i].mean()))) for i,c in enumerate(CONDITIONS)}
    if not ages:return None
    matrix=np.stack(matrices,axis=1);primary=bootstrap(matrix,20261001);groups=bootstrap(matrix.reshape(64,64,len(ages),5).mean(1),20261002)
    result=dict(updates=ages,logical_targets=[u*524288 for u in ages],endpoint_means=means,primary=primary,group_sensitivity=groups,main_endpoint=5000,complete=ages==list(MILESTONES),training_replications=1)
    atomic_json(archive/'ANALYSIS.json',result)
    for i,u in enumerate(ages):
        value=dict(update=u,means=means[str(u)],primary=primary['ages'][i],group_sensitivity=groups['ages'][i],family_size=18)
        atomic_json(archive/f'INTERIM_u{u:05d}.json',value)
        atomic_bytes(archive/f'INTERIM_u{u:05d}.md',milestone_report(value).encode())
    return result

def rehearsal():
    rng=np.random.default_rng(1);v=rng.normal(size=(64,3,5));d=contrasts(v)
    assert np.allclose(d[:,:,0]+d[:,:,2],d[:,:,1]) and np.allclose(d[:,:,5],d[:,:,4]-d[:,:,3])
    shared=np.repeat(rng.normal(size=(64,3,1)),5,axis=-1);r=bootstrap(shared,20261001,100)
    assert all(x['mean']==0 and x['adjusted_99_7222222222']==[0,0] for age in r['ages'] for x in age['contrasts'].values())
    atomic_json(PACKAGE/'results/ANALYSIS_REHEARSAL.json',dict(passed=True,contrast_algebra=True,across_age_pairing=True,family_size=18))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);a=p.parse_args();run(a.archive) if a.archive else rehearsal()
