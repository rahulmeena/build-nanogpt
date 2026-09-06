"""Frozen six-contrast joint analysis, executed locally after verified shutdown."""
import math
import numpy as np
from .common import *
NAMES=('A','B','C','G_H','G_R','I')

def contrasts(values):
    h,l,r,ho,ro=np.asarray(values,dtype=np.float64).T
    return np.stack((h-r,l-r,l-h,ho-h,ro-r,(ro-r)-(ho-h)),axis=1)

def flags(lo,hi):
    delta=.0001
    return dict(positive=bool(lo>0),beyond_reference=bool(lo>delta),negative=bool(hi<0),
        material_negative=bool(hi<-delta),equivalence=bool(lo>=-delta and hi<=delta))

def bootstrap(values,seed,resamples=50000):
    d=contrasts(values);rng=np.random.default_rng(seed);draws=np.empty((resamples,6),dtype=np.float64)
    for start in range(0,resamples,128):
        n=min(128,resamples-start);indices=rng.integers(0,len(d),size=(n,len(d)))
        # This exact shared index vector samples all five conditions/six contrasts.
        draws[start:start+n]=d[indices].mean(axis=1,dtype=np.float64)
    raw=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    adjusted=np.percentile(draws,[2.5/6,100-2.5/6],axis=0,method='linear')
    result={}
    for i,name in enumerate(NAMES):
        lo,hi=adjusted[:,i];mean=float(d[:,i].mean(dtype=np.float64))
        result[name]=dict(mean=mean,raw_95=raw[:,i].tolist(),adjusted_99_1666666667=[float(lo),float(hi)],
            flags=flags(lo,hi),positive_fraction=float((d[:,i]>0).mean()),negative_fraction=float((d[:,i]<0).mean()),
            tie_fraction=float((d[:,i]==0).mean()))
    result['A']['R_noninferior_to_H']=bool(adjusted[0,0]>-.0001)
    joint=bool(adjusted[0,0]>-.0001 and adjusted[0,4]>.0001 and adjusted[0,5]>.0001)
    return dict(contrasts=result,joint_goal_met=joint,seed=seed,resamples=resamples,
        paired=True,family_size=6,percentile_method='linear',adjusted_percentile_bounds=[2.5/6,100-2.5/6])

def analyze(summaries,panel):
    values=[];totals={};bindings=[]
    for label in CONDITIONS:
        s=summaries[label];b=s['binding'];bindings.append(b)
        assert b['label']==label and b['panel_identity']==panel['identity'] and b['batch_size']==128
        assert b['mode']=='true_incremental_bf16_fp32_ce_fp64_nll' and len(s['rows'])==4096
        for r,ref in zip(s['rows'],panel['sequences']):
            assert (r['id'],r['input_start'],r['token_sha256'],r['canonical_group'])==(ref['id'],ref['input_start'],ref['token_sha256'],ref['canonical_group'])
            assert r['count']==1024 and r['ce']==r['nll']/1024 and math.isfinite(r['nll'])
        nll=math.fsum(r['nll'] for r in s['rows']);ce=nll/4194304
        totals[label]=dict(nll=nll,count=4194304,ce=ce,ppl=math.exp(ce))
        values.append([r['ce'] for r in s['rows']])
    assert len({b['code_identity'] for b in bindings})==len({b['configuration_identity'] for b in bindings})==1
    assert bindings[0]['checkpoint_sha256']==bindings[3]['checkpoint_sha256']==H_SHA
    assert bindings[2]['checkpoint_sha256']==bindings[4]['checkpoint_sha256']
    matrix=np.array(values,dtype=np.float64).T
    primary=bootstrap(matrix,20260928);group=bootstrap(matrix.reshape(64,64,5).mean(1),20260929)
    a=primary['contrasts']['A']['mean'];b=primary['contrasts']['B']['mean']
    return dict(endpoints=totals,primary=primary,group_sensitivity=group,
        changed_flags={n:primary['contrasts'][n]['flags']!=group['contrasts'][n]['flags'] for n in NAMES},
        relative_ppl=dict(H_to_R_reduction_percent=100*(1-math.exp(-a)),H_to_R_definition='100*(PPL_H_ON-PPL_R_ON)/PPL_H_ON',
                          L_to_R_reduction_percent=100*(1-math.exp(-b)),L_to_R_definition='100*(PPL_L_LOCAL-PPL_R_ON)/PPL_L_LOCAL'),
        rows=20480,targets=20971520,training_seed_uncertainty_estimated=False)

def rehearsal():
    rng=np.random.default_rng(7);v=rng.normal(size=(128,5));d=contrasts(v)
    assert np.allclose(d[:,0]+d[:,2],d[:,1])
    assert np.allclose(d[:,5],d[:,4]-d[:,3])
    shared=np.repeat(rng.normal(size=(128,1)),5,axis=1)
    zero=bootstrap(shared,20260928,resamples=100)
    assert all(r['mean']==0 and r['adjusted_99_1666666667']==[0,0] for r in zero['contrasts'].values())
    assert not zero['joint_goal_met']
    atomic_json(PACKAGE/'results/ANALYSIS_REHEARSAL.json',dict(passed=True,shared_pairing_zero_control=True,
        contrast_algebra_verified=True,resamples_in_final=50000,synthetic_only=True))

if __name__=='__main__':rehearsal()
