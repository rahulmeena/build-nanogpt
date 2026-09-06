"""Three-age sequence/group pairing with a fixed 18-contrast family."""
import argparse,math
import numpy as np
from .common import *
NAMES=('A','B','C','G_H','G_R','I')

def contrasts(v):
    h,l,r,ho,ro=np.moveaxis(np.asarray(v,dtype=np.float64),-1,0)
    return np.stack((h-r,l-r,l-h,ho-h,ro-r,(ro-r)-(ho-h)),axis=-1)

def flags(lo,hi):return dict(positive=bool(lo>0),beyond_reference=bool(lo>.0001),negative=bool(hi<0),material_negative=bool(hi<-.0001),equivalence=bool(lo>=-.0001 and hi<=.0001))

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
    panel=read_json(FROZEN/'PANEL.json');ages=[];matrices=[];means={}
    for u in MILESTONES:
        files=[archive/('L_nf4' if c in ('H_ON','H_ALL_OFF','L_LOCAL') else 'R_nf4')/'evaluations'/f'{c}_u{u:05d}_COMPLETE.json' for c in CONDITIONS]
        if not all(f.exists() for f in files):continue
        summaries=[read_json(f) for f in files];columns=[];bindings=[]
        for c,s in zip(CONDITIONS,summaries):
            b=s['binding'];bindings.append(b);assert b['panel_identity']==panel['identity'] and b['milestone']==u and b['batch_size']==128 and b['world_size']==4
            assert len(s['rows'])==4096 and s['passed']
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
    for i,u in enumerate(ages):atomic_json(archive/f'INTERIM_u{u:05d}.json',dict(update=u,means=means[str(u)],primary=primary['ages'][i],group_sensitivity=groups['ages'][i],family_size=18))
    return result

def rehearsal():
    rng=np.random.default_rng(1);v=rng.normal(size=(64,3,5));d=contrasts(v)
    assert np.allclose(d[:,:,0]+d[:,:,2],d[:,:,1]) and np.allclose(d[:,:,5],d[:,:,4]-d[:,:,3])
    shared=np.repeat(rng.normal(size=(64,3,1)),5,axis=-1);r=bootstrap(shared,20261001,100)
    assert all(x['mean']==0 and x['adjusted_99_7222222222']==[0,0] for age in r['ages'] for x in age['contrasts'].values())
    atomic_json(PACKAGE/'results/ANALYSIS_REHEARSAL.json',dict(passed=True,contrast_algebra=True,across_age_pairing=True,family_size=18))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);a=p.parse_args();run(a.archive) if a.archive else rehearsal()
