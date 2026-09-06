"""Frozen paired bootstrap; callable only locally after verified shutdown."""
import csv,math
import numpy as np
from .common import *

def flags(bounds):
    l,u=map(float,bounds)
    return dict(removal_hurts=l>0,removal_hurts_beyond_reference=l>.0001,
        removal_helps=u<0,removal_helps_beyond_reference=u<-.0001,
        practical_equivalence_at_reference=l>=-.0001 and u<=.0001)

def bootstrap(d,seed,resamples=50000,chunk=64):
    d=np.asarray(d,dtype=np.float64);rng=np.random.default_rng(seed)
    out=np.empty((resamples,d.shape[1]),np.float64)
    for start in range(0,resamples,chunk):
        n=min(chunk,resamples-start)
        indices=rng.integers(0,len(d),size=(n,len(d)))
        out[start:start+n]=d[indices].mean(axis=1,dtype=np.float64)
    ordinary=np.percentile(out,[2.5,97.5],axis=0,method='linear').T
    adjusted=np.percentile(out,[.625,99.375],axis=0,method='linear').T
    return [dict(ci95=a.tolist(),ci9875=b.tolist(),flags=flags(b)) for a,b in zip(ordinary,adjusted)]

def analyze(raw,output):
    from .artifacts import collect
    output=Path(output);raw=Path(raw)
    assert read_json(output/'STOP_VERIFICATION.json')['passed']
    assert read_json(output/'EXPORT_VERIFICATION.json')['passed']
    cfg=read_json(FROZEN/'ANALYSIS_CONFIG.json');assert np.__version__==cfg['numpy']
    panel=read_json(FROZEN/'PANEL.json');data=collect(raw,panel)
    ces=np.array([[r['ce'] for r in data[c]] for c in CONDITIONS],dtype=np.float64)
    differences=(ces[1:]-ces[0]).T
    seq=bootstrap(differences,20260922)
    group=bootstrap(differences.reshape(64,64,4).mean(axis=1),20260923)
    contrasts=[]
    for i,c in enumerate(list(CONDITIONS)[1:]):
        d=differences[:,i];mean=float(d.mean())
        contrasts.append(dict(condition=c,mean_delta_ce=mean,sequence=seq[i],canonical_group=group[i],
            sensitivity_flags_changed=[k for k in seq[i]['flags'] if seq[i]['flags'][k]!=group[i]['flags'][k]],
            relative_perplexity_percent=100*math.expm1(mean),
            relative_perplexity_ci95=[100*math.expm1(v) for v in seq[i]['ci95']],
            relative_perplexity_ci9875=[100*math.expm1(v) for v in seq[i]['ci9875']],
            positive_fraction=float((d>0).mean()),negative_fraction=float((d<0).mean()),zero_fraction=float((d==0).mean())))
    scores={c:dict(ce=sum(r['nll'] for r in data[c])/4194304,targets=4194304,sequences=4096) for c in CONDITIONS}
    for s in scores.values():s['perplexity']=math.exp(s['ce'])
    result=dict(scores=scores,contrasts=contrasts,settings=cfg,
        nonadditivity_all_minus_sum_individual=float(differences[:,3].mean()-differences[:,:3].sum(axis=1).mean()),
        panel_identity=panel['identity'])
    atomic_json(output/'PAIRED_ANALYSIS.json',result)
    with (output/'contrasts.csv').open('w',newline='') as f:
        cols=['condition','mean_delta_ce','ci95_low','ci95_high','ci9875_low','ci9875_high','group_ci9875_low','group_ci9875_high','relative_perplexity_percent']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in contrasts:w.writerow(dict(condition=r['condition'],mean_delta_ce=r['mean_delta_ce'],
            ci95_low=r['sequence']['ci95'][0],ci95_high=r['sequence']['ci95'][1],ci9875_low=r['sequence']['ci9875'][0],ci9875_high=r['sequence']['ci9875'][1],
            group_ci9875_low=r['canonical_group']['ci9875'][0],group_ci9875_high=r['canonical_group']['ci9875'][1],relative_perplexity_percent=r['relative_perplexity_percent']))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for i,r in enumerate(contrasts):
        m=r['mean_delta_ce'];l,u=r['sequence']['ci9875']
        ax.hlines(i,l,u,color='#176b87',lw=1.5)
        ax.plot([l,u],[i,i],linestyle='none',marker='|',markersize=9,color='#176b87')
        ax.plot(m,i,'o',color='#176b87')
    ax.axvline(0,color='black',lw=1)
    for v in [-.0001,.0001]:ax.axvline(v,color='#ba7540',ls='--',lw=1)
    ax.set_yticks(range(4),[r['condition'] for r in contrasts]);ax.invert_yaxis()
    ax.set_xlabel('OFF − H_ON cross entropy (nats / target)')
    ax.set_title('H10B recurrence removal · paired sequence bootstrap\n98.75% Bonferroni marginal intervals; reference ±0.0001')
    ax.grid(axis='x',alpha=.2);fig.savefig(output/'contrasts.png',dpi=180);fig.savefig(output/'contrasts.pdf');plt.close(fig)
    return result
