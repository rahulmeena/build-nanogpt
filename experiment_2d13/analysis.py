"""Prespecified paired sequence and canonical-group bootstrap, exclusively local."""
import math
import numpy as np
from .common import *

def flags(lo,hi,delta=.0001):
    return dict(H_benefit=bool(lo>0),H_benefit_beyond_reference=bool(lo>delta),L_benefit=bool(hi<0),L_benefit_beyond_reference=bool(hi<-delta),practical_equivalence_at_reference=bool(lo>=-delta and hi<=delta),L_noninferiority_at_reference=bool(hi<delta))

def bootstrap(d,seed,resamples=50000):
    d=np.asarray(d,dtype=np.float64);rng=np.random.default_rng(seed);out=np.empty(resamples,np.float64)
    for start in range(0,resamples,256):
        n=min(256,resamples-start);ix=rng.integers(0,len(d),size=(n,len(d)));out[start:start+n]=d[ix].mean(1,dtype=np.float64)
    lo,hi=np.percentile(out,[2.5,97.5],method='linear')
    return dict(mean=float(d.mean()),lo=float(lo),hi=float(hi),flags=flags(lo,hi),relative_ppl_percent=float(100*np.expm1(d.mean())),relative_ppl_interval_percent=[float(100*np.expm1(lo)),float(100*np.expm1(hi))],seed=seed,resamples=resamples)

def analyze(l,h,panel):
    assert l['binding']['panel_identity']==h['binding']['panel_identity']==panel['identity']
    assert l['binding']['batch_size']==h['binding']['batch_size']
    assert l['binding']['mode']==h['binding']['mode']=='true_incremental_bf16_fp32_ce_fp64_nll'
    lr=l['rows'];hr=h['rows'];assert len(lr)==len(hr)==4096
    for a,b,c in zip(lr,hr,panel['sequences']):
        assert a['id']==b['id']==c['id'] and a['token_sha256']==b['token_sha256']==c['token_sha256'] and a['input_start']==b['input_start']==c['input_start']
        assert a['count']==b['count']==1024
    d=np.array([a['ce']-b['ce'] for a,b in zip(lr,hr)],dtype=np.float64)
    seq=bootstrap(d,20260925);groups=d.reshape(64,64).mean(1);group=bootstrap(groups,20260926)
    return dict(L_CE=math.fsum(r['nll'] for r in lr)/4194304,H_CE=math.fsum(r['nll'] for r in hr)/4194304,L_PPL=math.exp(math.fsum(r['nll'] for r in lr)/4194304),H_PPL=math.exp(math.fsum(r['nll'] for r in hr)/4194304),primary=seq,group_sensitivity=group,sampling_unit_changes_flags=seq['flags']!=group['flags'],L_win_fraction=float((d<0).mean()),H_win_fraction=float((d>0).mean()),tie_fraction=float((d==0).mean()),training_seed_variation_estimated=False,paired_d=d.tolist())

def plots(out,analysis,lmon,hmon,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4.3))
    ax.plot([r['update']*524288/1e6 for r in lmon],[r['ce'] for r in lmon],'o-',label='L local-only',color='#147c74')
    ax.plot([r['update']*524288/1e6 for r in hmon],[r['ce'] for r in hmon],'s-',label='H original recipe',color='#c27530')
    ax.set(xlabel='Logical training targets (millions)',ylabel='Monitoring CE (nats / target)',title='2D13 · Same monitoring panel');ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'monitoring.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.8));ax.plot([r['completed_updates'] for r in metrics],[r['single_pass_training_loss'] for r in metrics],lw=.9,color='#147c74');ax.set(xlabel='Completed update',ylabel='Single-pass training CE',title='L training objective');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'L_training.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.2));s=analysis['primary'];g=analysis['group_sensitivity']
    for y,r,name,color in [(1,s,'Paired sequence (primary)','#147c74'),(0,g,'Canonical B64 group (sensitivity)','#687688')]:ax.errorbar(r['mean'],y,xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',capsize=5,color=color,label=name)
    ax.axvline(0,color='#444',lw=1);ax.axvspan(-.0001,.0001,alpha=.15,color='#c27530');ax.set(yticks=[0,1],yticklabels=['B64 groups','Sequences'],xlabel='CE(L524M) − CE(H524M); positive favors H',title='2D13 · Paired effect and 95% intervals');fig.tight_layout();fig.savefig(out/'paired_effect.png');plt.close(fig)
