"""Local-only paired inference, scientific plots and explicitly qualified reports."""
import argparse
import csv
import io
import math
from pathlib import Path
import numpy as np
from .common import *


def flag(value):
    if value is True:return 'PASS'
    if value is False:return 'FAIL'
    if value is None:return 'UNAVAILABLE'
    raise TypeError('audit flags must be explicit booleans or null')


def bootstrap(contrasts,seed,resamples=50000):
    values=np.asarray(contrasts,dtype=np.float64)
    assert values.ndim==1 and len(values)>0 and np.isfinite(values).all()
    generator=np.random.default_rng(seed);means=np.empty(resamples)
    for start in range(0,resamples,128):
        n=min(128,resamples-start)
        means[start:start+n]=values[generator.integers(0,len(values),size=(n,len(values)))].mean(1)
    ci=np.percentile(means,[2.5,97.5,1.25,98.75],method='linear').tolist()
    classification='benefit' if ci[2]>0 else 'harm' if ci[3]<0 else 'unresolved'
    return dict(estimate=float(values.mean()),ci95=ci[:2],ci97_5=ci[2:],classification=classification,
                unit_count=len(values),resamples=resamples,seed=seed,numpy_version=np.__version__,
                percentile_method='linear',positive_favors='H')


def csv_file(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    atomic_bytes(path,buffer.getvalue().encode())


def collect(run):
    metrics={}
    for p in sorted(run.glob('metrics-*.jsonl')):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            metrics[r['completed_updates']]=r
    evaluations={}
    for p in sorted((run/'evaluations').rglob('*.json')):
        if '-rank' in p.name:continue
        e=read_json(p)
        if 'key' in e:evaluations[e['key']]=e
    historical=list(csv.DictReader((FROZEN/'baseline_learning_curve.csv').open()))
    return sorted(metrics.values(),key=lambda r:r['completed_updates']),evaluations,historical


def figure_save(fig,base):
    fig.tight_layout();fig.savefig(str(base)+'.png',dpi=180);fig.savefig(str(base)+'.pdf')
    import matplotlib.pyplot as plt
    plt.close(fig)


def interim_plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=run/'analysis';out.mkdir(exist_ok=True)
    metrics,evaluations,historical=collect(run)
    rows=[]
    for r in historical:
        rows.append(dict(model='G historical, 1 A100 80GB',updates=int(r['step']),logical_targets=int(r['step'])*TARGETS,
                         ce=float(r['val_loss']),perplexity=math.exp(float(r['val_loss'])),
                         hellaswag=float(r['hellaswag_accuracy'])))
    for u in CE_SCHEDULE:
        e=evaluations.get(f'ce:{u}')
        if e:
            rows.append(dict(model='H fresh, 4 A100 80GB',updates=u,logical_targets=u*TARGETS,
                ce=e['score'],perplexity=math.exp(e['score']),
                hellaswag=evaluations.get(f'hella:{u}',{}).get('score')))
    csv_file(out/'quality_curves.csv',rows);atomic_json(out/'quality_curves.json',rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for model in dict.fromkeys(r['model'] for r in rows):
        subset=[r for r in rows if r['model']==model]
        for ax,key in zip(axes,['ce','perplexity','hellaswag']):
            selected=[r for r in subset if r[key] is not None]
            ax.plot([r['logical_targets']/1e9 for r in selected],[r[key] for r in selected],
                    'o-' if model.startswith('H') else '-',label=model,markersize=3)
            ax.set_xlabel('Processed logical targets (billions)');ax.set_ylabel(key)
            ax.grid(alpha=.2)
        axes[1].set_yscale('log')
    axes[0].legend(fontsize=7);axes[2].set_title('H sampling is sparse; absent points omitted',fontsize=9)
    figure_save(fig,out/'quality_curves')
    if not metrics:return
    csv_file(out/'h_training_metrics.csv',[{k:r[k] for k in ('completed_updates','logical_targets',
            'ce_pass_targets','weighted_multipass_training_loss','training_seconds','cumulative_training_seconds',
            'targets_per_second','peak_allocated_bytes','timestamp')} for r in metrics])
    hist=read_json(FROZEN/'baseline_metrics.json');gtrain=[r for r in hist if r['kind']=='train']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['tokens']/1e9 for r in gtrain],[r['train_loss'] for r in gtrain],lw=.5)
    axes[0].set_title('G: single-pass CE objective (historical)')
    axes[1].plot([r['logical_targets']/1e9 for r in metrics],[r['weighted_multipass_training_loss'] for r in metrics],lw=.5)
    axes[1].set_title('H: weighted multipass objective')
    for ax in axes:ax.set_xlabel('Processed logical targets (billions)');ax.set_ylabel('Training objective');ax.grid(alpha=.2)
    figure_save(fig,out/'distinct_training_objectives')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['logical_targets']/1e9 for r in metrics],[r['targets_per_second'] for r in metrics],lw=.5)
    axes[0].set_ylabel('H logical targets / second, 4 GPUs')
    axes[1].plot([r['logical_targets']/1e9 for r in metrics],[r['peak_allocated_bytes']/1e9 for r in metrics])
    axes[1].set_ylabel('Rank 0 peak training allocation (GB)')
    for ax in axes:ax.set_xlabel('Processed logical targets (billions)');ax.grid(alpha=.2)
    figure_save(fig,out/'training_resources')
    # Historical timings establish training-only coordinates, not billed-wall measurements.
    gtime={0:0.};cumulative=0.
    for r in gtrain:
        cumulative+=(r['step_time_ms'] or 0)/1000;gtime[r['step']+1]=cumulative
    binding=read_json(run/'BINDING.json') if (run/'BINDING.json').exists() else {}
    htime={r['completed_updates']:r['cumulative_training_seconds'] for r in metrics};htime[0]=0.
    points=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for model,table,gpus in [('G historical',gtime,1),('H fresh',htime,4)]:
        selected=[r for r in rows if r['model'].startswith(model[0]) and r['updates'] in table]
        for ax,multiplier in zip(axes[:2],[1,gpus]):
            ax.plot([table[r['updates']]*multiplier/3600 for r in selected],[r['ce'] for r in selected],'.-',label=model)
        for r in selected:points.append(dict(model=model,updates=r['updates'],ce=r['ce'],training_hours=table[r['updates']]/3600,
                                            training_gpu_hours=table[r['updates']]*gpus/3600,gpu_count=gpus))
    axes[0].set_xlabel('Measured cumulative training hours (evaluations excluded)')
    axes[1].set_xlabel('Measured cumulative training GPU-hours')
    h=[e for k,e in evaluations.items() if k.startswith('ce:')]
    if binding and h:
        axes[2].plot([(e['completed_at']-binding['billing_start']+binding['prior_billed_seconds'])/3600 for e in h],
                     [e['score'] for e in h],'.-',label='H, evaluation-inclusive billed elapsed')
    axes[2].set_xlabel('Cumulative billed H pod-hours, all attempts')
    for ax in axes:ax.set_ylabel('Monitoring CE');ax.grid(alpha=.2);ax.legend(fontsize=7)
    figure_save(fig,out/'quality_vs_compute');csv_file(out/'quality_vs_compute.csv',points)


def paired_endpoints(evaluations):
    h=evaluations['final:H:ce']['rows'];g=evaluations['final:G:ce']['rows']
    assert [r['id'] for r in h]==[r['id'] for r in g] and len(h)==4096
    differences=np.array([gg['ce']-hh['ce'] for gg,hh in zip(g,h)])
    ce=bootstrap(differences,20260918)
    ce.update(g_ce=float(np.mean([r['ce'] for r in g])),h_ce=float(np.mean([r['ce'] for r in h])),
              relative_perplexity_change_h_over_g=math.exp(-ce['estimate'])-1,
              h_sequence_wins=int((differences>0).sum()),ties=int((differences==0).sum()),
              historical_small_effect_reference=.0001,adjusted_lower_exceeds_reference=ce['ci97_5'][0]>.0001)
    hh=evaluations['hella:19072']['rows'];gg=evaluations['final:G:hella']['rows']
    assert [r['id'] for r in hh]==[r['id'] for r in gg] and len(hh)==10042
    contrast=np.array([float(h['correct'])-float(g['correct']) for h,g in zip(hh,gg)])
    hs=bootstrap(contrast,20260919)
    table={f'G{int(a)}_H{int(b)}':sum(g['correct']==a and h['correct']==b for g,h in zip(gg,hh)) for a in (False,True) for b in (False,True)}
    hs.update(percentage_points=hs['estimate']*100,paired_correctness=table,
              g_accuracy=sum(r['correct'] for r in gg)/10042,h_accuracy=sum(r['correct'] for r in hh)/10042)
    return dict(ce=ce,hellaswag=hs,joint_claim_intervals='Bonferroni 97.5% marginal',seed_replication=False)


def analyze(run):
    assert np.__version__=='2.5.2', 'frozen bootstrap NumPy version required'
    run=Path(run);out=run/'analysis';out.mkdir(exist_ok=True)
    stop=read_json(run/'STOP_VERIFICATION.json');assert stop['passed'], 'analysis requires verified pod stop'
    metrics,evals,historical=collect(run)
    outcome=read_json(run/'SCIENTIFIC_OUTCOME.json') if (run/'SCIENTIFIC_OUTCOME.json').exists() else dict(outcome='INVALID_INCOMPLETE',**accounting(0))
    pf=read_json(run/'PREFLIGHT.json') if (run/'PREFLIGHT.json').exists() else None
    if pf and not pf['passed']:outcome=dict(outcome='BUDGET_STOPPED',**accounting(0))
    exports=read_json(run/'FINAL_EXPORTS_VERIFIED.json') if (run/'FINAL_EXPORTS_VERIFIED.json').exists() else {}
    billing=read_json(run/'BILLING_ACCOUNTING.json')
    expected=outcome['completed_updates']
    replay=[r['completed_updates'] for r in metrics]==list(range(1,expected+1))
    checks=dict(provider_stop_verified=stop['passed'],independent_exports_verified=exports.get('passed',False),
                cumulative_budget_respected=billing['cumulative_billed_seconds']<=86400,
                scientific_update_coverage=replay,terminal_cursor=outcome.get('terminal_plan_matches'),
                cuda_preflight_passed=pf['passed'] if pf else False,completed_10b=outcome['outcome']=='COMPLETED')
    atomic_json(out/'AUDIT.json',checks);checks=read_json(out/'AUDIT.json')
    interim_plot(run)
    paired=paired_endpoints(evals) if checks['completed_10b'] else None
    if paired:atomic_json(out/'PAIRED_ENDPOINTS.json',paired)
    for key,e in evals.items():csv_file(out/(key.replace(':','_')+'.csv'),e['rows'])
    resources=outcome.get('resources',{});training=sum(r['training_seconds'] for r in metrics)
    evaluation=sum(e['elapsed_seconds'] for e in evals.values())
    export_files=list((run/'checkpoints').glob('*.verification.json'))
    transfers=sum(read_json(p)['transfer_verify_seconds'] for p in export_files)
    resource=dict(**billing,training_seconds=training,evaluation_seconds=evaluation,
                  overlapped_checkpoint_transfer_seconds=transfers,
                  preflight_seconds=pf['seconds'] if pf else None,
                  residual_setup_save_failure_idle_seconds=max(0,billing['cumulative_billed_seconds']-training-evaluation-(pf['seconds'] if pf else 0)),
                  residual_categories_not_separately_measured=True,
                  cumulative_gpu_hours=4*billing['cumulative_billed_seconds']/3600,
                  quoted_pod_dollars_per_hour=6.36,estimated_compute_dollars=6.36*billing['cumulative_billed_seconds']/3600,
                  storage_cost_excluded=True,h_state_bytes_per_sequence=33289728,g_state_bytes_per_sequence=37711872,
                  h_registered_parameters=124697386,h_active_parameters=124697382,g_registered_parameters=124475904)
    atomic_json(out/'RESOURCE_ACCOUNTING.json',resource)
    historical_repro=[]
    for p in sorted((run/'preflight').glob('HISTORICAL_G_MONITOR_RANK*.json')):historical_repro+=read_json(p)['losses']
    reproduced=None
    if len(historical_repro)==20:
        assert sorted(r['logical_batch'] for r in historical_repro)==list(range(20))
        reproduced=pf.get('saved_g_historical_monitor_reproduction',{}).get('score')
    if pf:
        import matplotlib.pyplot as plt
        benchmark=[read_json(p) for p in sorted((run/'preflight').glob('BENCHMARK_RANK*.json'))]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        m=pf['measurements'];axes[0].bar(['G 1-pass','H 2-pass','H 3-pass'],[m['g_update_seconds'],m['two_pass_seconds'],m['three_pass_seconds']]);axes[0].set_ylabel('Measured seconds / global update')
        if benchmark:
            axes[1].bar(['G','H'],[min(r['inference'][a]['tokens_per_second'] for r in benchmark) for a in ['G','H']]);axes[1].set_ylabel('Incremental tokens/s; B4, prime768, time256')
        axes[2].bar(['G','H'],[37711872/1e6,33289728/1e6]);axes[2].set_ylabel('Persistent BF16 state MB / sequence')
        fig.suptitle('Same four A100 80GB hardware; disposable preflight weights')
        figure_save(fig,out/'same_hardware_benchmark')
    text=[f"# Experiment 2D11: {outcome['outcome']}",
          f"Completed {expected:,} updates, {outcome['logical_targets']:,} logical targets and {outcome['ce_pass_targets']:,} H CE pass-target evaluations. The mean CE pass count is {outcome['mean_ce_passes_per_logical_target']}.",
          f"Cumulative billed pod time: {billing['cumulative_billed_seconds']/3600:.4f} hours; aggregate GPU time {resource['cumulative_gpu_hours']:.4f} hours. Quoted total pod price $6.36/hour; estimated compute ${resource['estimated_compute_dollars']:.2f}, excluding storage.",
          '\n'.join(f'- {k}: **{flag(v)}**' for k,v in checks.items()),
          INTERPRETATION,
          'The frozen fresh base uses the original seed 1337/source initialization order and equals the mapped H base exactly. macOS ARM and Linux x86 seeded tensor generation differed in the disposable probe. Scientific training loads the immutable locally frozen tensors on every rank. No historical step-zero checkpoint is available, so historical bitwise initialization equality is not claimed.',
          'These are single training trajectories comparing complete architecture-plus-training recipes. H uses additional multipass training compute. Four-GPU wall time versus historical one-GPU G does not establish architecture speedup.',
          'Prespecified rules: at u=1000 stop if H CE minus historical G CE >=0.50 (skip HellaSwag in that case). At u=2000 stop if g500>=0.15 AND g1>=0.15 AND g1>=0.75*g500 AND HellaSwag advantage<=0.005. These compute-budget heuristics are not hypothesis tests.',
          f"Saved G original parallel B64 monitoring evaluator reproduction: {reproduced}. Historical parallel and new incremental execution can differ numerically; intermediate historical aggregates have no paired confidence intervals."]
    for title in ['500M','1B']:
        p=run/f'INTERIM_{title}.json'
        if p.exists():text.append(f"Approximately {title} review:\n\n```json\n{json.dumps(read_json(p),indent=2)}\n```")
    if pf:text.append('Measured preflight and full-work projection:\n\n```json\n'+json.dumps(pf['projection'],indent=2)+'\n```')
    if paired:
        text.append('Final paired contrasts, positive favors H:\n\n```json\n'+json.dumps(paired,indent=2)+'\n```')
        text.append('Adjusted 97.5% marginal intervals govern endpoint classifications; 95% intervals are descriptive. The CE margin 0.0001 is a small-effect reference, not a threshold for worthwhile compute. An unresolved or harmful endpoint prevents a blanket overall-superiority claim.')
    else:text.append('No completed matched 10B comparison is available. No final paired intervals or H-prefix versus G10B superiority claim is made.')
    text.append('See CSV/JSON and PNG/PDF artifacts in this directory. Full checkpoint files remain outside Git; manifests identify the local archive and independent persistent-volume copies. Transfer worker duration overlaps useful GPU work and is not additive billed time. Residual setup/save/failure/idle time is labeled when it cannot be separated reliably.')
    atomic_bytes(out/'FINAL_REPORT.md',('\n\n'.join(text)+'\n').encode())
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyze(a.run)
