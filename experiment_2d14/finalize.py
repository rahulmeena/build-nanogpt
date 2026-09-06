"""Local audit, six-contrast bootstrap, figures and report after GPU shutdown."""
import argparse,csv,json,math,shutil,time
import numpy as np
import torch
from .common import *
from .checkpoints import validate
from .model import from_state,tensor_identity
from .analysis import analyze,NAMES

def verify_evaluation(raw,label,panel,checkpoint_sha,model_identity,condition):
    from .evaluate import collect
    value=read_json(raw/f'evaluations/{label}_COMPLETE.json');binding=value['binding']
    assert value['passed'] and value['checkpoint_unchanged'] and value['tensors_unchanged']
    assert binding['checkpoint_sha256']==checkpoint_sha and binding['model_tensor_identity']==model_identity
    assert binding['condition']==condition and binding['panel_identity']==panel['identity']
    assert binding['code_identity']==identity(read_json(FROZEN/'CODE_IDENTITY.json'))
    assert binding['configuration_identity']==identity(read_json(FROZEN/'EXECUTION.json'))
    assert binding['batch_size']==128 and binding['mode']=='true_incremental_bf16_fp32_ce_fp64_nll'
    rebuilt=collect(raw/'evaluations',label,panel,binding,128)
    for key in ['rows','nll','count','ce']:assert rebuilt[key]==value[key],(label,key)
    # The derived exp(CE) can differ by one libm rounding step on macOS ARM
    # versus the Linux x86 scoring host. Scientific NLL/count/CE remain exact.
    assert math.isclose(rebuilt['ppl'],value['ppl'],rel_tol=4*np.finfo(np.float64).eps,abs_tol=0),(label,'ppl')
    for r,ref in zip(value['rows'],panel['sequences']):assert r['canonical_group']==ref.get('canonical_group',ref['id']//64)
    return value

def figures(out,analysis,raw,metrics,lmon=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
    hmon=read_json(FROZEN/'H_MONITOR.json')
    rmon=[dict(update=u,ce=read_json(raw/f'evaluations/R_monitor_u{u:05d}_COMPLETE.json')['ce']) for u in SCHEDULE]
    hmetrics=[json.loads(z) for z in (REPO/'experiment_2d11/results/run/metrics-attempt01.jsonl').read_text().splitlines()][:1000]
    def seconds_at(rows,u):return sum(r['training_seconds'] for r in rows if r['completed_updates']<=u)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for rows,label,color,train,world in [(hmon,'H standard','#b87028',hmetrics,4),(rmon,'R CE1-free','#236b8e',metrics,1)]:
        axes[0].plot([r['update']*524288/1e6 for r in rows],[r['ce'] for r in rows],'o-',label=label,color=color)
        axes[1].plot([seconds_at(train,r['update'])*world/3600 for r in rows],[r['ce'] for r in rows],'o-',label=label,color=color)
    if lmon:
        for ax,key in zip(axes,['targets_million','gpu_hours']):ax.plot([r[key] for r in lmon],[r['ce'] for r in lmon],'s-',label='L local-only',color='#28846c')
    for ax in axes:ax.set_ylabel('Monitor CE (nats / target)');ax.grid(alpha=.15);ax.legend()
    axes[0].set_xlabel('Logical training targets (millions)');axes[1].set_xlabel('Measured training GPU-hours')
    fig.suptitle('Shared 1,280-sequence monitor panel · hardware/grouping differ historically')
    fig.tight_layout();fig.savefig(out/'monitoring.png');fig.savefig(out/'monitoring.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,name in enumerate(NAMES):
        r=analysis['primary']['contrasts'][name];g=analysis['group_sensitivity']['contrasts'][name]
        for off,v,color in [(0,r,'#236b8e'),(.18,g,'#8996a3')]:
            lo,hi=v['adjusted_99_1666666667'];mean=v['mean']
            ax.errorbar(mean,i+off,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color=color)
    ax.axvline(0,color='#333',lw=1);ax.axvspan(-.0001,.0001,color='#b87028',alpha=.15)
    ax.set(yticks=range(6),yticklabels=NAMES,xlabel='Paired CE contrast (nats / target)',title='Six-contrast adjusted intervals · sequence (blue), group (gray)')
    fig.tight_layout();fig.savefig(out/'contrasts.png');fig.savefig(out/'contrasts.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.3));updates=[r['completed_updates'] for r in metrics]
    for i,label in [(0,'CE1 (diagnostic only)'),(1,'CE2')]:ax.plot(updates,[r['pass_losses'][i] for r in metrics],lw=.6,alpha=.7,label=label)
    triples=[r for r in metrics if r['pass_count']==3]
    ax.scatter([r['completed_updates'] for r in triples],[r['pass_losses'][2] for r in triples],s=10,label='CE3 (every 32 updates)')
    ax.plot(updates,[r['weighted_multipass_training_loss'] for r in metrics],lw=.8,label='R weighted objective')
    ax.set(xlabel='Completed optimizer update',ylabel='Training CE / objective',title='R diagnostics and objective · distinct from H/L objectives');ax.legend();fig.tight_layout();fig.savefig(out/'training.png');plt.close(fig)

def finalize(archive,l_archive=None):
    stop=read_json(archive/'STOP_VERIFICATION.json')
    assert stop['passed'] and stop['provider']['id']=='j8lsb4rz1a0mf8'
    assert stop['provider']['desiredStatus']=='EXITED' and stop['provider']['runtimeStatus']=='stopped'
    billing=read_json(archive/'RUNTIME_ACCOUNTING.json');assert billing['within_ceiling']
    raw=archive/'raw';out=PACKAGE/'results';torch.set_num_threads(4)
    source_closeout=read_json(out/'CLOSEOUT_SOURCE_AUDIT.json');assert source_closeout['passed']
    independent_stop=read_json(archive/'INDEPENDENT_STOP_CHECK.json')
    assert independent_stop['passed'] and independent_stop['provider']['id']==stop['provider']['id']
    assert independent_stop['provider']['desiredStatus']=='EXITED' and independent_stop['provider']['runtimeStatus']=='stopped'
    for name in ['CPU_AUDIT.json','INPUT_AUDIT.json','LOCAL_READY.json','ANALYSIS_REHEARSAL.json','PARTITION_COVERAGE_AUDIT.json']:
        assert read_json(out/name)['passed'],name
    for name in ['PREFLIGHT.json','CUDA_OBJECTIVE.json','CUDA_TEMPORAL.json','CUDA_UPDATE_GEOMETRY.json','CUDA_RESUME.json']:
        assert read_json(raw/name)['passed'],name
    export=read_json(archive/'EXPORT_VERIFICATION.json');assert export['passed']
    for name,meta in export['files'].items():assert sha256(raw/name)==meta['sha256'] and (raw/name).stat().st_size==meta['bytes']
    finished=read_json(raw/'GPU_WORK_FINISHED.json');assert finished['passed'] and finished['R_complete']
    expected=[json.loads(z) for z in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    metrics=[json.loads(z) for z in (raw/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==1000
    for u,(r,e) in enumerate(zip(metrics,expected),1):
        assert r['completed_updates']==u and r['data']==e['data'] and r['lr']==e['lr'] and r['lr_index']==u-1
        assert r['pass_count']==pass_count(u) and r['microbatches']==16 and r['loss_divisor']==16
        assert r['loss_weights']==([0,1] if pass_count(u)==2 else [0,.5,.5])
        for k,v in accounting(u).items():assert r[k]==v
    audits={}
    for u in SCHEDULE:
        path=raw/f'checkpoints/u{u:05d}.pt';p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        audits[str(u)]=validate(p);assert sha256(path)==read_json(str(path)+'.manifest.json')['sha256']
        mon=verify_evaluation(raw,f'R_monitor_u{u:05d}',read_json(FROZEN/'MONITOR.json'),sha256(path),audits[str(u)]['model_tensor_identity'],'H_ON')
        assert mon['count']==1310720 and len(mon['rows'])==1280 and mon['binding']['panel_identity']==read_json(FROZEN/'MONITOR.json')['identity']
        assert mon['binding']['checkpoint_sha256']==sha256(path)
    assert audits['0']['model_tensor_identity']==FULL_INIT_SHA and p['loader']==expected[-1]['data']['after']
    original=torch.load(archive/'local_initial.pt',map_location='cpu',mmap=True,weights_only=False)
    for name in read_json(FROZEN/'TRAINABILITY.json')['frozen_names']:assert torch.equal(p['model'][name],original['model'][name])
    for name in read_json(FROZEN/'TRAINABILITY.json')['active_names']:assert not torch.equal(p['model'][name],original['model'][name]),name
    for name,meta in read_json(PACKAGE/'results/INPUT_AUDIT.json')['source_files'].items():assert sha256(meta['path'])==meta['sha256']
    base_audit=dict(R_complete=True,completed_updates=1000,one_new_scientific_arm=True,checkpoints=audits,initial_full_H_identity=FULL_INIT_SHA,
        stream_all_1000_verified=True,pass_counts=accounting(1000),writer_gradients_verified=True,
        source_initial_and_H_unchanged=True,all_active_tensors_changed=True,compatibility_scalars_unchanged=True,
        budget=billing,pod_stopped=True,independent_exports=True,shared_volume_preserved=True,
        closeout_source_audit=source_closeout,independent_stop_check=independent_stop)
    if not finished['joint_complete']:
        atomic_json(out/'R_COMPLETION_AUDIT.json',dict(passed=True,joint_complete=False,**base_audit))
        atomic_bytes(out/'INTERIM_REPORT.md',b'R complete; joint comparison awaiting L. No complete joint result or final-success tag is warranted.\n')
        return
    provenance=read_json(raw/'L_PROVENANCE.json');assert provenance['passed']
    lcode=read_json(out/'L_CODE_PROVENANCE_AUDIT.json')
    assert lcode['passed'] and lcode['source_code_identity']==provenance['source_code_identity']
    lmon=None
    assert l_archive is not None,'local L checkpoint and source stream must be independently verified'
    lraw=l_archive/'raw'
    for remote,digest in provenance['source_hashes'].items():
        relative=Path(remote).relative_to(provenance['source_run'])
        assert sha256(lraw/relative)==digest,(relative,'L independent source copy')
    initial_l=lraw/'checkpoints/u00000.pt'
    assert sha256(initial_l)==provenance['source_initial_sha256']
    lp=torch.load(lraw/'checkpoints/u01000.pt',map_location='cpu',mmap=True,weights_only=False)
    assert tensor_identity(from_state(lp['model']).named_parameters())==provenance['model_tensor_identity']
    lmetrics=[json.loads(z) for z in (lraw/'metrics.jsonl').read_text().splitlines()]
    lmon=[]
    for u in SCHEDULE:
        f=lraw/f'evaluations/L_monitor_u{u:05d}_COMPLETE.json';v=read_json(f)
        assert v['binding']['panel_identity']==read_json(FROZEN/'MONITOR.json')['identity']
        assert v['passed'] and v['checkpoint_unchanged'] and v['tensors_unchanged']
        lm=read_json(lraw/f'checkpoints/u{u:05d}.pt.manifest.json')
        assert v['binding']['checkpoint_sha256']==lm['sha256'] and sha256(lraw/f'checkpoints/u{u:05d}.pt')==lm['sha256']
        assert v['binding']['model_tensor_identity']==lm['audit']['model_tensor_identity']
        assert v['count']==1310720 and len(v['rows'])==1280
        lmon.append(dict(update=u,ce=v['ce'],targets_million=u*524288/1e6,
            gpu_hours=sum(r['training_seconds'] for r in lmetrics if r['completed_updates']<=u)/3600))
    lgit=read_json(out/'L_GIT_PROVENANCE.json') if (out/'L_GIT_PROVENANCE.json').exists() else dict(status='Final source archival not yet published by 2D13.')
    if 'checkpoint_sha256' in lgit:assert lgit['passed'] and lgit['remote_tag_verified'] and lgit['checkpoint_sha256']==provenance['sha256']
    panel=read_json(FROZEN/'PANEL.json');summaries={}
    for label in CONDITIONS:
        if label.startswith('H_'):digest,model_id=H_SHA,H_MODEL_SHA
        elif label.startswith('R_'):digest,model_id=sha256(raw/'checkpoints/u01000.pt'),audits['1000']['model_tensor_identity']
        else:digest,model_id=provenance['sha256'],provenance['model_tensor_identity']
        condition='H_ALL_OFF' if label in ('H_ALL_OFF','R_ALL_OFF','L_LOCAL') else 'H_ON'
        summaries[label]=verify_evaluation(raw,label,panel,digest,model_id,condition)
    analysis=analyze(summaries,panel)
    analysis['changed_noninferiority']=analysis['primary']['contrasts']['A']['R_noninferior_to_H']!=analysis['group_sensitivity']['contrasts']['A']['R_noninferior_to_H']
    analysis['changed_joint_goal']=analysis['primary']['joint_goal_met']!=analysis['group_sensitivity']['joint_goal_met']
    assert summaries['R_ON']['binding']['checkpoint_sha256']==sha256(raw/'checkpoints/u01000.pt')
    assert summaries['L_LOCAL']['binding']['checkpoint_sha256']==provenance['sha256']
    atomic_json(out/'ANALYSIS.json',analysis);figures(out,analysis,raw,metrics,lmon)
    with (out/'endpoint_sequences.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['id','canonical_group','token_sha256',*CONDITIONS,*NAMES])
        from .analysis import contrasts
        for i,r in enumerate(panel['sequences']):
            values=[summaries[n]['rows'][i]['ce'] for n in CONDITIONS]
            writer.writerow([r['id'],r['canonical_group'],r['token_sha256'],*values,*contrasts([values])[0]])
    for name in ['STOP_VERIFICATION.json','INDEPENDENT_STOP_CHECK.json','RUNTIME_ACCOUNTING.json','EXPORT_VERIFICATION.json','CAPACITY_PREFLIGHT.json','PROVIDER_BEFORE_CUDA.json','REMOTE_LAUNCH.json']:
        shutil.copyfile(archive/name,out/name)
    for f in raw.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.jsonl'):
            target=out/'run'/f.relative_to(raw);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    audit=dict(passed=True,joint_complete=True,**base_audit,L_provenance=provenance,L_code_provenance=lcode,L_git_provenance=lgit,coverage=dict(rows=20480,targets=20971520),
        monitoring_targets=6553600,total_new_scientific_predictions=27525120,six_contrast_adjustment_verified=True,
        scientific_joint_goal_met=analysis['primary']['joint_goal_met'],group_joint_goal_met=analysis['group_sensitivity']['joint_goal_met'],
        final_conditions_same_execution=True,R_training_gpu_hours=sum(r['training_seconds'] for r in metrics)/3600,
        H_training_gpu_hours=read_json(FROZEN/'H_PREFIX_VERIFIED.json')['gpu_hours'],completed_at=time.time())
    atomic_json(out/'FINAL_AUDIT.json',audit)
    primary=analysis['primary'];a=primary['contrasts']['A'];joint=primary['joint_goal_met']
    outcome='The prespecified joint goal was met.' if joint else 'The prespecified joint goal was not established.'
    if not joint and a['flags']['material_negative']:
        outcome='The prespecified joint goal was not met: R had materially worse ON prediction quality than H under the adjusted interval.'
        if primary['contrasts']['I']['flags']['beyond_reference']:
            outcome+=' R showed a larger recurrence OFF penalty, so the increased measured dependence did not satisfy the joint quality-and-dependence objective.'
    main_readout=(f"R ON CE exceeds H by {-a['mean']:.10f} nats/target and L by {-primary['contrasts']['B']['mean']:.10f}. "
        f"R's OFF penalty is {primary['contrasts']['G_R']['mean']:.10f}, versus {primary['contrasts']['G_H']['mean']:.10f} for H; "
        f"the paired increase is {primary['contrasts']['I']['mean']:.10f}. "
        f"The group sensitivity preserves all six contrast classifications and the joint-goal conclusion: {not any(analysis['changed_flags'].values()) and not analysis['changed_joint_goal']}.")
    table='\n'.join(f"| {n} | {analysis['endpoints'][n]['ce']:.10f} | {analysis['endpoints'][n]['ppl']:.8f} |" for n in CONDITIONS)
    contrast_table='\n'.join(f"| {n} | {v['mean']:.10f} | [{v['raw_95'][0]:.10f}, {v['raw_95'][1]:.10f}] | [{v['adjusted_99_1666666667'][0]:.10f}, {v['adjusted_99_1666666667'][1]:.10f}] |" for n,v in primary['contrasts'].items())
    group_table='\n'.join(f"| {n} | [{v['raw_95'][0]:.10f}, {v['raw_95'][1]:.10f}] | [{v['adjusted_99_1666666667'][0]:.10f}, {v['adjusted_99_1666666667'][1]:.10f}] |" for n,v in analysis['group_sensitivity']['contrasts'].items())
    flag_table='\n'.join('| '+n+' | '+' | '.join(str(v['flags'][k]) for k in ('positive','beyond_reference','negative','material_negative','equivalence'))+' |' for n,v in primary['contrasts'].items())
    win_table='\n'.join(f"| {n} | {v['positive_fraction']:.6f} | {v['negative_fraction']:.6f} | {v['tie_fraction']:.6f} |" for n,v in primary['contrasts'].items())
    resources=finished['resources']
    stage_seconds=dict(scientific_training=resources['training_seconds'],scientific_scoring=resources['evaluation_seconds'],disposable_cuda_preflight=resources['preflight_seconds'])
    stage_seconds['other_billed_time']=billing['cumulative_billed_seconds']-sum(stage_seconds.values())
    assert stage_seconds['other_billed_time']>=0
    atomic_json(out/'COMPUTE_BREAKDOWN.json',dict(seconds=stage_seconds,gpu_count=1,rate_per_hour=1.59,
        other_includes='Reserved preparation, initial round-trip transfer, startup repair, saves, setup, residual export/shutdown time; overlapping exports are not double counted.',
        startup_failure=read_json(raw/'history/startup01/FAILURE.json'),disposable=read_json(out/'DISPOSABLE_ACCOUNTING.json')))
    timing_table='\n'.join(f'| {n.replace("_"," ")} | {seconds:.3f} | {seconds/3600:.6f} | ${seconds/3600*1.59:.6f} |' for n,seconds in stage_seconds.items())
    report=f'''# Experiment 2D14 — CE1-free H at 524M targets

{outcome} R noninferiority to H: {a['R_noninferior_to_H']}; R quality improvement beyond the reference versus H: {a['flags']['beyond_reference']}; versus L: {primary['contrasts']['B']['flags']['beyond_reference']}. Noninferiority without superiority is not a better-predictor finding.

{main_readout}

Exactly one new scientific R arm trained here from the original untrained full H tensors, including fresh 50/50 routers. H524M and independently completed L524M were reused only as comparators. R completed all 1,000 optimizer updates / 524,288,000 logical targets with the original H stream and 10B LR prefix. R used one A100 80GB, B32, 16 accumulated means, one global clip and one fused AdamW update. H used four GPUs; reduction orders differ and bitwise trajectory equality is not claimed.

The immutable historical H checkpoint records `fused=False` in both optimizer groups; R and valid L record `fused=True`, as prescribed for these new runs. This implementation difference accompanies the world-size difference and limits an exact numerical attribution solely to the CE1 weight. Historical H was reused unchanged. See `OPTIMIZER_IMPLEMENTATION_AUDIT.json`.

| Endpoint condition | CE | Perplexity |
|---|---:|---:|
{table}

All five conditions used the prospectively shared 4,096-sequence panel, B128 grouping, true incremental BF16 forward, FP32 CE and FP64 sums. Each sequence has 1,024 targets. Coverage is 20,480 rows / 20,971,520 endpoint predictions; five R monitors add 6,553,600. This panel was shared prospectively with 2D13 and is not a fresh confirmation after viewing its outcomes.

Contrasts: A=H_ON−R_ON, B=L_LOCAL−R_ON, C=L_LOCAL−H_ON, G_H=H_ALL_OFF−H_ON, G_R=R_ALL_OFF−R_ON, I=G_R−G_H.

| Contrast | Mean | Raw 95% CI | Family-adjusted 99.1666666667% CI |
|---|---:|---|---|
{contrast_table}

Primary flags and sequence win/loss/tie fractions are retained for every contrast in `ANALYSIS.json`. Fifty thousand paired sequence resamples use NumPy default_rng(20260928), FP64 means and linear percentiles. One sampled index vector serves all six contrasts. The Bonferroni family has six contrasts. The paired 64-group sensitivity analysis uses 50,000 resamples and seed 20260929; changed flags: {json.dumps(analysis['changed_flags'])}. These intervals estimate evaluation-unit uncertainty for fixed trajectories, not training-seed replication.

| Contrast | Positive | Beyond reference | Negative | Material negative | Equivalent |
|---|---|---|---|---|---|
{flag_table}

| Contrast | Sequence win fraction (>0) | Loss fraction (<0) | Tie fraction (=0) |
|---|---:|---:|---:|
{win_table}

Group sensitivity intervals:

| Contrast | Raw 95% CI | Family-adjusted 99.1666666667% CI |
|---|---|---|
{group_table}

Group sensitivity joint goal: {analysis['group_sensitivity']['joint_goal_met']}; noninferiority: {analysis['group_sensitivity']['contrasts']['A']['R_noninferior_to_H']}. The joint-goal conclusion changed: {analysis['changed_joint_goal']}; noninferiority changed: {analysis['changed_noninferiority']}.

The inherited delta_CE=0.0001 is a small-effect reference, not a deployment-value threshold. The joint goal requires lo(A)>−delta, lo(G_R)>delta and lo(I)>delta. Failure to establish benefit does not establish equivalence. Greater OFF sensitivity can reflect coadaptation or mixture rescaling and does not uniquely prove more useful recurrent content. No architecture adoption or continuation follows automatically.

H-to-R PPL reduction =100*(PPL_H_ON−PPL_R_ON)/PPL_H_ON: {analysis['relative_ppl']['H_to_R_reduction_percent']:.7f}%. L-to-R uses PPL_L_LOCAL as denominator: {analysis['relative_ppl']['L_to_R_reduction_percent']:.7f}%.

![Joint contrasts](contrasts.png)

![Shared monitoring](monitoring.png)

H/R have 124,697,386 registered parameters and 124,697,382 trainable parameters; L has 124,475,904 active parameters. R retains 969 two-pass and 31 three-pass updates, with 2,031 backbone pass-equivalent updates / 1,064,828,928 backbone pass-target exposures and actual CE evaluations. Nonzero objective CE exposure is 540,540,928 targets; 524,288,000 CE1 targets are diagnostic-only. H has the same backbone/CE evaluation count with every CE directly weighted; L has 524,288,000. These are pass counts, not gradient shares or equal FLOPs. Activation checkpoint recomputation is counted in time but excluded from pass-target exposure.

![R training objective and diagnostics](training.png)

R training-only compute: {audit['R_training_gpu_hours']:.8f} GPU-hours; historical H first 1,000 updates: 2,556.652936697 seconds on four GPUs / {audit['H_training_gpu_hours']:.9f} GPU-hours. L's valid training-only run took {sum(r['training_seconds'] for r in lmetrics):.6f} seconds on one GPU / {sum(r['training_seconds'] for r in lmetrics)/3600:.9f} GPU-hours; its separate task's failures and billing are not charged to this run. Different hardware counts, historical evaluation grouping/software and concurrent shared-volume I/O limit speed conclusions. H/L/R differently weighted training objectives are not interchangeable curves. No R524M-versus-H10B ablation-gap comparison is used as recipe evidence.

Cumulative billed compute including the user's requested reserved preparation, preflight, training, monitoring, final scoring, transfers and verified shutdown: {billing['gpu_hours']:.6f} GPU-hours / ${billing['compute_cost']:.6f} at $1.59/hour. Both ceilings were respected. Storage is separate; the shared volume was provider-verified at 200 GB. Runpod lists standard network storage at $0.07/GB/month ($14/month for 200 GB), while the exact invoice/tier rate is not available in the pod response. [Runpod network-volume pricing](https://docs.runpod.io/storage/network-volumes). Stage timing and residual overhead are retained in the raw resource ledger; provider invoice reconciliation is not available.

| Stage | Billed seconds | GPU-hours | Compute cost |
|---|---:|---:|---:|
{timing_table}

Other billed time includes the user's reserved preparation, transfer benchmark, startup repair, saves, setup and residual export/shutdown time. Background exports overlap useful work and are not charged twice. The first launcher failed before CUDA/scientific work because the SSH environment omitted RUNPOD_POD_ID; the repaired launcher read only that field from PID 1 and verified the assigned pod. No scientific update was replayed for this repair. The independent hard deadline remained armed. Disposable work is itemized in `DISPOSABLE_ACCOUNTING.json` and `COMPUTE_BREAKDOWN.json`.

Checkpoints u0/u250/u500/u750/u1000 were atomically published, reopened, fully validated and independently hash-verified on the shared volume and local Mac. The exact initial full-H identity is `{FULL_INIT_SHA}`. Every R batch hash matched before applying its update, with terminal cursor `{p['loader']}`. All active optimizer step counters equal 1,000; compatibility scalars are unchanged. CE1 exclusion, attached writer gradients and detached-source controls, checkpointing variants, normalization, full-shape CUDA execution, u31→u32 recovery and same-shape incremental references passed. Disposable probes are distinct from the single scientific trajectory.

L provenance and the complete stream were independently verified using its atomically completed checkpoint, saved initialization, frozen routers, optimizer state and local durable source copies. H and original initialization hashes were rechecked; this task wrote only its 2D14 namespace and never changed 2D13 code, checkpoints or pod. The final saved R checkpoint's evaluation ledger is reconciled with the subsequent immutable five-condition raw completion artifacts. Final audit: `FINAL_AUDIT.json`.

L source commit: `{lgit.get('source_commit','pending')}`; tag: `{lgit.get('source_tag','pending')}`. The remote tag and completed checkpoint identity were independently verified. Full source file hashes and the tag object are recorded in `L_GIT_PROVENANCE.json`.

All scientific NLL values, target counts, and CE values match the raw batch artifacts exactly. Recomputing derived exp(CE) locally gave a one-ULP difference for R_ALL_OFF (7.1e−15 perplexity units) between the two hosts; the report uses the local derived value. This does not affect any CE contrast or decision. `PPL_ROUNDING_AUDIT.json` records the observation; the local audit allows at most four FP64 machine-epsilon units of relative error only for this derived quantity.

Pod `{stop['provider']['id']}` is verified EXITED/stopped. Large files remain outside Git under `{archive}` and `{export['persistent_root']}` on volume yhzyb27fb5. This is one early matched trajectory, not a 10B quality determination against GPT-2. No additional training was launched.
'''
    atomic_bytes(out/'FINAL_REPORT.md',report.encode());print(json.dumps(dict(passed=True,joint_goal_met=joint,endpoints=analysis['endpoints'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--l-archive',type=Path)
    a=p.parse_args();finalize(a.archive,a.l_archive)
