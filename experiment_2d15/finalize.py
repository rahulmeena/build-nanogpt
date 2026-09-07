"""Local paired figures, execution ledger and final report after verified stop."""
import argparse,math,time
import numpy as np
from .common import *
from .analysis import run as analyze,NAMES

def arm_resource_details(rows,evaluations):
    last=rows[-1]
    fields=('completed_updates','logical_targets','actual_ce_target_evaluations',
            'nonzero_weight_ce_targets',
            'backbone_pass_equivalent_updates','backbone_pass_targets')
    value={key:last[key] for key in fields}
    value['diagnostic_only_ce1_targets']=0 if last['arm']=='L_nf4' else last['diagnostic_only_ce1_targets']
    value['two_pass_updates']=sum(r['pass_count']==2 for r in rows)
    value['three_pass_updates']=sum(r['pass_count']==3 for r in rows)
    value.update(training_seconds=last['cumulative_training_seconds'],
                 training_gpu_hours=last['cumulative_training_seconds']*4/3600,
                 evaluation_seconds=math.fsum(e['seconds'] for e in evaluations),
                 evaluation_count=len(evaluations),
                 evaluation_predictions=sum(e['count'] for e in evaluations))
    return value

def run(archive):
    stop=read_json(archive/'STOP_VERIFICATION.json');assert stop['passed']
    result=analyze(archive);assert result['complete']
    from .state import complete
    assert complete(read_json(archive/'COMBINED_STATE.json'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    binding=read_json(archive/'binding.json');end=stop['time'];bill=end-binding['billing_start']
    ledger=dict(billing_start=binding['billing_start'],verified_stop=end,total_pod_hours=bill/3600,total_gpu_hours=4*bill/3600,whole_pod_dollars_per_hour=binding['whole_pod_hourly_rate'],estimated_compute_dollars=bill/3600*binding['whole_pod_hourly_rate'],includes_held_preparation_preflights_transitions_evaluation_exports=True,storage_excluded_from_compute_quote=True)
    times={};metric_audits={};metrics={};resources={}
    expected=[__import__('json').loads(x) for x in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    for arm in ARMS:
        rows=[__import__('json').loads(x) for x in (archive/arm/'metrics.jsonl').read_text().splitlines()]
        metrics[arm]=rows
        assert [r['completed_updates'] for r in rows]==list(range(1,5001))
        for r,e in zip(rows,expected):assert r['data']==e['data'] and r['lr']==e['lr'] and r['fused'] is False and r['foreach'] is None
        times[arm]=[rows[u-1]['cumulative_training_seconds']*4/3600 for u in MILESTONES]
        metric_audits[arm]=dict(passed=True,updates=5000,logical_targets=2621440000,terminal_cursor=rows[-1]['data']['after'],training_gpu_hours=times[arm][-1],metrics_sha256=sha256(archive/arm/'metrics.jsonl'))
        from .state import required_labels
        evaluations=[read_json(archive/arm/'evaluations'/(label+'_COMPLETE.json')) for label in required_labels(arm)]
        resources[arm]=arm_resource_details(rows,evaluations)
    serial=read_json(archive/'controller/SERIAL_LAUNCH.json')
    preflight=read_json(archive/'controller/BOTH_ARMS_PREFLIGHT_PASSED.json')
    prestart=serial['time']-binding['billing_start']
    train_seconds=math.fsum(v['training_seconds'] for v in resources.values())
    evaluation_seconds=math.fsum(v['evaluation_seconds'] for v in resources.values())
    remaining_seconds=bill-prestart-train_seconds-evaluation_seconds
    assert remaining_seconds>=0
    ledger.update(per_arm=resources,combined_new_logical_targets=sum(v['logical_targets'] for v in resources.values()),
                  evaluation_predictions=sum(v['evaluation_predictions'] for v in resources.values()),
                  measured_stage_seconds=dict(held_preparation_and_all_disposable_preflight_attempts=prestart,
                                             scientific_training=train_seconds,required_gpu_evaluations=evaluation_seconds,
                                             remaining_serial_overhead_and_post_gpu_exports=remaining_seconds),
                  successful_disposable_preflight_seconds=preflight['seconds'],
                  successful_preflight_is_subset_of_preparation=True,
                  failed_preflight_attempt_durations_separately_attributed=False,
                  scientific_replay_records=[str(p.relative_to(archive)) for arm in ARMS for p in (archive/arm).glob('discarded-*.jsonl')],
                  overhead_note='Residual includes setup, checkpoint publication, transitions and final exports. Transfers overlapping GPU work are not added again.')
    assert ledger['combined_new_logical_targets']==5242880000 and ledger['evaluation_predictions']==86507520
    hr=[__import__('json').loads(x) for x in (REPO/'experiment_2d11/results/run/metrics-attempt01.jsonl').read_text().splitlines()][:5000]
    times['H']=[sum(r['training_seconds'] for r in hr[:u])*4/3600 for u in MILESTONES]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    historical=read_json(FROZEN/'H_MONITOR.json');fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for name,color in [('H','#225ea8'),('L_nf4','#238b45'),('R_nf4','#cb181d')]:
        ce_values=([r['ce'] for r in historical] if name=='H' else [read_json(archive/name/'evaluations'/f'{name}_monitor_u{u:05d}_COMPLETE.json')['ce'] for u in SCHEDULE])
        hours=([sum(r['training_seconds'] for r in hr[:u])*4/3600 for u in SCHEDULE] if name=='H' else [metrics[name][u-1]['cumulative_training_seconds']*4/3600 if u else 0 for u in SCHEDULE])
        for ax,x in zip(axes,[np.array(SCHEDULE)*524288/1e9,hours]):ax.plot(x,ce_values,'o-',label=name+(' (historical grouping)' if name=='H' else ''),color=color)
    axes[0].set_xlabel('Logical training targets (billions)');axes[1].set_xlabel('Measured training GPU-hours')
    for ax in axes:ax.set_ylabel('Incremental monitor CE (nats)');ax.legend(fontsize=8)
    fig.suptitle('Original 1,280-sequence monitor — descriptive; H batch grouping differs');fig.savefig(archive/'monitoring_trajectories.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');targets=np.array(MILESTONES)*524288/1e9
    colors=dict(H_ON='#225ea8',L_LOCAL='#238b45',R_ON='#cb181d',H_ALL_OFF='#9ecae1',R_ALL_OFF='#fcae91')
    for c in CONDITIONS:
        mean=[result['endpoint_means'][str(u)][c]['ce'] for u in MILESTONES]
        hours=times['H' if c.startswith('H') else 'L_nf4' if c.startswith('L') else 'R_nf4']
        for col,x in enumerate([targets,hours]):
            axes[0,col].plot(x,mean,'o-',label=c,color=colors[c]);axes[1,col].plot(x,np.exp(mean),'o-',label=c,color=colors[c])
    for ax in axes[0]:ax.set_ylabel('True incremental CE (nats)')
    for ax in axes[1]:ax.set_ylabel('Perplexity')
    for ax in axes[:,0]:ax.set_xlabel('Logical training targets (billions)')
    for ax in axes[:,1]:ax.set_xlabel('Measured training GPU-hours')
    axes[0,0].legend(fontsize=8);fig.suptitle('2D15 — same frozen panel, three ages, one trajectory per arm');fig.savefig(archive/'ce_ppl_trajectories.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,name in zip(axes.flat,NAMES):
        values=[a['contrasts'][name] for a in result['primary']['ages']];means=np.array([v['mean'] for v in values]);bounds=np.array([v['adjusted_99_7222222222'] for v in values])
        ax.errorbar(targets,means,yerr=np.stack((means-bounds[:,0],bounds[:,1]-means)),fmt='o-',capsize=4,color='#225ea8');ax.axhline(0,color='gray',linewidth=.7);ax.axhspan(-.0001,.0001,color='gray',alpha=.12);ax.set_title(name);ax.set_xlabel('Logical targets (billions)');ax.set_ylabel('Paired CE difference')
    fig.suptitle('18-contrast family: Bonferroni-adjusted 99.7222% intervals');fig.savefig(archive/'paired_contrasts.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');hours=np.array(times['L_nf4'])+np.array(times['R_nf4'])
    for ax,name in zip(axes.flat,NAMES):
        values=[a['contrasts'][name] for a in result['primary']['ages']];means=np.array([v['mean'] for v in values]);bounds=np.array([v['adjusted_99_7222222222'] for v in values])
        ax.errorbar(hours,means,yerr=np.stack((means-bounds[:,0],bounds[:,1]-means)),fmt='o-',capsize=4,color='#225ea8');ax.axhline(0,color='gray',linewidth=.7);ax.axhspan(-.0001,.0001,color='gray',alpha=.12);ax.set_title(name);ax.set_xlabel('L+R training GPU-hours at matched age');ax.set_ylabel('Paired CE difference')
    fig.suptitle('Paired contrasts versus measured new-arm training compute');fig.savefig(archive/'paired_contrasts_gpu_hours.png',dpi=180);plt.close(fig)
    atomic_json(archive/'RESOURCE_LEDGER.json',ledger);atomic_json(archive/'FINAL_STREAM_AUDIT.json',metric_audits)
    text=['# Experiment 2D15 — final combined report','',
        'Fresh L_nf4 and CE1-free R_nf4 each completed 5,000 updates / 2,621,440,000 logical targets. Historical H checkpoints were reused only as comparators. All 18 monitors and 15 milestone condition evaluations passed identity/coverage verification; required checkpoints and raw outputs were independently exported before the assigned pod was stopped.','',
        'The optimizer options and four-GPU geometry matched H, including fused=False and foreach=None. This intentional fresh restart preserves the original fused=True 2D13/2D14 results as separate historical experiments. L collapses equivalent local objectives to one pass; R retains attached recurrent writer gradients with CE1 diagnostic-only. Architecture/pass differences remain, and this is one training trajectory per arm.','',
        '| Updates | H_ON CE | L_LOCAL CE | R_ON CE | H_ALL_OFF CE | R_ALL_OFF CE | CE1-free joint goal |','|---:|---:|---:|---:|---:|---:|:---|']
    for i,u in enumerate(MILESTONES):
        text.append('| '+str(u)+' | '+' | '.join(f"{result['endpoint_means'][str(u)][c]['ce']:.9f}" for c in CONDITIONS)+' | '+str(result['primary']['ages'][i]['joint_goal_met'])+' |')
    text+=['','The main endpoint is update 5000. All classifications use the fixed 18-contrast family. Paired sequence and 64-group bootstrap results use 50,000 resamples each, with common sampled indices across conditions and ages. Raw 95% and adjusted 99.7222222222% intervals, practical-equivalence flags, and R-versus-L results are in ANALYSIS.json. Larger OFF sensitivity alone does not establish better quality or isolated recurrent-content usefulness. No matched 10B/HellaSwag claim is made.','',
        f"Total allocation through verified stop: {ledger['total_pod_hours']:.4f} pod-hours / {ledger['total_gpu_hours']:.4f} GPU-hours; estimated compute ${ledger['estimated_compute_dollars']:.2f} at $6.36/pod-hour, excluding storage. This includes held preparation, every disposable/retried preflight, transitions, scoring and exports.",'',
        'Disposable preflight and failure records are preserved under controller/preflight*; discarded scientific replay records, if any, remain within each arm. Their work is excluded from the two scientific token budgets and included in total billed time. The first disposable attempt caught an incorrect dispatch-audit assertion; both new arms were subsequently required to match H’s measured dispatch without changing numerical tolerances.','',
        'Execution success and scientific goal attainment are distinct. Complete contrasts and sensitivity classifications should be consulted even where the joint-goal cell is false.']
    text+=['','| New arm | Logical targets | Actual CE pass-target evaluations | Nonzero-weight CE targets | Diagnostic-only CE1 targets | Backbone passes | Training GPU-hours |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,value in resources.items():
        text.append(f"| {arm} | {value['logical_targets']:,} | {value['actual_ce_target_evaluations']:,} | {value['nonzero_weight_ce_targets']:,} | {value['diagnostic_only_ce1_targets']:,} | {value['backbone_pass_equivalent_updates']:,} | {value['training_gpu_hours']:.4f} |")
    text+=['', 'Activation-checkpoint recomputation is excluded from exposure counters and included in measured compute. Both new arms together used 5,242,880,000 logical training targets. The prescribed evaluations scored 86,507,520 target predictions.', '',
           '| Allocated stage | Pod-hours | GPU-hours |', '|---|---:|---:|']
    stage_names=dict(held_preparation_and_all_disposable_preflight_attempts='Held preparation and all disposable preflight attempts',scientific_training='Scientific L and R training',required_gpu_evaluations='Required GPU evaluations',remaining_serial_overhead_and_post_gpu_exports='Remaining serial overhead and final exports')
    for name,seconds in ledger['measured_stage_seconds'].items():text.append(f"| {stage_names[name]} | {seconds/3600:.4f} | {seconds*4/3600:.4f} |")
    text+=['',f"The successful disposable preflight took {preflight['seconds']:.3f} seconds, within the preparation row. Earlier failed-attempt durations are not separately attributed here; their work is included in that row and total allocation. All disposable checks consumed zero scientific updates. Transfers overlapping training/scoring are not double-counted.", '',
           'Both initial GPU audits and all failure records remain preserved. The second disposable attempt exposed an extra cross-restart bitwise-identity assertion after the numerical tolerance checks had passed; the successful preflight retained the original numerical tolerances and separately checked exact within-run rank agreement and repeat restoration.']
    if not ledger['scientific_replay_records']:text+=['','No discarded or replayed scientific updates were recorded. Export-controller recoveries did not restart either GPU training trajectory.']
    archival=archive/'HISTORICAL_ARCHIVAL_FINAL.json'
    if archival.exists():
        moved=read_json(archival);text+=['',f"Historical archival freed {moved['bytes_freed']/1e9:.2f} GB across {moved['files']} files, with verified Mac copies and removals gated by active training. Protected inputs and completed 2D13/2D14 artifacts were preserved. The Mac archive index is HISTORICAL_ARCHIVE_INDEX.md."]
    atomic_bytes(archive/'FINAL_REPORT.md',('\n'.join(text)+'\n').encode())
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);run(p.parse_args().archive)
