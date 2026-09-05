#!/usr/bin/env python3
"""Local post-shutdown analysis for the single prespecified H250M/D250M contrast."""
from pathlib import Path
import datetime as dt
import hashlib,json,math,time
import numpy as np
from experiment_2d10_analysis import flags,render_flag_values,FLAG_COLUMNS,sha,read
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/experiment_2d10_h_250m'
ARCHIVE=ROOT.parents[1]/'runpod-checkpoint-archive/experiment_2d10_h_250m'
OLD=ROOT/'results/experiment_2d10_retrieval_aware_gating_100m'
HIST=ROOT/'results/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m'
def write(name,value):
    (RESULT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def bootstrap(losses):
    values=losses['D']-losses['H']
    assert values.shape==(4096,) and np.isfinite(values).all()
    rng=np.random.default_rng(20260913);means=np.empty(50000,dtype=np.float64)
    for start in range(0,50000,128):
        end=min(start+128,50000)
        indices=rng.integers(0,4096,size=(end-start,4096))
        means[start:end]=values[indices].mean(axis=1)
    ci=np.percentile(means,[2.5,97.5],method='linear').tolist()
    np.save(ARCHIVE/'BOOTSTRAP_MEANS.npy',means)
    return {'contrast':'D-H','positive_favors':'H','mean':float(values.mean()),'ci_95':ci,
        'flags':flags(ci),'exp_contrast':math.exp(float(values.mean())),
        'H_wins':int((values>0).sum()),'D_wins':int((values<0).sum()),'ties':int((values==0).sum()),
        'seed':20260913,'resamples':50000,'numpy_version':np.__version__,'percentile_method':'linear',
        'resampling_unit':'paired sequence','confidence_percent':95,'percentiles':[2.5,97.5],
        'margin':0.0001,'bootstrap_means_sha256':sha(ARCHIVE/'BOOTSTRAP_MEANS.npy')}

def gate_statistics():
    result={'quantile_seed':20260914,'quantiles_percent':[1,5,25,50,75,95,99],
            'quantile_sampling':'one fixed sample of up to 131072 eligible positions per destination; same positions across arms and metrics',
            'moments':'full-panel FP64 population mean/variance/extrema/fractions; eligible and unavailable positions separate',
            'scalar_exports':{},'arms':{}}
    for arm in 'H':
        path=ARCHIVE/'gpu_artifacts/evaluation'/f'GATE_SCALARS_{arm}.npy'
        array=np.load(path,mmap_mode='r')
        keys=(['u','delta','g','cast_g'] if arm=='T' else ['logit_difference','lambda_L','lambda_R','cast_lambda_L','cast_lambda_R','entropy'])
        assert array.shape==(3,len(keys)+1,4096,1024) and np.isfinite(array).all()
        result['scalar_exports'][arm]={'path':str(path),'sha256':sha(path),'shape':list(array.shape)}
        rng=np.random.default_rng(20260914); blocks={}
        for bi,(block,lag) in enumerate(((1,1),(3,31),(5,63))):
            available=array[bi,-1].astype(bool)
            expected=np.broadcast_to(np.arange(1024)[None,:]>=lag,(4096,1024))
            assert np.array_equal(available,expected)
            eligible_indices=np.flatnonzero(available.ravel())
            sampled=rng.choice(eligible_indices,size=min(131072,len(eligible_indices)),replace=False)
            stats={'eligible_count':int(available.sum()),'unavailable_count':int((~available).sum()),
                'quantile_sample_count':len(sampled),'quantile_index_sha256':hashlib.sha256(sampled.tobytes()).hexdigest(),'metrics':{}}
            for mi,key in enumerate(keys):
                values=array[bi,mi].ravel()
                groups={}
                for group,mask in [('eligible',available.ravel()),('unavailable',~available.ravel())]:
                    v=values[mask].astype(np.float64)
                    groups[group]={'count':len(v),'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max()),
                        'negative_fraction':float((v<0).mean()),'quantiles':np.percentile(values[sampled] if group=='eligible' else v,[1,5,25,50,75,95,99],method='linear').tolist()}
                stats['metrics'][key]=groups
            if arm=='H':
                assert np.array_equal(array[bi,1][~available],np.ones((~available).sum(),dtype=np.float32))
                assert np.count_nonzero(array[bi,2][~available])==0
                assert np.max(np.abs(array[bi,1]+array[bi,2]-1))<=2e-7
            blocks[f'B{block}']=stats
        result['arms'][arm]=blocks
    return result

def parameter_changes(source,final):
    import torch
    a=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(final,map_location='cpu',mmap=True,weights_only=False)
    result={}
    for name in a['model']:
        if name.startswith('routers.'):
            x=a['model'][name].double();y=b['model'][name].double()
            result[name]={'source_norm':float(x.norm()),'final_norm':float(y.norm()),
                'change_norm':float((y-x).norm()),'max_absolute_change':float((y-x).abs().max()),
                'source_values':x.tolist() if name.endswith('.b2') else None,
                'final_values':y.tolist() if name.endswith('.b2') else None}
    assert a['parameter_names']==b['parameter_names']
    assert a['optimizer_parameter_names']==b['optimizer_parameter_names']
    assert a['scheduler']==b['scheduler']
    retired={'g_rec','g_rec_b3','g_rec_b5','g_rec_b6'}
    for name in retired:assert torch.equal(a['model'][name],b['model'][name])
    for ga,gb,names in zip(a['optimizer']['param_groups'],b['optimizer']['param_groups'],a['optimizer_parameter_names']):
        assert ga==gb
        for ia,ib,name in zip(ga['params'],gb['params'],names):
            sa=a['optimizer']['state'][ia];sb=b['optimizer']['state'][ib]
            if name in retired:
                assert set(sa)==set(sb)
                assert all(torch.equal(v,sb[k]) if torch.is_tensor(v) else v==sb[k] for k,v in sa.items())
            else:assert int(sb['step'])==int(sa['step'])+286
    return {'router_parameters':result,'complete_retired_state_unchanged':True,
        'all_active_counters_advanced_286':True,'scheduler_and_groups_preserved':True}

def main():
    started=time.time();stop=read(RESULT/'STOP_VERIFICATION.json')
    assert stop['passed'] and stop['pod']['runtimeStatus']=='stopped' and stop['persistent_volume_retained']
    assert read(RESULT/'OPERATIONS_STATUS.json')['outcome']=='gpu_work_complete'
    assert read(RESULT/'ARTIFACT_BACKUP_VERIFICATION.json')['passed']
    panel=read(RESULT/'EVALUATION_PANEL_MANIFEST.json')
    disjoint=read(RESULT/'DISJOINTNESS_AUDIT.json')
    assert disjoint['passed'] and disjoint['panel_sha256']==panel['panel_sha256']
    recovery=read(RESULT/'RECOVERY_CHECKPOINT_VERIFICATION.json')
    assert recovery['local_updates']==144 and recovery['strict_reopen']['passed']
    ev={a:read(RESULT/(a+'_REAL.json')) for a in 'DH'}
    losses={a:np.asarray(v['per_sequence_ce'],dtype=np.float64) for a,v in ev.items()}
    for a,v in ev.items():
        assert v['passed'] and v['status']=='complete' and v['targets']==4194304
        assert losses[a].shape==(4096,) and np.isfinite(losses[a]).all()
        assert v['sequence_identities']==panel['sequence_identities']
        assert v['batch_identities']==panel['batch_identities']
        assert v['completed_batch_indices']==panel['batch_indices_in_evaluation_order']
        assert v['panel_sha256']==panel['panel_sha256'] and v['panel_manifest_sha256']==sha(RESULT/'EVALUATION_PANEL_MANIFEST.json')
        assert v['checkpoint_file_unchanged'] and v['model_tensors_unchanged']
        assert v['persistent_state'] and all(x==33289728 for x in v['persistent_state'].values())
        np.testing.assert_allclose(losses[a]*1024,v['per_sequence_nll'],rtol=0,atol=1e-12)
        assert abs(losses[a].mean()-v['aggregate_ce'])<1e-12
    np.savez_compressed(RESULT/'PAIRED_SEQUENCE_LOSSES.npz',**losses,targets_per_sequence=np.full(4096,1024,dtype=np.int64))
    training=read(RESULT/'TRAINING_COMPLETE_H.json');preflight=read(RESULT/'PREFLIGHT_AUDIT.json')
    source=read(RESULT/'SOURCE_AND_RESUME_AUDIT.json');checkpoints=read(RESULT/'CHECKPOINT_MANIFESTS.json')
    ledger=[json.loads(l) for l in (RESULT/'MATCHED_BATCH_LEDGER.jsonl').read_text().splitlines()]
    log=[json.loads(l) for l in (RESULT/'TRAINING_H.jsonl').read_text().splitlines()]
    terminal=read(RESULT/'CONTINUATION_MANIFEST.json')
    assert len(log)==len(ledger)==286 and training['passed'] and all(training['checks'].values())
    assert sum(x['target_count'] for x in log)==149946368
    for key,ref in [('batch_sha256','logical_global_batch_sha256'),('stream_sha256','logical_global_stream_sha256'),('pass_count','pass_count')]:
        assert [x[key] for x in log]==[x[ref] for x in ledger]
    assert all(all(x['optimizer_checks'].values()) and all(x['pre_forward_invariants'].values()) and x['end_cursor_exact'] for x in log)
    assert checkpoints['H']['verified_independently'] and checkpoints['H']['verification']['strict_reopen']['passed']
    assert checkpoints['H']['sha256']==ev['H']['checkpoint_sha256']
    for key in ('next_global_batch_sha256','next_stream_sha256','final_loader_cursor'):assert training[key]==terminal[key]
    assert sha(RESULT/'MATCHED_BATCH_LEDGER.jsonl')=='0875d5533a4a8ae753f2e0aec661d81f314609c16d4de053a6ffd48df8e751ec'
    assert [x['global_update'] for x in log]==list(range(2482,2768))
    assert log[-1]['experiment_total_update']==477 and log[-1]['experiment_total_targets']==250085376
    assert preflight['authorized'] and preflight['complete_resume_checks_passed']
    assert all(source['H']['checks'].values()) and all(source['D_control']['checks'].values())
    files={'H250M':checkpoints['H']}
    for name,folder,arm,count,expected,persistent in (
        ('H100M','experiment_2d10_retrieval_aware_gating_100m','H','001300758528','d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c','exp2d10_retrieval_gating_100m'),
        ('D250M','experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m','D','001450704896','9714b2e3f53a8c15dfecfed3e9b56c358176c1f9f609bcce7e28c35b8a358a9b','exp2d9_dynamic_gating_250m')):
        path=ARCHIVE.parent/folder/arm/f'scientific_cumulative_{count}.pt'
        assert sha(path)==expected
        files[name]={'local_checkpoint':str(path),'persistent_checkpoint':f'/workspace/{persistent}/run/checkpoints/{arm}/{path.name}','sha256':expected}
    assert files['D250M']['sha256']==ev['D']['checkpoint_sha256']
    write('ALL_CHECKPOINT_IDENTITIES.json',files)
    changes=parameter_changes(files['H100M']['local_checkpoint'],files['H250M']['local_checkpoint']);write('H_PARAMETER_CHANGES.json',changes)
    stats=bootstrap(losses);write('PAIRED_BOOTSTRAP.json',stats)
    gates=gate_statistics();write('GATE_STATISTICS.json',gates)
    old=read(OLD/'PAIRED_BOOTSTRAP.json')['contrasts']['D-H']
    comparison={'old_D_minus_H_mean':old['mean'],'old_raw_95_ci':old['raw_95_ci'],
        'old_adjusted_98_333333_ci':old['adjusted_98_333333_ci'],'current_D_minus_H_mean':stats['mean'],
        'descriptive_mean_change':stats['mean']-old['mean'],
        'limitation':'Different panels: not a paired test of growth; no pooling of losses.'}
    oldg=read(OLD/'GATE_STATISTICS.json')['arms']['H']
    comparison['gate_distribution_changes']={b:{k:{'H100M':oldg[b]['metrics'][k]['eligible']['mean'],
        'H250M':gates['arms']['H'][b]['metrics'][k]['eligible']['mean']} for k in ('lambda_R','logit_difference','entropy')} for b in ('B1','B3','B5')}
    write('DESCRIPTIVE_100M_COMPARISON.json',comparison)
    adopt=stats['flags']['beyond_margin']
    classification=('H CLEARS PRACTICAL MARGIN' if adopt else 'PRACTICAL EQUIVALENCE' if stats['flags']['practical_equivalence'] else 'H STATISTICALLY BETTER WITHOUT PRACTICAL-MARGIN CLEARANCE' if stats['flags']['positive'] else 'H HARMFUL' if stats['flags']['negative'] else 'UNRESOLVED')
    decision={'classification':classification,'preferred_architecture':'H' if adopt else 'Dynamic',
        'replace_Dynamic':adopt,'integrity_checks_passed':True,'rule':'lower ordinary 95% paired CI(D-H) > 0.0001 AND all integrity checks pass',
        'recommendation':'If separately authorized, compare the preferred architecture against a from-scratch GPT-2 baseline under a new matched protocol.',
        'further_experiment_launched':False}
    write('ADOPTION_DECISION.json',decision)
    stopped=dt.datetime.fromisoformat(stop['verified_at_utc']);supervised=dt.datetime.fromisoformat(read(RESULT/'STOP_CAPABILITY_PREFLIGHT.json')['observed_at_utc'])
    resumed=dt.datetime(2026,9,5,11,1,52,tzinfo=dt.timezone.utc)
    runtime={'H_continuation_training_seconds':training['training_wall_seconds'],
        'evaluation_seconds':{a:ev[a]['wall_seconds'] for a in 'DH'},
        'historical_D_continuation_training_seconds':read(HIST/'TRAINING_COMPLETE_D.json')['training_wall_seconds'],
        'pod_resume_to_verified_stop_seconds':(stopped-resumed).total_seconds(),
        'pod_supervised_seconds':(stopped-supervised).total_seconds(),
        'aggregate_GPU_hours':(stopped-resumed).total_seconds()/3600,'GPU_count':1,
        'verified_stopped_utc':stopped.isoformat(),'resumed_utc':resumed.isoformat(),
        'parameter_counts':{a:ev[a]['parameter_count'] for a in 'DH'},'H_extra_parameters_vs_D':219174,
        'H_extra_FP32_bytes_vs_D':876696,'persistent_bytes_per_sequence':33289728,'persistent_state_delta':0,
        'note':'D training time is historical. H evaluation includes gate diagnostics. These timings do not isolate production inference overhead.'}
    write('MEMORY_AND_RUNTIME.json',runtime)
    audit={'passed':True,'scientific_implementation_commit':preflight['git_commit'],
        'frozen_implementation_sha256':preflight['implementation_sha256'],'additional_updates':286,'additional_targets':149946368,
        'total_adaptation_updates':477,'total_adaptation_targets':250085376,'final_global_update':2767,'final_cumulative_targets':1450704896,
        'all_286_batches_streams_cursors_passes_and_optimizer_checks_exact':True,
        'three_pass_updates':[x['global_update'] for x in log if x['pass_count']==3],
        'H_parent_and_D_control_verified':True,'strict_checkpoint_reopens':True,'terminal_equality_with_D250M':True,
        'all_active_counters_advanced_286':True,'complete_retired_state_unchanged':True,'router_counters_477':True,
        'same_fresh_panel_two_evaluations':True,'panel_sha256':panel['panel_sha256'],
        'unchanged_prescribed_persistent_state':True,'single_primary_ordinary_95CI':True,
        'independent_local_and_persistent_checkpoint_and_artifact_hashes_verified':True,
        'pod_stopped_verified':True,'persistent_volume_retained':True,
        'statistics_started_unix':started,'statistics_after_verified_stop':started>stopped.timestamp(),
        'flag_column_order':list(FLAG_COLUMNS),'sealed_100M_numerical_artifacts_untouched':True,
        'sequence_loss_archive_sha256':sha(RESULT/'PAIRED_SEQUENCE_LOSSES.npz'),'no_further_experiment_launched':True}
    assert audit['statistics_after_verified_stop'] and runtime['parameter_counts']=={'D':124478212,'H':124697386}
    write('FINAL_AUDIT.json',audit)
    report(ev,stats,gates,decision,runtime,files,comparison,audit)
    print(json.dumps({'statistics':stats,'decision':decision},indent=2),flush=True)

def report(ev,stats,gates,decision,runtime,files,comparison,audit):
    lo,hi=stats['ci_95']
    lines=['# EXPERIMENT 2D10 — H-ONLY 250M CONTINUATION COMPLETE','',
        f"**{decision['classification']}. Preferred tested 250M architecture: {decision['preferred_architecture']}.**",
        f"H {'should' if decision['replace_Dynamic'] else 'should not'} replace Dynamic as the preferred baseline under the prespecified adoption rule.",'',
        'H only: 286 additional updates / 149,946,368 additional targets; 477 updates / 250,085,376 total adaptation targets. One reused 250M Dynamic control; exactly two final evaluations; no further experiment launched.','',
        '| Condition | CE | Perplexity |','|---|---:|---:|']
    for a in 'DH':lines.append(f"| {a}250M | {ev[a]['aggregate_ce']:.12f} | {ev[a]['perplexity']:.12f} |")
    lines+=['',f"D−H = **{stats['mean']:+.12f}**, ordinary paired 95% CI **[{lo:+.12f}, {hi:+.12f}]**. Positive favors H. exp(D−H) = {stats['exp_contrast']:.12f}.",
        f"Sequence wins: H {stats['H_wins']}; D {stats['D_wins']}; ties {stats['ties']}.",'',
        '| Contrast | Positive L>0 | Beyond margin L>δ | Negative U<0 | Material harm U<−δ | Practical equivalence | H noninferiority |',
        '|---|---|---|---|---|---|---|','| D−H | '+render_flag_values(stats['flags'])+' |','',
        'One prespecified primary comparison; 50,000 paired sequence bootstrap resamples, isolated default_rng(20260913), NumPy '+stats['numpy_version']+', linear 2.5/97.5 percentiles. Margin δ=0.0001. Bounds must strictly cross thresholds; touching does not pass. Adoption additionally requires every integrity and prescribed-state check to pass.','',
        f"The 100M D−H mean was {comparison['old_D_minus_H_mean']:+.12f}, raw 95% CI {comparison['old_raw_95_ci']}, adjusted 98.333333% CI {comparison['old_adjusted_98_333333_ci']}. The descriptive mean change is {comparison['descriptive_mean_change']:+.12f}. These are different panels: this is not a paired test of growth, and losses were not pooled.",'',
        'The old 100M report’s flag-table ordering bug is corrected in the new renderer by explicit keys, with a JSON-roundtrip test. The sealed old report, numerical artifacts and immutable tag remain unchanged. Its D−H/T−H beyond-margin flags are True and material-harm flags False; see 100M_REPORT_ERRATUM.md.','',
        '## H gates and measured costs','',
        '| Destination | b2 | Logit difference mean | λL mean | λR mean / std / range | BF16 λR mean | Entropy mean |',
        '|---|---|---:|---:|---|---:|---:|']
    for b,i in zip(('B1','B3','B5'),('0','2','4')):
        m=gates['arms']['H'][b]['metrics'];r=m['lambda_R']['eligible']
        lines.append(f"| {b} | {ev['H']['H_output_biases'][i]} | {m['logit_difference']['eligible']['mean']:.8f} | {m['lambda_L']['eligible']['mean']:.8f} | {r['mean']:.8f} / {r['std']:.8f} / [{r['min']:.8f}, {r['max']:.8f}] | {m['cast_lambda_R']['eligible']['mean']:.8f} | {m['entropy']['eligible']['mean']:.8f} |")
    lines+=['','| Destination | H100M λR mean | H250M λR mean |','|---|---:|---:|']
    for b in ('B1','B3','B5'):
        v=comparison['gate_distribution_changes'][b]['lambda_R']
        lines.append(f"| {b} | {v['H100M']:.8f} | {v['H250M']:.8f} |")
    lines+=['','These gate-distribution comparisons use different panels and are descriptive.','',
        '| Router tensor | H100M norm | H250M norm | Change norm |','|---|---:|---:|---:|']
    changes=read(RESULT/'H_PARAMETER_CHANGES.json')['router_parameters']
    for name,v in changes.items():
        lines.append(f"| {name} | {v['source_norm']:.8f} | {v['final_norm']:.8f} | {v['change_norm']:.8f} |")
    lines+=['','Gate table covers eligible-memory positions. GATE_STATISTICS.json contains separate unavailable positions, FP32 and actual BF16 coefficients, full-panel moments/extrema and quantiles from at most 131,072 eligible positions per destination using isolated seed20260914. Unavailable memory is exactly local-only. Router norms and b2 are in H_REAL.json; H_PARAMETER_CHANGES.json records H100M-to-H250M tensor changes. DESCRIPTIVE_100M_COMPARISON.json describes gate distributions across different panels.','',
        'H registers 124,697,386 parameters; D registers 124,478,212. H adds 219,174 parameters / 876,696 FP32 bytes. Both retain exactly 33,289,728 BF16 persistent bytes per B=1 sequence; state delta is zero. No geometry or attention/router kernel changed.','',
        f"H continuation training: {runtime['H_continuation_training_seconds']/60:.2f} minutes. D evaluation: {runtime['evaluation_seconds']['D']/60:.2f} minutes. H evaluation including diagnostics: {runtime['evaluation_seconds']['H']/60:.2f} minutes. Historical D continuation training: {runtime['historical_D_continuation_training_seconds']/60:.2f} minutes.",
        f"One-GPU pod interval from resume to verified stop: {runtime['pod_resume_to_verified_stop_seconds']/60:.2f} minutes / {runtime['aggregate_GPU_hours']:.3f} GPU-hours. These workload timings do not isolate production inference overhead.",'',
        '## Integrity and durable outputs','',
        'All 286 logical batches, streams, loader cursors, target counts and pass counts match the historical D continuation. There are 277 two-pass and 9 three-pass updates. H reaches global2767 and cumulative1,450,704,896 targets, with terminal cursor and next batch/stream identical to D250M. Active Adam counters advance286; router counters reach477. All four retired/dormant gate tensors, moments and counters remain unchanged. Complete trained H state was restored without fresh router initialization; strict checkpoint reopen passed.','',
        'The fresh 4096×1024-target panel is '+audit['panel_sha256']+'. Both evaluations used true incremental BF16 execution, FP32 token CE and FP64 accumulation, with cache resets and identical ordered sequences. D tensors and file are unchanged. Panel and recoverable historical target spans are disjoint.','']
    for name,v in files.items():
        lines += [f"**{name}** SHA256 `{v['sha256']}`",'',f"- Local: `{v['local_checkpoint']}`",f"- Persistent: `{v['persistent_checkpoint']}`",'']
    lines+=['Final checkpoint and all required GPU artifacts have independent local/persistent hash verification. FINAL_AUDIT.json, CHECKPOINT_MANIFESTS.json, ARTIFACT_BACKUP_VERIFICATION.json and STOP_VERIFICATION.json record the evidence.',
        f"Pod nagj1hv18p3z2c is verified EXITED/stopped at {runtime['verified_stopped_utc']}; volume yhzyb27fb5 is retained. Statistics began after verified shutdown.",'',
        'Scientific implementation commit: `'+audit['scientific_implementation_commit']+'`. Result branch: `codex/experiment-2d10-h-250m`; final immutable tag: `experiment-2d10-h-250m-final`. Git push verification is retained alongside the local operations archive.','',
        '## Interpretation and stopping point','',
        'This compares complete trained architectures. It does not isolate retrieved-output inputs, token variation, mean scaling or initialization; no same-checkpoint router-ablation claim is available. Evaluation-sequence uncertainty is not training-seed replication and does not establish superiority to an independently trained Karpathy GPT-2 baseline.','',
        decision['recommendation'],'No further training, window change, extra evaluation panel or from-scratch comparison was launched.']
    (RESULT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
