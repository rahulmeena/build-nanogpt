"""CPU analysis of the sealed 250M continuation, using the original decision rules."""
import argparse
import json
import math
import pathlib
import numpy as np
from experiment_2d9_analysis import (CONDITIONS, DELTA, bootstrap, flags, classify,
    gate_stats, gate_detail_tables, sha, canonical, read, write)
SOURCE_SHA256 = {'S':'676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b',
                'D':'c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44'}
POD='7kk5yyti00rnrp'
THREE_PASS=[2496,2528,2560,2592,2624,2656,2688,2720,2752]


def rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--prior-results',required=True)
    parser.add_argument('--gate-scalars',required=True)
    args=parser.parse_args()
    out=pathlib.Path(args.results);old=pathlib.Path(args.prior_results)
    stop=read(out/'STOP_VERIFICATION.json')
    assert stop['passed'] and stop['pod']['id']==POD and stop['pod']['runtimeStatus']=='stopped' and stop['pod']['desiredStatus']=='EXITED'
    previous=read(old/'SCIENTIFIC_RESULT_SUMMARY.json')
    source=read(out/'SOURCE_AND_PARAMETER_AUDIT.json');preflight=read(out/'PREFLIGHT_AUDIT.json')
    panel=read(out/'EVALUATION_PANEL_MANIFEST.json');disjoint=read(out/'DISJOINTNESS_AUDIT.json')
    continuation=read(out/'CONTINUATION_MANIFEST.json');ledger=rows(out/'MATCHED_BATCH_LEDGER.jsonl')
    checkpoints=read(out/'CHECKPOINT_MANIFESTS.json');operations=read(out/'OPERATIONS_STATUS.json')
    training={a:read(out/f'TRAINING_COMPLETE_{a}.json') for a in ('S','D')}
    logs={a:rows(out/f'TRAINING_{a}.jsonl') for a in ('S','D')}
    evaluations={n:read(out/(n+'.json')) for n in CONDITIONS}
    arrays=np.asarray([evaluations[n]['per_sequence_ce'] for n in CONDITIONS],dtype=np.float64)
    assert arrays.shape==(3,4096) and np.isfinite(arrays).all()
    paired=bootstrap(arrays);write(out/'PAIRED_BOOTSTRAP.json',paired)
    parent_gates={b:r['g0'] for b,r in previous['gates']['destinations'].items()}
    gates=gate_stats(args.gate_scalars,evaluations['DYNAMIC_REAL']['model_gate_parameters'],parent_gates)
    for b,r in gates['destinations'].items():
        old_gate=previous['gates']['destinations'][b]
        r['parent_stage']='sealed 100M Dynamic checkpoint'
        r['source_100m_w_norm']=old_gate['w_norm']
        r['raw_g0_change_from_100m']=r['g0']-old_gate['g0']
        r['w_norm_change_from_100m']=r['w_norm']-old_gate['w_norm']
    write(out/'GATE_STATISTICS.json',gates)
    comparison={'scope':'Descriptive contrasts on different disjoint panels; no pooled losses or paired effect-growth test.',
        'prior_panel_sha256':read(old/'EVALUATION_PANEL_MANIFEST.json')['panel_sha256'],'current_panel_sha256':panel['panel_sha256'],
        'contrasts':{n:{'100m':previous['paired']['contrasts'][n],'250m':paired['contrasts'][n],
            'descriptive_mean_change':paired['contrasts'][n]['mean']-previous['paired']['contrasts'][n]['mean']} for n in ('A','P','R')}}
    write(out/'DESCRIPTIVE_100M_250M_COMPARISON.json',comparison)
    memory={'persistent_state_by_condition':{n:evaluations[n]['persistent_state'] for n in CONDITIONS},
        'persistent_state_delta_bytes':evaluations['DYNAMIC_REAL']['persistent_state']['logical_bytes_per_sequence']-evaluations['STATIC_REAL']['persistent_state']['logical_bytes_per_sequence'],
        'parameter_counts':{'S':124475908,'D':124478212},'parameter_increase':2304,'additional_fp32_parameter_bytes':9216,'additional_bf16_parameter_bytes':4608,
        'training_wall_seconds':{a:training[a]['training_wall_seconds'] for a in ('S','D')},
        'evaluation_wall_seconds':{n:evaluations[n]['wall_seconds'] for n in CONDITIONS},
        'timing_scope':'Descriptive workload timings. Dynamic Real includes gate collection; training update times include audits and exclude recovery-checkpoint writing. No isolated timing benchmark.',
        'training_only_state':'Adam state is training-only. Gate scalar arrays are analysis artifacts, not persistent model state.'}
    write(out/'MEMORY_AND_RUNTIME.json',memory)
    names={a:{r['name'] for r in source[a]['parameter_inventory']} for a in ('S','D')}
    chain=canonical(SOURCE_SHA256);chain_ok=True
    for row in ledger:
        copy=dict(row);digest=copy.pop('chain_sha256')
        chain_ok=chain_ok and copy['previous_chain_sha256']==chain and canonical(copy)==digest
        chain=digest
    old_panel=read(old/'EVALUATION_PANEL_MANIFEST.json')
    test=read(out/'TARGETED_RESUME_TESTS.json')
    checks={
        'correct_sealed_source_per_arm':all(source[a]['source_checkpoint_sha256']==SOURCE_SHA256[a]==previous['checkpoints'][a]['sha256'] and all(source[a]['checks'].values()) for a in ('S','D')),
        'exact_model_optimizer_moments_and_steps_restored':all(all(source[a]['checks'][k] for k in ('all_model_tensors_exact','optimizer_names_exact','optimizer_groups_exact','optimizer_state_exact','individual_steps_exact','rng_exact','loader_exact')) for a in ('S','D')),
        'learned_D_vectors_and_moments_retained':source['D']['checks']['learned_dynamic_vectors_retained'] and source['D']['checks']['dynamic_steps_191'] and source['D']['checks']['optimizer_state_exact'],
        'unchanged_parameter_inventory_and_counts':names['D']-names['S']=={'w_B1','w_B3','w_B5'} and names['S']<=names['D'] and all(sum(math.prod(r['shape']) for r in source[a]['parameter_inventory'])==n for a,n in [('S',124475908),('D',124478212)]),
        'tied_weights_and_fp32_masters':all(source[a]['checks']['tied_weights'] and source[a]['checks']['master_parameters_fp32'] for a in ('S','D')),
        'unchanged_attention_and_gate_kernels':all(preflight['unchanged_kernel_checks'].values()) and preflight['inherited_implementation_evidence_passed'] and preflight['inherited_100m_audit_sha256']==sha(old/'FINAL_AUDIT.json'),
        'inherited_zero_effect_and_causality_evidence':previous['audit']['passed'] and all(previous['audit']['checks'][k] for k in ('zero_effect_FP32_BF16_parallel_incremental','causality_row_isolation_and_roundtrip_tests','first_backward_all_new_vectors_finite_nonzero')),
        'targeted_resume_tests':preflight['targeted_resume_tests_passed'] and test['returncode']==0 and '6 passed' in test['stdout'],
        'full_source_roundtrips_without_updates':all(r['passed'] and r['scientific_updates_performed']==0 and r['strict_reopen']['passed'] and all(r['checks'].values()) for r in preflight['source_roundtrips'].values()),
        'committed_implementation_bound_before_training':all(training[a]['checks']['code_git_commit'] and training[a]['checks']['code_implementation'] for a in ('S','D')),
        'correct_continuation_start':continuation['passed'] and continuation['starting_global_update']==2481 and continuation['first_global_update']==2482 and continuation['first_global_batch_sha256']=='400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649',
        'ledger_hash_and_chain':sha(out/'MATCHED_BATCH_LEDGER.jsonl')==continuation['ledger_sha256'] and chain_ok and chain==continuation['terminal_chain_sha256'],
        'exact_286_additional_updates':all(len(logs[a])==286 and [r['global_update'] for r in logs[a]]==list(range(2482,2768)) for a in ('S','D')),
        'exact_149946368_additional_targets':all(sum(r['target_count'] for r in logs[a])==149946368 and logs[a][-1]['new_targets']==149946368 for a in ('S','D')),
        'exact_477_updates_250085376_total_targets':all(logs[a][-1]['experiment_total_update']==477 and logs[a][-1]['experiment_total_targets']==250085376 and logs[a][-1]['cumulative_targets']==1450704896 for a in ('S','D')),
        'unambiguous_per_update_counters':all(r['continuation_local_update']==i and r['experiment_total_update']==191+i and r['experiment_total_targets']==100139008+i*524288 for arm in ('S','D') for i,r in enumerate(logs[arm],1)),
        'same_matched_logical_batches':all([r['batch_sha256'] for r in logs[a]]==[r['logical_global_batch_sha256'] for r in ledger] for a in ('S','D')),
        'same_matched_logical_streams':all([r['stream_sha256'] for r in logs[a]]==[r['logical_global_stream_sha256'] for r in ledger] for a in ('S','D')),
        'exact_pass_cadence':all([r['global_update'] for r in logs[a] if r['pass_count']==3]==THREE_PASS and sum(r['pass_count']==2 for r in logs[a])==277 for a in ('S','D')),
        'all_gradient_and_optimizer_checks':all(all(g['finite'] for g in r['active_gradient_groups'].values()) and math.isfinite(r['gradient_norm_before_clip']) and all(r['optimizer_checks'].values()) for records in logs.values() for r in records),
        'all_update_replay_invariants':all(all(r['pre_forward_invariants'].values()) and r['end_cursor_exact'] for records in logs.values() for r in records),
        'dormant_B6_unchanged':all(r['optimizer_checks']['dormant_optimizer_unchanged'] and r['optimizer_checks']['dormant_parameter_unchanged'] for records in logs.values() for r in records),
        'complete_recovery_checkpoints':all(r['local_updates']==144 and r['strict_reopen']['passed'] and all(r['strict_reopen']['checks'].values()) for r in read(out/'RECOVERY_CHECKPOINTS.json').values()),
        'strict_final_reopen':all(checkpoints[a]['verification']['strict_reopen']['passed'] for a in ('S','D')),
        'every_active_counter_advanced_286':all(checkpoints[a]['verification']['strict_reopen']['checks']['optimizer_progression'] for a in ('S','D')),
        'dynamic_w_counters_reach_477':checkpoints['D']['verification']['strict_reopen']['checks']['dynamic_steps'],
        'exact_final_snapshots_and_scheduler':all(all(checkpoints[a]['verification']['strict_reopen']['checks'][k] for k in ('model_snapshot_exact','optimizer_snapshot_exact','optimizer_groups_exact','scheduler')) for a in ('S','D')),
        'terminal_cursor_equality':training['S']['final_loader_cursor']==training['D']['final_loader_cursor']==continuation['final_loader_cursor'],
        'terminal_next_batch_equality':training['S']['next_global_batch_sha256']==training['D']['next_global_batch_sha256']==continuation['next_global_batch_sha256'],
        'terminal_next_stream_equality':training['S']['next_stream_sha256']==training['D']['next_stream_sha256']==continuation['next_stream_sha256'],
        'one_fresh_panel_seed_20260907':panel['candidate_panels_constructed']==1 and panel['selection_seed']==20260907 and panel['sealed_before_training_and_scoring'] and disjoint['passed'],
        'all_historical_exclusions_and_100m_panel':all(disjoint['checks'][k] for k in ('no_historical_overlap','recent_panels_recovered','sealed_100m_panel_excluded')) and all(not any(max(a,c)<min(b,d) for c,d in old_panel['canonical_target_spans_half_open']) for a,b in panel['canonical_target_spans_half_open']),
        'exactly_three_final_conditions':{p.name for p in pathlib.Path(args.gate_scalars).parent.glob('*.json')}=={n+'.json' for n in CONDITIONS} and all(evaluations[n]['condition']==n for n in CONDITIONS),
        'three_finite_4096_loss_arrays':arrays.shape==(3,4096) and bool(np.isfinite(arrays).all()) and all(evaluations[n]['targets']==4194304 and evaluations[n]['passed'] for n in CONDITIONS),
        'ordered_panel_identity_equality':all(evaluations[n]['sequence_identities']==panel['sequence_identities'] and evaluations[n]['panel_sha256']==panel['panel_sha256'] and evaluations[n]['panel_manifest_sha256']==sha(out/'EVALUATION_PANEL_MANIFEST.json') for n in CONDITIONS),
        'same_D_checkpoint_for_staticized':evaluations['DYNAMIC_REAL']['checkpoint_sha256']==evaluations['DYNAMIC_STATICIZED']['checkpoint_sha256']==checkpoints['D']['sha256'],
        'S_evaluation_from_S_checkpoint':evaluations['STATIC_REAL']['checkpoint_sha256']==checkpoints['S']['sha256'],
        'evaluation_weights_and_checkpoint_unchanged':all(evaluations[n]['checkpoint_file_unchanged'] and evaluations[n]['model_tensors_unchanged'] for n in CONDITIONS),
        'paired_50k_bootstrap_original_seed':paired['resamples']==50000 and paired['seed']==20260906 and paired['shared_indices_across_contrasts'] and abs(paired['decomposition_residual'])<1e-14,
        'zero_persistent_state_growth':all(all(v==33289728 for v in evaluations[n]['persistent_state'].values()) for n in CONDITIONS),
        'full_panel_gate_scalars_independently_verified':gates['token_count_per_destination']==4194304 and gates['array_sha256']==read(out/'GATE_SCALAR_EXPORT.json')['sha256'] and read(out/'GATE_SCALAR_EXPORT.json')['independently_verified'],
        'inherited_microbatch_accumulation_and_LRs':continuation['microbatch']==32 and continuation['gradient_accumulation']==16 and all(all(v==(3e-5 if n.startswith('base_') or n=='dynamic_nodecay' else 3e-4) for n,v in r['lr'].items()) for records in logs.values() for r in records),
        'independently_verified_final_backups':all(checkpoints[a]['verified_independently'] and sha(checkpoints[a]['local_checkpoint'])==checkpoints[a]['sha256']==checkpoints[a]['persistent_sha256'] for a in ('S','D')),
        'verified_exact_GPU_stop':stop['passed'] and stop['pod']['id']==POD and stop['pod']['runtimeStatus']=='stopped' and stop['pod']['desiredStatus']=='EXITED',
        'persistent_volume_retained':stop['persistent_volume_retained'] and stop['pod']['networkVolumeId']=='yhzyb27fb5',
        'successful_workflow_and_stop':operations['outcome']=='gpu_work_complete' and operations['workflow_exit_code']==0 and not operations['error'] and not operations['stop_error'],
        'training_terminal_audits':all(training[a]['passed'] and all(training[a]['checks'].values()) for a in ('S','D')),
    }
    final={'passed':all(checks.values()),'checks':checks,'passed_count':sum(checks.values()),'total_checks':len(checks),
        'implementation_commit':preflight['git_commit'],'inherited_100m_commit':'482ad55637c2a0adb5c7c268b37c7be243ac15c8'}
    write(out/'FINAL_AUDIT.json',final)
    label,preferred=classify(paired['contrasts']['A'],paired['contrasts']['P'],final['passed'],memory['persistent_state_delta_bytes'])
    summary={'classification':label,'preferred_architecture':preferred,'aggregate_ce':{n:evaluations[n]['aggregate_ce'] for n in CONDITIONS},
        'perplexity':{n:evaluations[n]['perplexity'] for n in CONDITIONS},'paired':paired,'gates':gates,'memory_and_runtime':memory,
        'comparison_100m_250m':comparison,'source_checkpoint_sha256':SOURCE_SHA256,'checkpoints':checkpoints,
        'terminal':{'loader_cursor_sha256':canonical(continuation['final_loader_cursor']),'next_batch_sha256':continuation['next_global_batch_sha256'],
            'next_stream_sha256':continuation['next_stream_sha256'],'equal':checks['terminal_cursor_equality'] and checks['terminal_next_batch_equality'] and checks['terminal_next_stream_equality']},
        'audit':final,'stop':stop}
    write(out/'SCIENTIFIC_RESULT_SUMMARY.json',summary)
    (out/'EXPERIMENT_2D9_250M_FINAL_REPORT.md').write_text(report(summary,preflight,panel))
    print(json.dumps({'classification':label,'preferred':preferred,'ce':summary['aggregate_ce'],'contrasts':paired['contrasts'],'audit':final},indent=2))
    if not final['passed']:raise SystemExit('final integrity audit failed')


# Report rendering is below; it does not alter the statistical decisions.
def report(s,preflight,panel):
    status='COMPLETE' if s['audit']['passed'] else 'INCOMPLETE'
    lines=[f'# EXPERIMENT 2D9 — 250M MATCHED CONTINUATION {status}','',
        f"Primary classification: **{s['classification']}**",f"Preferred architecture recommendation: **{s['preferred_architecture']}**",'',
        '| Condition | CE | Perplexity |','|---|---:|---:|']
    for n in CONDITIONS:lines.append(f"| {n} | {s['aggregate_ce'][n]:.12f} | {s['perplexity'][n]:.12f} |")
    lines+=['','| Contrast | Mean CE | 95% paired CI | exp(contrast) | Second / first wins / ties |','|---|---:|---|---:|---|']
    for name,r in s['paired']['contrasts'].items():
        lines.append(f"| {name}: {r['contrast']} | {r['mean']:+.12f} | [{r['ci95'][0]:+.12f}, {r['ci95'][1]:+.12f}] | {r['exp_contrast']:.12f} | {r['second_condition_sequence_wins']} / {r['first_condition_sequence_wins']} / {r['ties']} |")
    lines+=['','delta_CE = 0.0001. A measures the matched architecture benefit. P measures the inference benefit of the complete learned w term on D weights. R is descriptive; A = R + P. Positive contrasts favor the second named condition.',
        '','| Primary contrast | Positive utility | Beyond margin | Negative utility | Material harm | Equivalent | Second condition noninferior |','|---|---|---|---|---|---|---|']
    for name in ('A','P'):
        f=s['paired']['contrasts'][name]['flags']
        lines.append('| '+name+' | '+' | '.join(str(f[k]) for k in ('positive_utility','benefit_beyond_margin','negative_utility','harm_beyond_margin','practical_equivalence','noninferiority_of_second_named_condition'))+' |')
    lines+=['','All flags use the original strict inequalities. An interval touching a boundary does not clear it. Architecture adoption requires A lower CI > +0.0001 and P lower CI > 0, passing integrity, and zero persistent-state growth.',
        '','## Descriptive comparison with 100M','',
        '| Contrast | 100M mean [95% CI] | 250M mean [95% CI] | Descriptive mean change |','|---|---|---|---:|']
    for name,row in s['comparison_100m_250m']['contrasts'].items():
        a,b=row['100m'],row['250m']
        lines.append(f"| {name} | {a['mean']:+.12f} [{a['ci95'][0]:+.12f}, {a['ci95'][1]:+.12f}] | {b['mean']:+.12f} [{b['ci95'][0]:+.12f}, {b['ci95'][1]:+.12f}] | {row['descriptive_mean_change']:+.12f} |")
    lines+=['','The panels differ and are disjoint. These mean changes are descriptive, not a paired test of effect growth. Losses were not pooled, and absolute CEs across stages are not interpreted as a learning curve. Intervals quantify evaluation-sequence uncertainty for these continued training trajectories, not replication across training seeds. A and P share Dynamic and are not independent replications.',
        '','## Learned gates','',
        '| Destination | 100M raw g0 | 250M raw g0 | Change | 100M ‖w‖₂ | 250M ‖w‖₂ | Change |','|---|---:|---:|---:|---:|---:|---:|']
    for b,r in s['gates']['destinations'].items():
        lines.append(f"| {b} | {r['parent_raw_g0']:.9f} | {r['g0']:.9f} | {r['raw_g0_change_from_100m']:+.9f} | {r['source_100m_w_norm']:.9f} | {r['w_norm']:.9f} | {r['w_norm_change_from_100m']:+.9f} |")
    lines+=['','| Destination | Final tanh(g0) | FP32 gate mean / std / range | BF16 coefficient mean / std / range |','|---|---:|---|---|']
    for b,r in s['gates']['destinations'].items():
        f=r['mathematical_fp32_gate'];q=r['effective_bf16_coefficient']
        lines.append(f"| {b} | {r['tanh_g0']:.9f} | {f['mean']:.9f} / {f['std']:.9f} / [{f['min']:.9f}, {f['max']:.9f}] | {q['mean']:.9f} / {q['std']:.9f} / [{q['min']:.9f}, {q['max']:.9f}] |")
    lines+=['','B1 gates use token embedding plus position embedding; B3/B5 additionally contain preceding contextual processing. Gates use the current pre-attention residual with FP32 RMS epsilon 1e-5, dot product and tanh, then cast the coefficient to the attention dtype. The attention and gating kernels are unchanged from the sealed 100M experiment.',
        '','All statistics use the 4,194,304 positions per destination. Quantiles are exact full-panel linear percentiles, with no sampling. Across-token standard deviation, standard deviation of per-sequence means, and mean within-sequence standard deviation all use population definitions (ddof=0). Intrinsic gates are also reported at positions without eligible recurrent memory, where they scale zero recurrent output. B1 first has eligible memory at position 1, B3 at 31, and B5 at 63.',
        *gate_detail_tables(s['gates']),
        '','Gate scalars were collected during DYNAMIC_REAL with at most one batch on-device and one CPU transfer per batch. No attention matrices or extra evaluation condition were collected. Nonzero w and gate variance alone are not evidence of usefulness.',
        '','Staticization removes the entire learned w term, including its average shift and token variation. A positive P establishes that term’s inference utility; it does not establish superiority to every optimally refitted constant gate.',
        '','## Matched continuation and provenance','',
        '**286 additional updates / 149,946,368 additional targets per arm; 477 updates / 250,085,376 total 2D9 targets per arm.**',
        '','Each arm resumed its own sealed 100M checkpoint. S and D model/optimizer tensors were not required to equal one another. Both sources restored their exact model, optimizer groups and moments, individual Adam counters, scheduler metadata, RNG states, and shared loader cursor. D’s learned w vectors and their moments were preserved.',
        '','Continuation global updates: 2482–2767 inclusive. Source inherited target counter: 1,300,758,528. Final inherited target counter: 1,450,704,896. B32×T1024, accumulation16 and 524,288 targets/update were unchanged. The stage used 277 two-pass updates and 9 three-pass updates, at 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720 and 2752, with inherited loss weights.',
        '','Every active parameter’s Adam counter advanced 286 from its own source value; D’s w counters reached 477. Dormant B6 parameters and optimizer state were unchanged. Base and w LR remained 3e-5, scalar-gate LR 3e-4, and w weight decay 0. Warmup was not restarted.',
        '',f"Committed continuation implementation: `{preflight['git_commit']}`",'Sealed 100M result commit: `482ad55637c2a0adb5c7c268b37c7be243ac15c8`',
        '',f"Fresh panel SHA: `{panel['panel_sha256']}`",f"Panel selection seed: {panel['selection_seed']}. Exactly 64 canonical B64 validation batches, 4096 sequences and 4,194,304 targets per condition. All historical exclusions and the sealed 100M panel were applied before continuation training or scoring.",
        '','Exactly three final conditions used true incremental inference, BF16 execution, FP32 token CE and FP64 accumulation. All model/cache state resets between sequence batches and conditions. Staticized generated its entire trajectory with the w term omitted, using D’s final g0 and base weights; it did not reuse Dynamic-Real caches or mutate/refit the checkpoint.',
        '','CPU analysis used 50,000 paired sequence-level bootstrap resamples, RNG seed 20260906, shared indices across A/P/R, and 95% NumPy linear-percentile intervals. The original ordered classification and adoption rules were applied unchanged.',
        '','## Memory, runtime, checkpoints and shutdown','',
        '**Persistent state: 33,289,728 BF16 bytes per B=1 sequence for all conditions; delta 0.** Dynamic still adds only 2,304 parameters: 9,216 FP32 bytes, or 4,608 bytes if stored in BF16. Optimizer state is training-only; gate arrays are analysis artifacts.',
        '','| Workload | Minutes |','|---|---:|']
    for name,seconds in {**s['memory_and_runtime']['training_wall_seconds'],**s['memory_and_runtime']['evaluation_wall_seconds']}.items():
        lines.append(f'| {name} | {seconds/60:.2f} |')
    lines+=['','Timings describe these workloads. Dynamic Real includes gate collection; training update times include state audits and exclude recovery-checkpoint writing. No isolated benchmark was run.',
        '','One complete recovery checkpoint was written at continuation update 144 for each arm. Recovery saves preserve scientific RNG state. Final checkpoints passed strict reopen and independent local/persistent SHA verification. All historical checkpoints, results, manifests and sealed tags were retained.',
        '','| Arm | Source 100M SHA-256 | Final 250M SHA-256 |','|---|---|---|']
    for arm in ('S','D'):
        lines.append(f"| {arm} | `{s['source_checkpoint_sha256'][arm]}` | `{s['checkpoints'][arm]['sha256']}` |")
    lines.append('')
    for arm,m in s['checkpoints'].items():
        lines += [f"{arm} local final checkpoint: `{m['local_checkpoint']}`",f"{arm} retained persistent final checkpoint: `{m['persistent_checkpoint']}`",'']
    lines += [f"Terminal loader cursor SHA: `{s['terminal']['loader_cursor_sha256']}`",f"Next global batch SHA: `{s['terminal']['next_batch_sha256']}`",f"Next stream SHA: `{s['terminal']['next_stream_sha256']}`",
        f"Terminal equality across S, D and the frozen continuation: **{'PASS' if s['terminal']['equal'] else 'FAIL'}**.",'',
        f"Final audit: **{'PASS' if s['audit']['passed'] else 'FAIL'}, {s['audit']['passed_count']}/{s['audit']['total_checks']} checks**.",
        f"GPU pod `{s['stop']['pod']['name']}` (`{s['stop']['pod']['id']}`): desired status `{s['stop']['pod']['desiredStatus']}`, runtime status `{s['stop']['pod']['runtimeStatus']}`. Stop verification: **{'PASS' if s['stop']['passed'] else 'FAIL'}**. Persistent volume `yhzyb27fb5` retained.",
        '','## Decision and one next recommendation','']
    a,p=s['paired']['contrasts']['A'],s['paired']['contrasts']['P']
    lines += [f"Matched architecture benefit: {'statistically established' if a['ci95'][0]>0 else 'not statistically established'}; benefit beyond the practical margin: {'established' if a['ci95'][0]>DELTA else 'not established'}.",
        f"The learned w term’s inference utility: {'statistically established' if p['ci95'][0]>0 else 'not statistically established'}."]
    if not s['audit']['passed']:
        lines.append('Integrity is incomplete: no scientific winner or architecture adoption recommendation. Resolve the failed audit checks before interpreting the contrasts.')
    elif s['preferred_architecture']=='Dynamic':
        lines.append('Prefer Dynamic among the tested architectures: the existing adoption rule is met. Next recommendation: use the sealed 250M Dynamic checkpoint as the candidate baseline for the next separately scoped experiment.')
    else:
        lines.append('Retain static O1 provisionally because the existing Dynamic adoption rule is not met. Next recommendation: keep static O1 as the baseline for the next separately scoped architecture experiment.')
    lines += ['', 'No automatic 500M extension or further experiment was launched. '+('This continuation is complete and stopped.' if s['audit']['passed'] else 'This continuation is incomplete; consult the failed audit checks.'), '']
    return '\n'.join(lines)


if __name__=='__main__':main()
