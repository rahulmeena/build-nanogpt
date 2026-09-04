"""CPU-only sealed 2D9 analysis. Execute only after verified GPU shutdown."""
import argparse, hashlib, json, math, pathlib, statistics
import numpy as np
CONDITIONS=('STATIC_REAL','DYNAMIC_REAL','DYNAMIC_STATICIZED')
DELTA=.0001
PARENT='c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6'
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def flags(row):
 lo,hi=row['ci95']
 return {'positive_utility':lo>0,'benefit_beyond_margin':lo>DELTA,'negative_utility':hi<0,
         'harm_beyond_margin':hi < -DELTA,'practical_equivalence':lo > -DELTA and hi < DELTA,
         'noninferiority_of_second_named_condition':lo > -DELTA}
def classify(a,p,integrity=True,state_delta=0):
 if not integrity:return 'INVALID / INCOMPLETE — no scientific winner','No scientific winner'
 if a['ci95'][1] < -DELTA or p['ci95'][1] < -DELTA:
  where=[]
  if a['ci95'][1] < -DELTA:where.append('matched architecture')
  if p['ci95'][1] < -DELTA:where.append('active w term')
  label='MATERIAL HARM ESTABLISHED — '+', '.join(where)
 elif a['ci95'][0]>0 and p['ci95'][0]>0:label='TOKEN-CONDITIONED GATING ESTABLISHES UTILITY'
 elif a['ci95'][0]>0:label='MATCHED ARCHITECTURE BENEFIT; ACTIVE W-TERM UTILITY NOT ESTABLISHED'
 elif p['ci95'][0]>0:label='ACTIVE W-TERM UTILITY; MATCHED ARCHITECTURE BENEFIT NOT ESTABLISHED'
 elif a['mean']>0 and p['mean']>0:label='DIRECTIONALLY POSITIVE, NOT ESTABLISHED'
 elif flags(a)['practical_equivalence'] and p['ci95'][0]<=0:label='ARCHITECTURES PRACTICALLY EQUIVALENT; NO ESTABLISHED ADDED UTILITY'
 else:label='MIXED / UNRESOLVED — report both contrasts explicitly'
 preferred='Dynamic' if a['ci95'][0]>DELTA and p['ci95'][0]>0 and state_delta==0 else 'Retain static O1 provisionally'
 return label,preferred

def bootstrap(arrays):
 s,d,z=arrays
 differences=np.stack((s-d,z-d,s-z),axis=1)
 rng=np.random.default_rng(20260906)
 sampled=np.empty((50000,3),dtype=np.float64)
 for start in range(0,50000,250):
  stop=min(start+250,50000)
  indices=rng.integers(0,4096,size=(stop-start,4096))
  sampled[start:stop]=differences[indices].mean(axis=1)
 names=('A','P','R'); labels=('Static − Dynamic','Staticized − Dynamic','Static − Staticized')
 rows={}
 for i,(name,label) in enumerate(zip(names,labels)):
  values=differences[:,i];mean=float(values.mean());ci=np.percentile(sampled[:,i],[2.5,97.5],method='linear')
  row={'contrast':label,'mean':mean,'ci95':ci.tolist(),'exp_contrast':math.exp(mean),'exp_ci95':np.exp(ci).tolist(),
       'second_condition_sequence_wins':int((values>0).sum()),'first_condition_sequence_wins':int((values<0).sum()),'ties':int((values==0).sum())}
  row['flags']=flags(row);rows[name]=row
 return {'seed':20260906,'resamples':50000,'unit':'paired sequence (4096)','shared_indices_across_contrasts':True,
         'interval':'95% percentile; NumPy linear','numpy_version':np.__version__,'bit_generator':'PCG64','delta_ce':DELTA,'contrasts':rows,
         'decomposition_residual':float(differences[:,0].mean()-differences[:,1].mean()-differences[:,2].mean()),
         'uncertainty_scope':'Evaluation-sequence uncertainty for two sealed training trajectories; no training-seed replication.',
         'shared_dynamic_condition':'A and P share Dynamic and are not independent replications.'}

def basic_stats(a):
 x=np.asarray(a,dtype=np.float64)
 return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max())}
def gate_stats(path,parameters,parent):
 a=np.load(path,mmap_mode='r');assert a.shape==(3,3,4096,1024) and a.dtype==np.float32 and np.isfinite(a).all()
 result={'array_sha256':sha(path),'array_shape':list(a.shape),'array_layout':['destination B1/B3/B5','preactivation / FP32 g / BF16 coefficient represented as FP32','sequence','token position'],
         'quantiles':'Exact full-panel linear percentiles; no sampling','token_count_per_destination':4194304,
         'standard_deviation_definitions':{'across_tokens':'Population std over all 4,194,304 token coefficients (ddof=0)',
          'sequence_means':'Population std of 4,096 per-sequence means (ddof=0)','within_sequence':'Mean of 4,096 per-sequence population std values'},
         'device_buffer_bound':'One B64 × T1024 batch; one batched device-to-CPU transfer; no RNG consumed','destinations':{}}
 edges=[0,1,32,64,128,256,512,768,1024]
 for i,(b,minimum) in enumerate(zip(('B1','B3','B5'),(1,31,63))):
  row={'parent_raw_g0':parent[b],**parameters[b],'preactivation_fp32':basic_stats(a[i,0]),
       'gate_input':'token embedding + position embedding' if b=='B1' else 'current pre-attention residual with preceding contextual processing',
       'first_position_with_recurrent_memory':minimum}
  for j,name in ((1,'mathematical_fp32_gate'),(2,'effective_bf16_coefficient')):
   v=np.asarray(a[i,j],dtype=np.float64);stats=basic_stats(v)
   stats.update(percentiles=dict(zip(['p1','p5','p25','median','p75','p95','p99'],np.percentile(v,[1,5,25,50,75,95,99],method='linear').tolist())),
                negative_fraction=float((v<0).mean()),mean_absolute_deviation_from_final_tanh_g0=float(np.abs(v-parameters[b]['tanh_g0']).mean()),
                std_of_per_sequence_means=float(v.mean(axis=1).std()),mean_within_sequence_std=float(v.std(axis=1).mean()))
   stats['position_bins']={}
   for start,end in zip(edges,edges[1:]):
    key=str(start) if end-start==1 else f'{start}–{end-1}'
    stats['position_bins'][key]={**basic_stats(v[:,start:end]),'positions':[start,end-1],
      'fraction_of_positions_with_eligible_recurrent_memory':max(0,end-max(start,minimum))/(end-start)}
   row[name]=stats
  result['destinations'][b]=row
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--gate-scalars',required=True);args=p.parse_args()
 out=pathlib.Path(args.results)
 stop=read(out/'STOP_VERIFICATION.json')
 assert stop['passed'] and stop['pod']['runtimeStatus']=='stopped' and stop['pod']['desiredStatus']=='EXITED'
 source=read(out/'SOURCE_AND_PARAMETER_AUDIT.json');preflight=read(out/'PREFLIGHT_AUDIT.json')
 zero=read(out/'FULL_SOURCE_ZERO_EQUIVALENCE.json');panel=read(out/'EVALUATION_PANEL_MANIFEST.json');disjoint=read(out/'DISJOINTNESS_AUDIT.json')
 continuation=read(out/'CONTINUATION_MANIFEST.json');ledger=[json.loads(x) for x in (out/'MATCHED_BATCH_LEDGER.jsonl').read_text().splitlines()]
 checkpoints=read(out/'CHECKPOINT_MANIFESTS.json');operations=read(out/'OPERATIONS_STATUS.json')
 training={a:read(out/f'TRAINING_COMPLETE_{a}.json') for a in ('S','D')}
 logs={a:[json.loads(x) for x in (out/f'TRAINING_{a}.jsonl').read_text().splitlines()] for a in ('S','D')}
 evaluations={n:read(out/(n+'.json')) for n in CONDITIONS}
 arrays=np.asarray([evaluations[n]['per_sequence_ce'] for n in CONDITIONS],dtype=np.float64)
 assert arrays.shape==(3,4096) and np.isfinite(arrays).all()
 paired=bootstrap(arrays);write(out/'PAIRED_BOOTSTRAP.json',paired)
 gates=gate_stats(args.gate_scalars,evaluations['DYNAMIC_REAL']['model_gate_parameters'],zero['parent_gate_values']);write(out/'GATE_STATISTICS.json',gates)
 memory={'persistent_state_by_condition':{n:evaluations[n]['persistent_state'] for n in CONDITIONS},'persistent_state_delta_bytes':evaluations['DYNAMIC_REAL']['persistent_state']['logical_bytes_per_sequence']-evaluations['STATIC_REAL']['persistent_state']['logical_bytes_per_sequence'],
         'parameter_counts':{'S':124475908,'D':124478212},'parameter_increase':2304,'additional_fp32_parameter_bytes':9216,'additional_bf16_parameter_bytes':4608,
         'training_wall_seconds':{a:training[a]['training_wall_seconds'] for a in ('S','D')},
         'evaluation_wall_seconds':{n:evaluations[n]['wall_seconds'] for n in CONDITIONS},
         'timing_scope':'Observed workload wall times; no isolated timing benchmark. Dynamic Real includes gate diagnostics. Training includes gradient, optimizer, and finite-state audits.',
         'training_only_state':'Adam moments and steps excluded from persistent inference state; diagnostic scalar arrays are analysis artifacts.'}
 write(out/'MEMORY_AND_RUNTIME.json',memory)
 names={a:{r['name'] for r in source[a]['parameter_inventory']} for a in ('S','D')}
 checks={
  'sealed_O1_source':all(all(source[a]['checks'].values()) for a in ('S','D')) and zero['source_sha256']==PARENT,
  'exact_old_tensor_optimizer_restore':all(source[a]['checks']['all_old_tensors_exact'] and source[a]['checks']['old_optimizer_states_exact'] for a in ('S','D')),
  'only_three_new_vectors':names['D']-names['S']=={'w_B1','w_B3','w_B5'} and names['S']<=names['D'],
  'parameter_counts':all(sum(math.prod(r['shape']) for r in source[a]['parameter_inventory'])==expected for a,expected in [('S',124475908),('D',124478212)]),
  'zero_vectors_fresh_moments':source['D']['checks']['new_zero_vectors_fresh_state'],
  'tied_weights':all(source[a]['checks']['tied_weights'] for a in ('S','D')),
  'zero_effect_FP32_BF16_parallel_incremental':zero['passed'] and len(zero['checks'])==4 and all(r['ce_abs_difference']<=1e-6 and r['max_logits_abs_difference']<=2e-6 for r in zero['checks']),
  'causality_row_isolation_and_roundtrip_tests':preflight['targeted_tests_passed'] and '10 passed' in (out/'TARGETED_TESTS.txt').read_text(),
  'first_backward_all_new_vectors_finite_nonzero':all(preflight['disposable_smoke']['D']['gradients'][n]['finite'] and preflight['disposable_smoke']['D']['gradients'][n]['nonzero'] for n in ('w_B1','w_B3','w_B5')),
  'disposable_state_reloads':all(preflight['disposable_smoke'][a]['state_reload_exact'] for a in ('S','D')),
  'implementation_bound_before_training':all(training[a]['checks']['code_git_commit'] and training[a]['checks']['code_implementation'] for a in ('S','D')),
  'new_continuation_from_2290':continuation['starting_global_update']==2290 and continuation['first_global_update']==2291 and continuation['passed'],
  'ledger_hash':sha(out/'MATCHED_BATCH_LEDGER.jsonl')==continuation['ledger_sha256'],
  'exact_191_updates_each':all(len(logs[a])==191 and [r['global_update'] for r in logs[a]]==list(range(2291,2482)) for a in ('S','D')),
  'exact_target_budget_each':all(sum(r['target_count'] for r in logs[a])==100139008 and logs[a][-1]['cumulative_targets']==1300758528 for a in ('S','D')),
  'same_logical_batches':all([r['batch_sha256'] for r in logs[a]]==[r['logical_global_batch_sha256'] for r in ledger] for a in ('S','D')),
  'same_logical_streams':all([r['stream_sha256'] for r in logs[a]]==[r['logical_global_stream_sha256'] for r in ledger] for a in ('S','D')),
  'exact_pass_schedule':all([r['global_update'] for r in logs[a] if r['pass_count']==3]==[2304,2336,2368,2400,2432,2464] and sum(r['pass_count']==2 for r in logs[a])==185 for a in ('S','D')),
  'all_gradients_finite':all(all(g['finite'] for g in r['active_gradient_groups'].values()) and math.isfinite(r['gradient_norm_before_clip']) for rows in logs.values() for r in rows),
  'all_update_optimizer_checks':all(all(r['optimizer_checks'].values()) for rows in logs.values() for r in rows),
  'dormant_B6_unchanged':all(r['optimizer_checks']['dormant_optimizer_unchanged'] and r['optimizer_checks']['dormant_parameter_unchanged'] for rows in logs.values() for r in rows),
  'strict_final_reopen':all(checkpoints[a]['verification']['strict_reopen']['passed'] for a in ('S','D')),
  'old_and_new_optimizer_step_progression':all(checkpoints[a]['verification']['strict_reopen']['checks']['optimizer_progression'] for a in ('S','D')),
  'terminal_cursor_equality':training['S']['final_loader_cursor']==training['D']['final_loader_cursor']==continuation['final_loader_cursor'],
  'terminal_batch_equality':training['S']['next_global_batch_sha256']==training['D']['next_global_batch_sha256']==continuation['next_global_batch_sha256'],
  'terminal_stream_equality':training['S']['next_stream_sha256']==training['D']['next_stream_sha256']==continuation['next_stream_sha256'],
  'one_frozen_fresh_panel':panel['candidate_panels_constructed']==1 and panel['selection_seed']==20260905 and panel['sealed_before_training_and_scoring'] and disjoint['passed'],
  'historical_exclusion_evidence':all(disjoint['checks'][k] for k in ('no_historical_overlap','recent_panels_recovered')),
  'exactly_three_final_conditions':set(n for n in CONDITIONS if (out/(n+'.json')).exists())==set(CONDITIONS) and all(evaluations[n]['condition']==n for n in CONDITIONS),
  'three_4096_loss_arrays':all(len(evaluations[n]['per_sequence_ce'])==4096 and evaluations[n]['targets']==4194304 and evaluations[n]['passed'] for n in CONDITIONS),
  'ordered_sequence_identity_equality':all(evaluations[n]['sequence_identities']==panel['sequence_identities'] and evaluations[n]['panel_sha256']==panel['panel_sha256'] and evaluations[n]['panel_manifest_sha256']==sha(out/'EVALUATION_PANEL_MANIFEST.json') for n in CONDITIONS),
  'staticized_uses_identical_D_checkpoint':evaluations['DYNAMIC_REAL']['checkpoint_sha256']==evaluations['DYNAMIC_STATICIZED']['checkpoint_sha256']==checkpoints['D']['sha256'],
  'evaluation_did_not_mutate_checkpoints_or_weights':all(evaluations[n]['checkpoint_file_unchanged'] and evaluations[n]['model_tensors_unchanged'] for n in CONDITIONS),
  'paired_50k_sequence_bootstrap':paired['resamples']==50000 and paired['shared_indices_across_contrasts'] and abs(paired['decomposition_residual'])<1e-14,
  'zero_persistent_state_growth':all(all(v==33289728 for v in evaluations[n]['persistent_state'].values()) for n in CONDITIONS),
  'full_panel_gate_statistics':gates['token_count_per_destination']==4194304,
  'scalar_export_hash_matches_persistent':gates['array_sha256']==read(out/'GATE_SCALAR_EXPORT.json')['sha256'],
  'exact_evaluation_file_inventory':{p.name for p in pathlib.Path(args.gate_scalars).parent.glob('*.json')}=={n+'.json' for n in CONDITIONS},
  'constant_learning_rates':all(all(value==(3e-5 if name.startswith('base_') or name=='dynamic_nodecay' else 3e-4) for name,value in r['lr'].items()) for rows in logs.values() for r in rows),
  'microbatch_and_accumulation':continuation['microbatch']==32 and continuation['gradient_accumulation']==16,
  'independently_verified_local_persistent_backups':all(checkpoints[a]['verified_independently'] and sha(checkpoints[a]['local_checkpoint'])==checkpoints[a]['persistent_sha256']==checkpoints[a]['sha256'] for a in ('S','D')),
  'verified_exact_GPU_stop':stop['passed'] and stop['pod']['id']=='rx11t3e4lvfuhf' and stop['pod']['runtimeStatus']=='stopped' and stop['pod']['desiredStatus']=='EXITED',
  'persistent_volume_retained':stop['persistent_volume_retained'] and stop['pod']['networkVolumeId']=='yhzyb27fb5',
  'workflow_completed':operations['outcome']=='gpu_work_complete' and operations['workflow_exit_code']==0 and not operations['error'] and not operations['stop_error'],
 }
 final={'passed':all(checks.values()),'checks':checks,'passed_count':sum(checks.values()),'total_checks':len(checks),'implementation_commit':preflight['git_commit']}
 write(out/'FINAL_AUDIT.json',final)
 label,preferred=classify(paired['contrasts']['A'],paired['contrasts']['P'],final['passed'])
 summary={'classification':label,'preferred_architecture':preferred,'aggregate_ce':{n:evaluations[n]['aggregate_ce'] for n in CONDITIONS},
          'perplexity':{n:evaluations[n]['perplexity'] for n in CONDITIONS},'paired':paired,'gates':gates,'memory_and_runtime':memory,
          'checkpoints':checkpoints,'terminal':{'loader_cursor_sha256':canonical(continuation['final_loader_cursor']),
             'next_batch_sha256':continuation['next_global_batch_sha256'],'next_stream_sha256':continuation['next_stream_sha256'],'equal':True},
          'audit':final,'stop':stop}
 write(out/'SCIENTIFIC_RESULT_SUMMARY.json',summary)
 text=report(summary,preflight,panel)
 (out/'EXPERIMENT_2D9_FINAL_REPORT.md').write_text(text)
 print(json.dumps({'classification':label,'preferred':preferred,'ce':summary['aggregate_ce'],'contrasts':paired['contrasts'],'audit':final},indent=2))
 if not final['passed']:raise SystemExit('final integrity audit failed')

def report(s,preflight,panel):
 status='COMPLETE' if s['audit']['passed'] else 'INCOMPLETE'
 lines=[f'# EXPERIMENT 2D9 — TOKEN-CONDITIONED DYNAMIC RECURRENT GATING {status}',
 '',f"Primary classification: **{s['classification']}**",f"Preferred architecture recommendation: **{s['preferred_architecture']}**",'',
 '| Condition | CE | Perplexity |','|---|---:|---:|']
 for n in CONDITIONS:lines.append(f"| {n} | {s['aggregate_ce'][n]:.12f} | {s['perplexity'][n]:.12f} |")
 lines+=['','| Contrast | Mean CE | 95% paired CI | exp(contrast) | Second / first wins / ties |','|---|---:|---|---:|---|']
 for name,r in s['paired']['contrasts'].items():
  lines.append(f"| {name}: {r['contrast']} | {r['mean']:+.12f} | [{r['ci95'][0]:+.12f}, {r['ci95'][1]:+.12f}] | {r['exp_contrast']:.12f} | {r['second_condition_sequence_wins']} / {r['first_condition_sequence_wins']} / {r['ties']} |")
 lines+=['','delta_CE = 0.0001. Positive A favors Dynamic over matched Static; positive P favors the learned w term on D weights. R is descriptive; A = R + P.','',
 '| Primary contrast | Positive utility | Beyond margin | Negative utility | Material harm | Equivalent | Second condition noninferior |','|---|---|---|---|---|---|---|']
 for name in ('A','P'):
  f=s['paired']['contrasts'][name]['flags'];lines.append('| '+name+' | '+' | '.join(str(f[k]) for k in ('positive_utility','benefit_beyond_margin','negative_utility','harm_beyond_margin','practical_equivalence','noninferiority_of_second_named_condition'))+' |')
 lines+=['','All flags use strict inequalities. Statistical utility and practical equivalence can coexist. Unresolved significance is not equivalence.',
 '','## Learned gates','', '| Destination | Parent raw g0 | Final raw g0 | tanh(g0) | ‖w‖₂ | FP32 gate mean / std / range | BF16 coefficient mean / std / range |', '|---|---:|---:|---:|---:|---|---|']
 for name,r in s['gates']['destinations'].items():
  a=r['mathematical_fp32_gate'];b=r['effective_bf16_coefficient']
  lines.append(f"| {name} | {r['parent_raw_g0']:.9f} | {r['g0']:.9f} | {r['tanh_g0']:.9f} | {r['w_norm']:.9f} | {a['mean']:.9f} / {a['std']:.9f} / [{a['min']:.9f}, {a['max']:.9f}] | {b['mean']:.9f} / {b['std']:.9f} / [{b['min']:.9f}, {b['max']:.9f}] |")
 lines+=['','B1 gates see token embedding plus position embedding. B3 and B5 gates also see preceding contextual processing. The FP32 RMS epsilon is 1e-5, with no sqrt(768) scaling. The coefficient is cast to the attention dtype immediately before multiplying the recurrent output. Local and recurrent softmaxes remain separate and the destination projection, including its bias, is applied once.',
 '','Gate summaries use all 4,194,304 positions per destination, including positions without eligible memory. An intrinsic gate at those positions has no recurrent output to scale. Percentiles are exact full-panel linear percentiles. Global token std, std of sequence means, and mean within-sequence std are population measures (ddof=0). Full moments, percentiles, negative fractions, deviation from tanh(g0), and position bins for both precisions are in GATE_STATISTICS.json. Nonzero vectors or gate variance alone are not evidence of utility.',
 *gate_detail_tables(s['gates']), '', '## Training, evaluation and provenance','',
 'Exactly **191 updates / 100,139,008 new targets per arm**, independently from the accepted O1 parent. Updates 2291–2481; final cumulative targets 1,300,758,528. B32×T1024, 16 microbatches/update; 185 two-pass and 6 three-pass updates. Three-pass updates: 2304, 2336, 2368, 2400, 2432, 2464. Old parameter values, moments, individual Adam counters, constant LR metadata, all RNG states and the loader were restored. The new vectors use a separate group with the inherited base no-decay settings and fresh moments. Dormant g_rec_b6 and its old optimizer state remained unchanged.',
 '',f"Source SHA: `{PARENT}`",f"Committed scientific implementation: `{preflight['git_commit']}`",f"Fresh panel SHA: `{panel['panel_sha256']}`",f"Panel seed: {panel['selection_seed']}; 64 canonical B64 batches, 4096 sequences, 4,194,304 targets per condition. Recovered historical target spans and the reserved prefix were excluded before training/scoring. Exactly three final evaluations used BF16 execution, FP32 token CE, and FP64 accumulation.",
 '','Staticized resets all caches and generates its entire trajectory with the w term omitted. D-trained g0 and base weights are retained. The control does not mutate D tensors or its checkpoint.',
 '','50,000 paired sequence bootstrap resamples use NumPy RNG seed 20260906, shared indices for A/P/R, and 95% linear percentile intervals. These intervals quantify evaluation-sequence uncertainty for two sealed training trajectories. A and P share Dynamic and are not independent replications; this is not a training-seed study.',
 '','Staticization removes the complete learned w term, including an average shift as well as token variation. A positive P establishes the term’s inference utility; it does not show token variation beats every optimally refitted constant gate.',
 '','## State, runtime and artifacts','',
 'Persistent inference state is **33,289,728 BF16 bytes per B=1 sequence in all conditions; delta 0**. Dynamic adds 2,304 parameters (9,216 FP32 bytes; 4,608 bytes if stored in BF16). Adam state is training-only. Gate diagnostic arrays are analysis artifacts, not persistent model state.',
 '','| Workload | Wall time (minutes) |','|---|---:|']
 for name,seconds in {**s['memory_and_runtime']['training_wall_seconds'],**s['memory_and_runtime']['evaluation_wall_seconds']}.items():lines.append(f'| {name} | {seconds/60:.2f} |')
 lines+=['','These are descriptive workload timings, not an isolated gate timing benchmark. Dynamic Real includes gate collection; training includes state audits. The optional timing campaign was omitted.', '']
 for a,m in s['checkpoints'].items():lines += [f"{a} checkpoint SHA: `{m['sha256']}`",f"Local checkpoint: `{m['local_checkpoint']}`",f"Retained persistent checkpoint: `{m['persistent_checkpoint']}`",'']
 lines += [f"Terminal loader cursor SHA: `{s['terminal']['loader_cursor_sha256']}`",f"Next global batch SHA: `{s['terminal']['next_batch_sha256']}`",f"Next stream SHA: `{s['terminal']['next_stream_sha256']}`",'Terminal identities match S versus D and the frozen continuation.', '',
 f"Final audit: **{'PASS' if s['audit']['passed'] else 'FAIL'}, {s['audit']['passed_count']}/{s['audit']['total_checks']} checks**. Final checkpoints passed strict reopen and independent local/persistent SHA verification before GPU shutdown.",
 'GPU pod `lazy_tan_louse` (`rx11t3e4lvfuhf`) is verified **stopped**, with desired status EXITED and runtime status stopped. Persistent volume `yhzyb27fb5` is retained.', '',
 '## Decision and next recommendation','']
 a,p=s['paired']['contrasts']['A'],s['paired']['contrasts']['P']
 lines += [f"Matched architecture benefit: {'established statistically' if a['ci95'][0]>0 else 'not established statistically'}; beyond-margin benefit: {'established' if a['ci95'][0]>.0001 else 'not established'}.",
 f"Learned w-term inference benefit: {'established statistically' if p['ci95'][0]>0 else 'not established statistically'}."]
 if s['preferred_architecture']=='Dynamic':lines.append('Prefer Dynamic among the tested architectures. Next recommendation: use Dynamic as the candidate for a separately authorized longer matched continuation to assess persistence of the gain.')
 elif a['ci95'][0]>0 and p['ci95'][0]>0:lines.append('Retain static O1 provisionally because the matched architecture benefit does not clear the practical margin. Next recommendation: consider a separately authorized matched 250M continuation.')
 else:lines.append('Retain static O1. Next recommendation: keep the simpler static architecture as the continuation baseline; additional Dynamic training would require a new scientific decision.')
 lines += ['','No follow-up launched. No extra seeds, panels, conditions, overlap-width sweeps or diagnostic campaigns were run. Checkpoints, results, panel manifests and the retained volume must not be deleted without explicit authorization.','']
 return '\n'.join(lines)

def gate_detail_tables(gates):
 lines=['', 'FP32 preactivation summaries (all panel positions):', '', '| Destination | Mean | Std | Minimum | Maximum |', '|---|---:|---:|---:|---:|']
 for name,row in gates['destinations'].items():
  r=row['preactivation_fp32'];lines.append('| '+name+' | '+' | '.join(f"{r[k]:.9f}" for k in ('mean','std','min','max'))+' |')
 lines+=['', 'Exact gate percentiles:', '', '| Destination / precision | p1 | p5 | p25 | Median | p75 | p95 | p99 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,row in gates['destinations'].items():
  for key,label in [('mathematical_fp32_gate','FP32'),('effective_bf16_coefficient','BF16')]:
   r=row[key]['percentiles'];lines.append('| '+name+' / '+label+' | '+' | '.join(f"{r[k]:.9f}" for k in ('p1','p5','p25','median','p75','p95','p99'))+' |')
 lines+=['', 'Gate variation and sign summaries:', '', '| Destination / precision | Negative fraction | Mean abs. deviation from tanh(g0) | Std of sequence means | Mean within-sequence std |', '|---|---:|---:|---:|---:|']
 for name,row in gates['destinations'].items():
  for key,label in [('mathematical_fp32_gate','FP32'),('effective_bf16_coefficient','BF16')]:
   r=row[key];lines.append('| '+name+' / '+label+' | '+' | '.join(f"{r[k]:.9f}" for k in ('negative_fraction','mean_absolute_deviation_from_final_tanh_g0','std_of_per_sequence_means','mean_within_sequence_std'))+' |')
 lines+=['', 'Position-bin summaries (0, 1–31, 32–63, 64–127, 128–255, 256–511, 512–767, 768–1023) and eligible-memory fractions are preserved in GATE_STATISTICS.json. First eligible positions are B1: 1, B3: 31, B5: 63.']
 return lines

if __name__=='__main__':main()
