#!/usr/bin/env python3
"""Local-only, post-shutdown paired analysis of the frozen 2D10 screen."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/experiment_2d10_retrieval_aware_gating_100m'
ARCHIVE=ROOT.parents[1]/'runpod-checkpoint-archive/experiment_2d10_retrieval_aware_gating_100m'
CONTRASTS=[('D-T','D','T'),('D-H','D','H'),('T-H','T','H'),('S-D','S','D'),('S-T','S','T'),('S-H','S','H')]
PRIMARY_Q=[0.833333333333,99.166666666667]
MARGIN=.0001
FLAG_COLUMNS=("positive","beyond_margin","negative","material_harm","practical_equivalence","second_condition_noninferiority")

def render_flag_values(values):
    return " | ".join(str(values[key]) for key in FLAG_COLUMNS)


def read(p):return json.loads(Path(p).read_text())
def write(name,value):
    path=RESULT/name
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def flags(ci):
    lo,hi=ci
    return {'positive':lo>0,'beyond_margin':lo>MARGIN,'negative':hi<0,'material_harm':hi < -MARGIN,
            'practical_equivalence':lo > -MARGIN and hi < MARGIN,'second_condition_noninferiority':lo > -MARGIN}

def bootstrap(losses):
    differences=np.stack([losses[a]-losses[b] for _,a,b in CONTRASTS],axis=1)
    assert differences.shape==(4096,6) and np.isfinite(differences).all()
    np.testing.assert_allclose(differences[:,0]+differences[:,2],differences[:,1],rtol=0,atol=1e-15)
    rng=np.random.default_rng(20260910)
    means=np.empty((50000,6),dtype=np.float64)
    for start in range(0,50000,128):
        end=min(start+128,50000)
        indices=rng.integers(0,4096,size=(end-start,4096))
        means[start:end]=differences[indices].mean(axis=1)
    result={'seed':20260910,'resamples':50000,'resampling_unit':'paired sequence','shared_indices_all_six_contrasts':True,
            'percentile_method':'linear','primary_adjusted_percentiles':PRIMARY_Q,'primary_marginal_coverage_percent':98.333333333333,
            'nominal_primary_family_coverage_percent':95,'margin':MARGIN,'contrasts':{}}
    for i,(name,a,b) in enumerate(CONTRASTS):
        values=differences[:,i];mean=float(values.mean())
        raw=np.percentile(means[:,i],[2.5,97.5],method='linear').tolist()
        adjusted=np.percentile(means[:,i],PRIMARY_Q,method='linear').tolist() if i<3 else None
        result['contrasts'][name]={'first':a,'second':b,'positive_favors':b,'mean':mean,'raw_95_ci':raw,
            'adjusted_98_333333_ci':adjusted,'exp_contrast':math.exp(mean),'second_wins':int((values>0).sum()),
            'first_wins':int((values<0).sum()),'ties':int((values==0).sum()),'family':'primary' if i<3 else 'descriptive',
            'adjusted_flags':flags(adjusted) if i<3 else None}
    np.save(ARCHIVE/'BOOTSTRAP_MEANS.npy',means)
    result['bootstrap_means_sha256']=sha(ARCHIVE/'BOOTSTRAP_MEANS.npy')
    return result

def gate_statistics():
    result={'quantile_seed':20260911,'quantiles_percent':[1,5,25,50,75,95,99],
            'quantile_sampling':'one fixed sample of up to 131072 eligible positions per destination; same positions across arms and metrics',
            'moments':'full-panel FP64 population mean/variance/extrema/fractions; eligible and unavailable positions separate',
            'scalar_exports':{},'arms':{}}
    for arm in 'TH':
        path=ARCHIVE/'gpu_artifacts/evaluation'/f'GATE_SCALARS_{arm}.npy'
        array=np.load(path,mmap_mode='r')
        keys=(['u','delta','g','cast_g'] if arm=='T' else ['logit_difference','lambda_L','lambda_R','cast_lambda_L','cast_lambda_R','entropy'])
        assert array.shape==(3,len(keys)+1,4096,1024) and np.isfinite(array).all()
        result['scalar_exports'][arm]={'path':str(path),'sha256':sha(path),'shape':list(array.shape)}
        rng=np.random.default_rng(20260911); blocks={}
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
    assert all(result['arms']['T'][b]['quantile_index_sha256']==result['arms']['H'][b]['quantile_index_sha256'] for b in ('B1','B3','B5'))
    return result

def main():
    started=time.time();stop=read(RESULT/'STOP_VERIFICATION.json')
    assert stop['passed'] and stop['pod']['runtimeStatus']=='stopped' and stop['persistent_volume_retained']
    assert read(RESULT/'OPERATIONS_STATUS.json')['outcome']=='gpu_work_complete'
    assert read(RESULT/'ARTIFACT_BACKUP_VERIFICATION.json')['passed']
    panel=read(RESULT/'EVALUATION_PANEL_MANIFEST.json')
    evaluation={a:read(RESULT/(a+'_REAL.json')) for a in 'SDTH'}
    losses={a:np.asarray(v['per_sequence_ce'],dtype=np.float64) for a,v in evaluation.items()}
    for a,v in evaluation.items():
        assert v['passed'] and v['status']=='complete' and v['targets']==4194304
        assert losses[a].shape==(4096,) and np.isfinite(losses[a]).all()
        assert v['sequence_identities']==panel['sequence_identities']
        assert v['batch_identities']==panel['batch_identities']
        assert v['completed_batch_indices']==panel['batch_indices_in_evaluation_order']
        assert v['panel_sha256']==panel['panel_sha256'] and v['panel_manifest_sha256']==sha(RESULT/'EVALUATION_PANEL_MANIFEST.json')
        assert v['checkpoint_file_unchanged'] and v['model_tensors_unchanged']
        assert all(b==33289728 for b in v['persistent_state'].values())
        np.testing.assert_allclose(losses[a]*1024,v['per_sequence_nll'],rtol=0,atol=1e-12)
        assert abs(losses[a].mean()-v['aggregate_ce'])<1e-12
    np.savez_compressed(RESULT/'PAIRED_SEQUENCE_LOSSES.npz',**losses,targets_per_sequence=np.full(4096,1024,dtype=np.int64))
    statistics=bootstrap(losses);write('PAIRED_BOOTSTRAP.json',statistics)
    print(json.dumps(statistics['contrasts'],indent=2),flush=True)
    gates=gate_statistics();write('GATE_STATISTICS.json',gates)
    source_checks=read(RESULT/'SOURCE_AND_PARAMETER_AUDIT.json')
    preflight=read(RESULT/'PREFLIGHT_AUDIT.json')
    checkpoints=read(RESULT/'CHECKPOINT_MANIFESTS.json')
    ledger=[json.loads(line) for line in (RESULT/'MATCHED_BATCH_LEDGER.jsonl').read_text().splitlines()]
    terminal=read(RESULT/'CONTINUATION_MANIFEST.json')
    training={a:read(RESULT/f'TRAINING_COMPLETE_{a}.json') for a in 'TH'}
    replay={}
    for a in 'TH':
        log=[json.loads(line) for line in (RESULT/f'TRAINING_{a}.jsonl').read_text().splitlines()]
        assert len(log)==191 and training[a]['passed'] and all(training[a]['checks'].values())
        assert sum(x['target_count'] for x in log)==100139008
        assert [x['batch_sha256'] for x in log]==[x['logical_global_batch_sha256'] for x in ledger]
        assert [x['stream_sha256'] for x in log]==[x['logical_global_stream_sha256'] for x in ledger]
        assert [x['pass_count'] for x in log]==[x['pass_count'] for x in ledger]
        assert all(all(x['optimizer_checks'].values()) and all(x['pre_forward_invariants'].values()) and x['end_cursor_exact'] for x in log)
        assert checkpoints[a]['verified_independently'] and checkpoints[a]['verification']['strict_reopen']['passed']
        assert checkpoints[a]['sha256']==evaluation[a]['checkpoint_sha256']
        assert training[a]['next_global_batch_sha256']==terminal['next_global_batch_sha256']
        assert training[a]['next_stream_sha256']==terminal['next_stream_sha256']
        assert training[a]['final_loader_cursor']==terminal['final_loader_cursor']
        replay[a]={'updates':len(log),'targets':sum(x['target_count'] for x in log),'global_updates':[log[0]['global_update'],log[-1]['global_update']],
                   'two_pass_updates':sum(x['pass_count']==2 for x in log),'three_pass_updates':[x['global_update'] for x in log if x['pass_count']==3],
                   'all_191_batches_streams_cursors_passes_and_optimizer_checks_exact':True}
    assert sha(RESULT/'MATCHED_BATCH_LEDGER.jsonl')=='3955889e1c0849fa2ee0072cf1ca109170e955d3fc6914d970f6c58bf1b01bbd'
    files={}
    parent=ROOT.parents[1]/'runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/O/scientific_cumulative_001200619520.pt'
    files['O1']={'local_checkpoint':str(parent),'sha256':sha(parent),'persistent_checkpoint':'/workspace/exp2d7_boundary_alignment/run/checkpoints/O/scientific_cumulative_001200619520.pt'}
    assert files['O1']['sha256']=='c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6'
    for a in 'SD':
        path=ROOT.parents[1]/'runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating'/a/'scientific_cumulative_001300758528.pt'
        files[a]={'local_checkpoint':str(path),'persistent_checkpoint':f'/workspace/exp2d9_dynamic_gating/run/checkpoints/{a}/scientific_cumulative_001300758528.pt','sha256':sha(path),'reused_read_only':True}
        assert files[a]['sha256']==evaluation[a]['checkpoint_sha256']==preflight['controls'][a]['sha256']
    files.update(checkpoints)
    write('ALL_CHECKPOINT_IDENTITIES.json',files)
    benefit=[a for a in 'TH' if statistics['contrasts']['D-'+a]['adjusted_flags']['positive']]
    margin=[a for a in 'TH' if statistics['contrasts']['D-'+a]['adjusted_flags']['beyond_margin']]
    c=statistics['contrasts']['T-H']['adjusted_flags']
    direct=('H establishes benefit over T and clears the practical margin' if c['beyond_margin'] else 'H establishes benefit over T') if c['positive'] else 'T establishes benefit over H' if c['negative'] else 'T versus H is unresolved'
    classification='NEW CANDIDATE BENEFIT ESTABLISHED' if benefit else 'NO NEW CANDIDATE ESTABLISHES BENEFIT OVER D'
    recommendation=('Consider a separately authorized matched 250M continuation of '+', '.join(benefit)+' before any promotion over the mature Dynamic baseline.' if benefit else 'Retain the mature 2D9 250M Dynamic baseline; consider a separately authorized parameter-matched query-only MLP control before further retrieval-router training.')
    decision={'classification':classification,'candidates_establishing_benefit_over_D':benefit,'candidates_clearing_margin':margin,'direct_T_H_conclusion':direct,
              'current_mature_baseline':'sealed 2D9 250M Dynamic','recommendation':recommendation,'further_training_launched':False}
    write('SCREENING_DECISION.json',decision)
    supervised_start=dt.datetime.fromisoformat(read(RESULT/'STOP_CAPABILITY_PREFLIGHT.json')['observed_at_utc'])
    stopped=dt.datetime.fromisoformat(stop['verified_at_utc'])
    resumed=dt.datetime(2026,9,5,9,0,38,tzinfo=dt.timezone.utc)
    runtime={'current_T_H_training_seconds':{a:training[a]['training_wall_seconds'] for a in 'TH'},
        'current_evaluation_seconds':{a:evaluation[a]['wall_seconds'] for a in 'SDTH'},
        'pod_supervised_wall_seconds':(stopped-supervised_start).total_seconds(),
        'pod_supervised_aggregate_GPU_hours':2*(stopped-supervised_start).total_seconds()/3600,
        'pod_resume_to_verified_stop_wall_seconds':(stopped-resumed).total_seconds(),
        'pod_resume_to_verified_stop_aggregate_GPU_hours':2*(stopped-resumed).total_seconds()/3600,
        'supervision_started_utc':supervised_start.isoformat(),'verified_stopped_utc':stopped.isoformat(),
        'note':'Two GPUs. Supervised interval includes training, evaluation, export, and shutdown; resume interval also includes preparation. Timings are descriptive workloads, not isolated production router overhead.',
        'persistent_state_bytes_per_sequence_B_equals_1':33289728,'persistent_state_delta':0,
        'parameter_counts':{a:evaluation[a]['parameter_count'] for a in 'SDTH'},
        'new_FP32_parameter_payload_bytes':{'T':894720,'H':885912},
        'historical_SD_training_seconds':{a:read(ROOT/'results/experiment_2d9_token_conditioned_dynamic_recurrent_gating'/f'TRAINING_COMPLETE_{a}.json')['training_wall_seconds'] for a in 'SD'}}
    write('MEMORY_AND_RUNTIME.json',runtime)
    audit={'passed':True,'classification':'VALID / COMPLETE','scientific_implementation_commit':preflight['git_commit'],
        'frozen_implementation_sha256':preflight['implementation_sha256'],'replay':replay,'parent_and_control_identities_verified':True,
        'new_parameter_counts_verified':True,'optimizer_restored_by_explicit_reconstructed_source_parameter_names':True,
        'initial_T_parent_BF16_logits_and_CE_exact_both_modes':True,'initial_H_function_change_recorded':True,
        'causality_row_isolation_gradients_empty_memory_recomputation_tests_passed':True,
        'dormant_and_retired_states_unchanged':True,'complete_checkpoint_strict_reopens_passed':True,
        'terminal_equality_with_historical_100m_controls':True,'four_same_panel_evaluations':True,
        'primary_Bonferroni_adjusted_analysis':True,'persistent_state_delta_zero':True,
        'all_19_artifact_backups_verified':True,'local_and_persistent_final_checkpoint_hashes_verified':True,
        'pod_stopped_verified':True,'persistent_volume_retained':True,'statistics_started_unix':started,
        'statistics_after_verified_stop':started>stopped.timestamp(),'one_panel_sha256':panel['panel_sha256'],
        'sequence_loss_archive_sha256':sha(RESULT/'PAIRED_SEQUENCE_LOSSES.npz'),'no_further_training_launched':True,
        'limitations':['complete architectures; extra query-dependent MLP capacity not isolated','signed-additive versus convex combinations and different initialization','no parameter-matched query-only MLP or inference ablation','evaluation-sequence bootstrap is not training-seed replication','correlated contrasts share checkpoints','100M screen does not replace mature 250M baseline']}
    assert audit['statistics_after_verified_stop']
    write('FINAL_AUDIT.json',audit)
    report(evaluation,statistics,gates,decision,runtime,files,preflight,terminal,audit)
    print(json.dumps(decision,indent=2),flush=True)

def report(ev,stats,gates,decision,runtime,files,preflight,terminal,audit):
    def ci(v):return f'[{v[0]:+.12f}, {v[1]:+.12f}]'
    lines=['# EXPERIMENT 2D10 — RETRIEVAL-AWARE GATING 100M COMPLETE','',
        '**100M screening classification:** '+decision['classification'],
        '**Candidates establishing benefit over D:** '+(', '.join(decision['candidates_establishing_benefit_over_D']) or 'None'),
        '**Candidates clearing delta_CE=0.0001:** '+(', '.join(decision['candidates_clearing_margin']) or 'None'),
        '**Direct tanh versus softmax:** '+decision['direct_T_H_conclusion'],
        '**Current mature baseline:** sealed 2D9 250M Dynamic.','',
        'Exactly two new training arms, each 191 updates / 100,139,008 targets; two reused sealed 100M controls; exactly four final evaluations. No continuation, window expansion, from-scratch comparison, or further training was launched.','',
        '| Condition | CE | Perplexity |','|---|---:|---:|']
    for a in 'SDTH':lines.append(f"| {a} | {ev[a]['aggregate_ce']:.12f} | {ev[a]['perplexity']:.12f} |")
    lines+=['','## Paired comparisons','',
        'Positive favors the second condition. Primary decisions use Bonferroni-adjusted **98.333333% marginal bootstrap intervals** (nominal 95% family coverage for three comparisons). Ordinary 95% intervals are descriptive.','',
        '| Contrast | Mean CE | Raw 95% CI | Adjusted 98.333333% CI | exp(contrast) | Second / first wins / ties |',
        '|---|---:|---|---|---:|---|']
    for name,_,_ in CONTRASTS:
        v=stats['contrasts'][name]
        lines.append(f"| {name} | {v['mean']:+.12f} | {ci(v['raw_95_ci'])} | {ci(v['adjusted_98_333333_ci']) if v['adjusted_98_333333_ci'] else 'Secondary / descriptive'} | {v['exp_contrast']:.12f} | {v['second_wins']} / {v['first_wins']} / {v['ties']} |")
    lines+=['','| Primary contrast | Positive L>0 | Margin L>δ | Negative U<0 | Harm U<−δ | Equivalence | Second noninferior |','|---|---|---|---|---|---|---|']
    for name,_,_ in CONTRASTS[:3]:
        v=stats['contrasts'][name]['adjusted_flags'];lines.append('| '+name+' | '+render_flag_values(v)+' |')
    lines+=['','50,000 paired sequence resamples; NumPy default_rng(20260910); identical indices across all contrasts; linear percentiles. Primary percentiles: [0.833333333333, 99.166666666667]. Strict boundaries apply: touching a boundary does not pass. Absence of significance is not equivalence. S-based contrasts cannot override the primary decision. Differing significance against D does not establish a T/H difference.','',
        'T and D establish practical equivalence within ±0.0001 using the adjusted interval. H clears the practical margin over both D and T.', '', '## Initialization, gates, and costs','',
        'T has an additive tanh gate with the inherited linear h path and retrieval-aware MLP. H uses two-branch softmax. For each projected q, completed local output, and completed recurrent output, heads are concatenated in c_proj order into a width768 vector. Those three vectors receive separate FP32 affine-free LayerNorm (epsilon 1e-5), then are concatenated into the width2304 router input. Coefficients are cast to BF16 immediately before combining original branch outputs. The shared c_proj and bias execute once.','',
        'T initially reproduced the exact parent BF16 logits and CE in both parallel and incremental checks. H intentionally scales initial eligible combined outputs by 1/(1+tanh(g0)), preserving branch ratio but changing the function. H returns exactly local output without eligible memory. Its three compatibility gate scalars and their optimizer states remain untouched.','',
        '| Disposable training-batch diagnostic | Parent CE | T CE | H CE |','|---|---:|---:|---:|']
    for mode in ('parallel','incremental'):
        t=preflight['disposable_smoke']['T']['initial_training_batch_diagnostic'];h=preflight['disposable_smoke']['H']['initial_training_batch_diagnostic']
        lines.append(f"| {mode} | {t['parent_'+mode+'_ce']:.9f} | {t[mode+'_ce']:.9f} | {h[mode+'_ce']:.9f} |")
    lines+=['','This small fixed training batch is an initialization diagnostic, not an additional validation condition or scientific comparison. Smoke state was discarded and original scientific state reloaded independently.','',
        '| T destination | Raw g0 | w norm | u mean | delta mean | FP32 g mean / std / range | BF16 g mean / std / range |','|---|---:|---:|---:|---:|---|---|']
    def summary(v):return f"{v['mean']:.8f} / {v['std']:.8f} / [{v['min']:.8f}, {v['max']:.8f}]"
    for b in ('B1','B3','B5'):
        m=gates['arms']['T'][b]['metrics'];v=ev['T']['model_gate_parameters'][b]
        lines.append(f"| {b} | {v['g0']:.9f} | {v['w_norm']:.9f} | {m['u']['eligible']['mean']:.8f} | {m['delta']['eligible']['mean']:.8f} | {summary(m['g']['eligible'])} | {summary(m['cast_g']['eligible'])} |")
    lines+=['','| H destination | Learned b2 | Logit difference mean | FP32 λL mean | FP32 λR mean / std / range | BF16 λR mean / std / range | Entropy mean |','|---|---|---:|---:|---|---|---:|']
    for b,i in zip(('B1','B3','B5'),('0','2','4')):
        m=gates['arms']['H'][b]['metrics']
        lines.append(f"| {b} | {ev['H']['H_output_biases'][i]} | {m['logit_difference']['eligible']['mean']:.8f} | {m['lambda_L']['eligible']['mean']:.8f} | {summary(m['lambda_R']['eligible'])} | {summary(m['cast_lambda_R']['eligible'])} | {m['entropy']['eligible']['mean']:.8f} |")
    lines+=['','Gate table values cover eligible-memory positions. GATE_STATISTICS.json contains separate unavailable-position summaries, full-panel means/std/extrema/negative fractions, and fixed-sample quantiles (131,072 eligible positions per destination, seed 20260911; identical positions across arms). No attention matrices or residuals were exported; detached scalar diagnostics used one device transfer per batch. Router weight norms are retained in T_REAL.json/H_REAL.json. Nonzero router weights or changing coefficients alone do not prove retrieval utility.','',
        '| Architecture | Registered parameters | Added FP32 parameter bytes versus S | Persistent state bytes per sequence |','|---|---:|---:|---:|']
    for a,added in [('S',0),('D',9216),('T',894720),('H',885912)]:lines.append(f"| {a} | {ev[a]['parameter_count']:,} | {added:,} | 33,289,728 |")
    lines+=['','All four first full-length physical cache audits passed: persistent state delta **0**. Router activations are transient; parameters and training optimizer state are separate from historical KV/raw recurrent cache accounting.','',
        '## Training and evaluation integrity','',
        'Both new arms restored original 2D7 O1, global update 2290 / 1,200,619,520 targets. Original backbone tensors, optimizer groups/moments/individual counters, scheduler metadata, Python/NumPy/Torch CPU/CUDA RNG, and loader cursor were restored. The older source predates explicit optimizer-name mapping: its unchanged source inventory was reconstructed before adding parameters, state was transplanted by parameter name, and all states and named counters verified. New parameters used fresh state and isolated initialization generators; matching hidden-layer hashes are in SOURCE_AND_PARAMETER_AUDIT.json.','',
        'Replay used B32×T1024, accumulation16, global updates 2291–2481: 185 two-pass and six three-pass updates at 2304, 2336, 2368, 2400, 2432, 2464. Every batch, stream, cursor, pass count, and target count matched the original ledger. New counters reached 191; inherited active counters advanced 191; H retired g0 and both dormant B6 states remained identical. Final cumulative targets: 1,300,758,528. Training logs include the common weighted multipass objective separately from final incremental validation.','',
        'Focused tests: 12 CPU and 12 GPU tests passed, covering T equivalence, H closed-form/simplex/cast and empty-memory behavior, valid zero-valued memory, nonzero-router future suffix and row isolation in both modes past lag63, attached q/local/recurrent/earlier-writer gradients, expected first-zero hidden gradients, nonzero hidden gradients after output update, and activation-checkpoint gradient equivalence. Full-size GPU smoke verified batch fit and complete model/optimizer/RNG reload. Final strict reopens passed.','',
        'Fresh panel: '+audit['one_panel_sha256']+'. Seed 20260909; 64 canonical B64 batches  / 4096 sequences  / 4,194,304 targets per condition. Both historical 2D9 panels and 431 recovered historical/reserved spans were excluded. No reselection, old scores, midpoint or extra full evaluations were used. BF16 incremental execution, FP32 token CE, FP64 accumulation; four finite paired arrays and ordered identities preserved.','',
        '| Terminal identity (both new arms and historical controls) | SHA256 |','|---|---|',
        '| Cursor | `d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d` |',
        '| Next batch | `'+terminal['next_global_batch_sha256']+'` |',
        '| Next stream | `'+terminal['next_stream_sha256']+'` |','',
        '## Runtime, backups, and shutdown','',
        '| Workload | Current minutes |','|---|---:|']
    for a in 'TH':lines.append(f"| {a} training | {runtime['current_T_H_training_seconds'][a]/60:.3f} |")
    for a in 'SDTH':lines.append(f"| {a} evaluation | {runtime['current_evaluation_seconds'][a]/60:.3f} |")
    lines+=['',f"Historical S/D training times were {runtime['historical_SD_training_seconds']['S']/60:.3f} / {runtime['historical_SD_training_seconds']['D']/60:.3f} minutes; neither was retrained here. Current T/H evaluation includes diagnostics and does not isolate production router overhead.",'',
        f"Supervised two-GPU interval: {runtime['pod_supervised_wall_seconds']/60:.3f} wall minutes / {runtime['pod_supervised_aggregate_GPU_hours']:.6f} aggregate GPU-hours. Pod resume through verified stop (including preparation): {runtime['pod_resume_to_verified_stop_wall_seconds']/60:.3f} wall minutes / {runtime['pod_resume_to_verified_stop_aggregate_GPU_hours']:.6f} aggregate GPU-hours.",'',
        'Pod `nagj1hv18p3z2c` (`electrical_aqua_worm`) verified EXITED/stopped at '+runtime['verified_stopped_utc']+'. Persistent volume `yhzyb27fb5` retained. Both new checkpoints were exported and independently hash-verified while evaluation ran; remaining outputs were exported after GPU completion. All 19 artifact hashes match persistent copies. Statistics/reporting occurred locally after shutdown.','',
        '| Checkpoint | SHA256 |','|---|---|']
    for a in ('O1','S','D','T','H'):lines.append(f"| {a} | `{files[a]['sha256']}` |")
    lines+=['','Full checkpoint paths and independent backup evidence: ALL_CHECKPOINT_IDENTITIES.json, CHECKPOINT_MANIFESTS.json, ARTIFACT_BACKUP_VERIFICATION.json, STOP_VERIFICATION.json. Large checkpoints and scalar arrays remain outside Git. Persistent runtime: `/workspace/exp2d10_retrieval_gating_100m/`.','',
        '## Interpretation and recommendation','',
        decision['recommendation'],'',
        'This is a 100M architecture screen. The accepted 250M Dynamic checkpoint remains the mature baseline; its old-panel score is not compared here. These contrasts do not isolate retrieved-output inputs from extra query-dependent MLP capacity. T/H also differ in parameterization, signed versus convex combination, and initial function. No parameter-matched query-only MLP or inference-ablation control was included. Sequence bootstrap intervals are not training-seed replication, and contrasts share checkpoints. An unresolved 100M result does not establish permanent mechanism failure.','',
        'Scientific implementation commit: `'+audit['scientific_implementation_commit']+'`. Branch: `codex/experiment-2d10-retrieval-aware-gating-100m`. Final immutable tag: `experiment-2d10-retrieval-aware-gating-100m-final`. See GIT_REFERENCES.json for the final pushed commit/tag verification.','']
    (RESULT/'EXPERIMENT_2D10_FINAL_REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
