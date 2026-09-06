"""Local report, integrity audit, and compact result archival after provider stop."""
import argparse,math,shutil,time
from .common import *
from .analysis import analyze
from .artifacts import collect


def finalize(archive):
    archive=Path(archive);out=PACKAGE/'results'
    stop=read_json(archive/'STOP_VERIFICATION.json');assert stop['passed']
    for name in ('STOP_VERIFICATION.json','EXPORT_VERIFICATION.json','RUNTIME_ACCOUNTING.json','REMOTE_INPUT_IDENTITIES.json','PROVIDER_PREFLIGHT.json'):
        shutil.copyfile(archive/name,out/name)
    raw=archive/'raw';export=read_json(out/'EXPORT_VERIFICATION.json')
    assert all(sha256(raw/n)==h for n,h in export['files'].items())
    for name in ('PREFLIGHT_AUDIT.json','RUN_CONFIG.json','GPU_WORK_FINISHED.json'):
        shutil.copyfile(raw/name,out/name)
    result=analyze(raw,out);panel=read_json(FROZEN/'PANEL.json');data=collect(raw,panel)
    source=read_json(FROZEN/'SOURCE_IDENTITIES.json');gpu=read_json(raw/'GPU_WORK_FINISHED.json')
    assert sha256(source['checkpoint_path'])==CHECKPOINT_SHA
    tensor=source['model_tensor_identity'];runtime=read_json(out/'RUNTIME_ACCOUNTING.json')
    timings={}
    for condition in CONDITIONS:
        complete=read_json(raw/(condition+'_COMPLETE.json'))
        assert complete['tensor_identity_before']==complete['tensor_identity_after']==tensor
        files=sorted((raw/condition).glob('batch_*.json'))
        seconds=sum(read_json(p)['seconds'] for p in files)
        timings[condition]=dict(scoring_seconds=seconds,condition_elapsed_seconds=complete['seconds'],
            new_batches=complete['new_batches'],resumed_batches=complete['resumed_batches'])
        # Compact per-sequence raw export in Git; full atomic originals in local and volume archives.
        atomic_json(out/(condition+'_RAW.json'),dict(condition=condition,binding=read_json(raw/'RUN_CONFIG.json')['binding'],rows=data[condition],
            source_batch_hashes={p.name:sha256(p) for p in files}))
    pre=read_json(out/'PREFLIGHT_AUDIT.json');scientific=sum(v['scoring_seconds'] for v in timings.values())
    runtime.update(per_condition=timings,scientific_scoring_seconds=scientific,
        successful_preflight_seconds=pre['seconds'],
        residual_startup_transfers_verification_stop_and_unmeasured_overhead_seconds=runtime['cumulative_billed_seconds']-scientific-pre['seconds'],
        scientific_retries=0,final_gpu_finished_to_verified_stop_seconds=stop['verified_at']-gpu['time'])
    atomic_json(out/'RUNTIME_ACCOUNTING.json',runtime)
    checks=dict(five_conditions=list(data)==list(CONDITIONS),zero_scientific_training_updates=gpu['training_updates']==0,
        exact_sequence_condition_coverage=sum(map(len,data.values()))==20480,
        exact_target_coverage=sum(r['count'] for rows in data.values() for r in rows)==20971520,
        unchanged_model_tensors=gpu['tensor_identity_before']==gpu['tensor_identity_after']==tensor,
        unchanged_full_checkpoint=sha256(source['checkpoint_path'])==CHECKPOINT_SHA,
        panel_disjointness=read_json(FROZEN/'PANEL_DISJOINTNESS.json')['passed'],cpu_checks=read_json(out/'CPU_MODEL_AUDIT.json')['passed'],
        cuda_checks=pre['passed'],raw_durable_backup_hashes=all(sha256(raw/n)==h for n,h in export['files'].items()),
        two_independent_durable_raw_locations=export['persistent_storage'].startswith('/workspace/') and bool(export['local_archive']),
        stopped_provider=stop['provider']['desiredStatus']=='EXITED' and stop['provider']['runtimeStatus']=='stopped',
        within_compute_ceiling=runtime['within_ceiling'],analysis_after_stop=time.time()>stop['verified_at'])
    audit=dict(passed=all(checks.values()),checks=checks,time=time.time());atomic_json(out/'FINAL_AUDIT.json',audit);assert audit['passed']
    def interval(a):return '['+', '.join(f'{v:.9f}' for v in a)+']'
    report=['# Experiment 2D12 — frozen H10B recurrence ablation',
        'Completed all five conditions on one newly frozen 4,096-sequence panel. All effects below compare OFF against H_ON on the same sequences. Positive values mean recurrence removal hurts prediction.',
        '| Condition | CE (nats/target) | Perplexity | Scoring minutes |','|---|---:|---:|---:|']
    for c,s in result['scores'].items():report.append(f"| {c} | {s['ce']:.9f} | {s['perplexity']:.6f} | {timings[c]['scoring_seconds']/60:.2f} |")
    report+=['','## Primary paired sequence analysis','50,000 paired sequence resamples; isolated NumPy default_rng(20260922); FP64 means and linear percentiles. One common index vector is applied to all four contrasts per replicate. Conclusions use the Bonferroni-adjusted 98.75% marginal intervals for the four-contrast family.',
        '| OFF condition | ΔCE | Ordinary 95% CI | Adjusted 98.75% CI | Relative perplexity change |','|---|---:|---|---|---:|']
    for r in result['contrasts']:report.append(f"| {r['condition']} | {r['mean_delta_ce']:.9f} | {interval(r['sequence']['ci95'])} | {interval(r['sequence']['ci9875'])} | {r['relative_perplexity_percent']:.6f}% |")
    report+=['','![Four recurrence-removal contrasts](contrasts.png)','',
        'The reference is ±0.0001 nats/target. Flags are evaluated separately: harm requires adjusted lower bound >0; harm beyond reference requires >+0.0001; help requires adjusted upper bound <0; help beyond reference requires <−0.0001; equivalence requires both bounds within ±0.0001. False means that claim is unresolved. A small established effect can also meet equivalence.',
        '| Condition | Hurts | Hurts beyond reference | Helps | Helps beyond reference | Equivalent at reference |','|---|---|---|---|---|---|']
    for r in result['contrasts']:report.append('| '+r['condition']+' | '+' | '.join(str(v) for v in r['sequence']['flags'].values())+' |')
    report+=['','| Condition | Positive / negative / zero sequence fraction | Relative PPL 95% CI | Relative PPL adjusted CI |','|---|---|---|---|']
    for r in result['contrasts']:report.append(f"| {r['condition']} | {r['positive_fraction']:.6f} / {r['negative_fraction']:.6f} / {r['zero_fraction']:.6f} | {interval(r['relative_perplexity_ci95'])}% | {interval(r['relative_perplexity_ci9875'])}% |")
    report+=['','## Canonical-group sensitivity analysis','50,000 paired resamples of the 64 canonical B64 groups, keeping each group’s sequences together; isolated seed 20260923. This sensitivity analysis does not replace the prespecified sequence analysis.',
        '| Condition | Group 95% CI | Group adjusted 98.75% CI | Changed flags versus sequence analysis |','|---|---|---|---|']
    for r in result['contrasts']:report.append(f"| {r['condition']} | {interval(r['canonical_group']['ci95'])} | {interval(r['canonical_group']['ci9875'])} | {', '.join(r['sensitivity_flags_changed']) or 'None'} |")
    report+=['','| Condition | Group hurts | Group hurts beyond reference | Group helps | Group helps beyond reference | Group equivalent |','|---|---|---|---|---|---|']
    for r in result['contrasts']:report.append('| '+r['condition']+' | '+' | '.join(str(v) for v in r['canonical_group']['flags'].values())+' |')
    report+=['','Group-analysis flags are retained in PAIRED_ANALYSIS.json. Neither sampling unit measures variation across independent training runs or guarantees independence across documents.',
        '','## Interpretation']
    for r in result['contrasts']:
        f=r['sequence']['flags'];decision='removal hurts beyond the reference' if f['removal_hurts_beyond_reference'] else 'removal hurts' if f['removal_hurts'] else 'removal helps beyond the reference' if f['removal_helps_beyond_reference'] else 'removal helps' if f['removal_helps'] else 'direction is unresolved'
        report.append(f"{r['condition']}: {decision}. Practical equivalence at the reference is {'established' if f['practical_equivalence_at_reference'] else 'unresolved'}.")
    report += [f"Descriptive nonadditivity (ALL_OFF ΔCE minus the sum of individual ΔCEs): {result['nonadditivity_all_minus_sum_individual']:.9f} nats/target. No additional confirmatory test was performed.",
        'OFF removes the entire recurrent branch and restores the local coefficient to one. It does not isolate recurrent content while holding learned mixture weights fixed. Each condition evolves its own hidden states, writer rings, and enabled gates from empty sequence state. Individual effects need not add to ALL_OFF.',
        'These frozen interventions measure dependence under this removal rule. Harm does not establish how poorly a model trained without recurrence would perform; harmless removal would not rule out training benefit. The preceding temporal-gradient audit is not measured again here. Gate coefficients are not predictive contribution percentages. No effect is normalized by a historical G−H gap from another panel.',
        'Next recommendation: retain the measured frozen-model conclusion, then separately scope a matched from-scratch local-only control if the question is whether recurrence improves learning. That proposed training experiment has not been launched.',
        '','## Provenance and verification',
        f"Sealed 2D11 commit: `{SOURCE_COMMIT}`. Original tag preserved. New implementation commit: `{read_json(FROZEN/'CODE_IDENTITY.json')['commit']}`. Full checkpoint SHA-256: `{CHECKPOINT_SHA}`. Completed historical updates: 19,072; historical logical targets: 9,999,220,736; registered parameters: 124,697,386, including four unused scalars. Strict FreshH load and tied embeddings verified.",
        f"Model tensor identity before/after: `{tensor}`. Validation SHA-256: `{VAL_SHA}`. Panel identity: `{panel['identity']}`. Panel selection seed: 20260921; NumPy {panel['numpy_version']}; accepted order preserved. The exclusion union includes the sealed 2D11 audit and source manifests, both 2D11 panels, and all discovered project manifests with target offsets. Exactly 918 eligible batches were found; 64 selected. Every sequence token hash and target-span nonoverlap was independently checked.",
        'Native local windows remain W2/W32/W64 at B1/B3/B5 and W1024 elsewhere. Enabled recurrent source/lag mappings remain B12→B1 at 1–1023, B10→B3 at 31–1023, B8→B5 at 63–1023. OFF bypasses the router and recurrent read, uses exact local output, and applies the shared projection/bias once; residual and MLP operations remain intact.',
        f"CPU fixtures checked exact ON incremental and parallel equality; per-sequence CE; independent local-only blocks; empty/nonempty and changed bank/mask independence; router/read traps; lag boundaries; suffix causality; row isolation; resets; full 1,024-position ring bookkeeping and writer ownership in all conditions; full local-backbone parallel/incremental agreement; atomic batches, stale/duplicate rejection, paired coverage, bootstrap signs/flags, export hashing and stop retries. FP32 reference tolerances were atol 2e-6 / rtol 2e-5; actual maxima: `{read_json(out/'CPU_MODEL_AUDIT.json')['maxima']}`.",
        f"Disposable CUDA preflight used synthetic data. Device: {pre['device_name']}; physical batch: {pre['batch_size']}; torch {pre['torch_version']}; NumPy {pre['numpy_version']}. ON equality was exact. BF16 independent-reference tolerance was fixed before scoring at atol {pre['atol']} / rtol {pre['rtol']}; actual maxima: `{pre['maxima']}`. Full-length B{pre['batch_size']} memory and throughput fit passed. Scientific evaluation used eval + inference_mode, BF16 forward, explicit FP32 token CE and FP64 accumulation, deterministic algorithms, CUBLAS :4096:8, high matmul precision, no compilation, no warmup targets discarded, and no full gate/attention dumps.",
        'All five conditions retain exactly 33,289,728 BF16 persistent cache bytes per sequence at full length. Timing includes retained cache bookkeeping and skips disabled reads/routers. This is not a minimal-cache local-only deployment benchmark.',
        'Scientific coverage: 20,480 sequence-condition results and 20,971,520 target predictions (4,194,304 per condition). Zero optimizer construction, backward calls, training updates, new training targets, or checkpoint writes. Runtime traps prohibit optimizer/backward entrypoints. Full checkpoint file and model tensors unchanged. Per-sequence raw values, batch identity/config provenance, code hashes and backup hashes are preserved.',
        '','## Resources, backups and shutdown',
        f"Pod `{provider_id(stop)}`; one {pre['device_name']}; volume `yhzyb27fb5` preserved at 190 GB. Initially user-running pod was stopped for local preparation; its {runtime['initial_billed_seconds']:.2f} seconds are included. Cumulative conservatively measured billed wall/GPU time: {runtime['cumulative_billed_gpu_hours']:.6f} hours. Quoted rate: ${runtime['quoted_hourly_rate']:.2f}/hour. Compute cost at that rate: ${runtime['compute_cost_at_quoted_rate']:.6f}, below 3 GPU-hours and $10. Provider invoice is unavailable; storage charges are separate.",
        f"Scientific scoring: {scientific:.2f} seconds. Successful disposable preflight: {pre['seconds']:.2f} seconds. Remaining startup, transfer, verification, shutdown and unmeasured overhead: {runtime['residual_startup_transfers_verification_stop_and_unmeasured_overhead_seconds']:.2f} seconds. Transfers overlapped scoring. Scientific retries: 0; earlier staging attempts: {runtime.get('startup_retry_attempts',0)}. GPU completion to verified stop: {runtime['final_gpu_finished_to_verified_stop_seconds']:.2f} seconds.",
        f"All raw GPU outputs have independently hash-verified copies at `{archive/'raw'}` and `{export['persistent_storage']}` on the existing volume. Two independent remote hash passes and a local hash pass preceded shutdown. The unchanged 1.65 GB checkpoint was not re-exported. Provider EXITED / stopped was verified at Unix time {stop['verified_at']:.6f}; bootstrap, plotting and this report ran afterward.",
        'An initial staging attempt failed during tar extraction. The likely cause was ownership restoration on the persistent filesystem; the original stderr was not captured. The attempt ran no CUDA preflight or science, was stopped and archived, and its billed interval is included. The corrected extraction uses the established --no-same-owner option; no scientific definition or panel changed.',
        'The final audit passed every recorded check. Final Git branch/tag identities are recorded separately in GIT_ARCHIVAL.json after the report commit and verified remote push.' ]
    atomic_bytes(out/'FINAL_REPORT.md',('\n'.join(report)+'\n').encode())
    hashes={str(p.relative_to(PACKAGE)):sha256(p) for folder in (FROZEN,out) for p in folder.rglob('*') if p.is_file() and p.name not in ('ARTIFACT_HASHES.json','GIT_ARCHIVAL.json') and p.suffix not in ('.log',)}
    atomic_json(out/'ARTIFACT_HASHES.json',hashes)
    shutil.copytree(out,archive/'final',dirs_exist_ok=True);shutil.copytree(FROZEN,archive/'frozen',dirs_exist_ok=True)
    print(dict(passed=True,report=str(out/'FINAL_REPORT.md'),scores=result['scores']))

def provider_id(stop):return stop['provider']['id']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);finalize(p.parse_args().archive)
