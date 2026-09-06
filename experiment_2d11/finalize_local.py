"""Package verified stopped-run evidence and final scientific interpretation locally."""
import shutil
from .common import *
from .analysis import collect,csv_file,figure_save


def finalize(archive):
    archive=Path(archive);run=archive/'run';out=run/'analysis'
    assert read_json(run/'STOP_VERIFICATION.json')['passed']
    exports=read_json(run/'FINAL_EXPORTS_VERIFIED.json');assert exports['passed']
    for record in exports['independently_hashed_artifacts']:
        p=run/record['path'];assert p.stat().st_size==record['bytes'] and sha256(p)==record['sha256'],str(p)
    manifest=read_json(run/'checkpoints/u19072.pt.manifest.json')
    assert sha256(run/'checkpoints/u19072.pt')==manifest['sha256']
    metrics,evals,historical=collect(run)
    plan=[json.loads(x) for x in (FROZEN/'stream_plan.jsonl').read_text().splitlines()]
    assert len(metrics)==19072
    for m,p in zip(metrics,plan):
        assert m['completed_updates']==p['update'] and m['pass_count']==p['pass_count']
        assert m['data']['before']==p['before'] and m['data']['after']==p['after']
        assert m['logical_targets']==p['update']*p['logical_targets'] and m['data']['target_count']==TARGETS
        assert len(m['data']['actual_token_sha256'])==64
    assert set(evals)=={f'ce:{u}' for u in CE_SCHEDULE}|{f'hella:{u}' for u in HELLA_SCHEDULE}|{'final:H:ce','final:G:ce','final:G:hella'}
    for key,e in evals.items():
        n=4096 if key in ['final:H:ce','final:G:ce'] else 10042 if 'hella' in key else 1280
        assert e['ownership_verified'] and e['example_count']==n and len({r['id'] for r in e['rows']})==n
    audit=read_json(out/'AUDIT.json');audit.update(all_exported_artifact_hashes_rechecked=True,
       final_checkpoint_hash_rechecked=True,all_update_cursors_pass_cadence_verified=True,
       complete_evaluation_schedule_and_ownership=True)
    assert all(audit.values());atomic_json(out/'AUDIT.json',audit)
    resource=read_json(out/'RESOURCE_ACCOUNTING.json');paired=read_json(out/'PAIRED_ENDPOINTS.json');ce=paired['ce'];hs=paired['hellaswag']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,result,scale,title in zip(axes,[ce,hs],[1,100],['Fresh-panel CE benefit G − H (nats)','HellaSwag benefit H − G (percentage points)']):
        point=result['estimate']*scale;lo,hi=[x*scale for x in result['ci97_5']]
        ax.errorbar(point,0,xerr=[[point-lo],[hi-point]],fmt='o',capsize=6)
        ax.axvline(0,color='black',lw=1,ls='--');ax.set_yticks([]);ax.set_xlabel(title);ax.set_title(result['classification'].capitalize());ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired bootstrap: 97.5% marginal intervals, 50,000 resamples');figure_save(fig,out/'paired_final_endpoints')
    bench=[read_json(p) for p in sorted((run/'preflight').glob('BENCHMARK_RANK*.json'))]
    pf=read_json(run/'PREFLIGHT.json');rows=[]
    for key in ['final:G:ce','final:H:ce','final:G:hella','hella:19072']:
        e=evals[key];rows.append(dict(key=key,examples=e['example_count'],seconds=e['elapsed_seconds'],examples_per_second=e['example_count']/e['elapsed_seconds']))
    csv_file(out/'evaluation_throughput.csv',rows)
    hardware=dict(disposable_preflight=pf['measurements'],inference_min_rank_tokens_per_second={k:min(r['inference'][k]['tokens_per_second'] for r in bench) for k in ['G','H']},h_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in metrics),parameters=dict(G=124475904,H_registered=124697386,H_active=124697382))
    atomic_json(out/'HARDWARE_SUMMARY.json',hardware)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].bar(['G CE','H CE','G Hella','H Hella'],[r['examples_per_second'] for r in rows]);axes[0].set_ylabel('Examples / second (4 GPUs)');axes[0].set_title('Final evaluation throughput')
    axes[1].bar(['G','H'],[124475904/1e6,124697386/1e6]);axes[1].set_ylabel('Registered parameters (millions)')
    axes[2].bar(['H training'],[hardware['h_peak_allocated_bytes']/1e9]);axes[2].set_ylabel('Rank-0 peak allocated VRAM (GB)')
    figure_save(fig,out/'evaluation_and_model_resources')
    comparisons=[]
    for name in ['500M','1B','2_62144B','5_24288B','7_86432B']:
        d=read_json(run/f'INTERIM_{name}.json');comparisons.append(dict(label=name,**d))
    def interval(x,scale=1):return '['+', '.join(f'{v*scale:.6f}' for v in x)+']'
    table='| Trained targets | H CE | Historical G CE | H HellaSwag | Historical G HellaSwag |\n|---:|---:|---:|---:|---:|\n'
    for d in comparisons:
        targets=d.get('logical_targets',d.get('targets'));table+=f"| {targets:,} | {d['h_ce']:.6f} | {d['g_ce']:.6f} | {100*d['h_hellaswag']:.4f}% | {100*d['g_hellaswag']:.4f}% |\n"
    audit_details=out/'TECHNICAL_DETAILS.md';shutil.copy2(out/'FINAL_REPORT.md',audit_details)
    report=f'''# Experiment 2D11 — COMPLETED

One fresh O1 + H model completed **19,072 updates / 9,999,220,736 processed logical targets**. The existing completed GPT-2 control was reused. H used 38,740 CE pass-equivalent updates / 20,310,917,120 CE pass-targets (mean 2.03125 passes). Both preregistered early screens permitted continuation. All requested intermediate comparisons completed without waiting for user responses.

## Final paired comparison

| Endpoint | H | Existing G | H benefit | Adjusted 97.5% interval | Classification |
|---|---:|---:|---:|---|---|
| Fresh 4,096-sequence CE | {ce['h_ce']:.8f} | {ce['g_ce']:.8f} | {ce['estimate']:.8f} nats (G − H) | {interval(ce['ci97_5'])} | **Benefit** |
| Full 10,042-example HellaSwag | {hs['h_accuracy']*100:.4f}% | {hs['g_accuracy']*100:.4f}% | {hs['percentage_points']:.4f} pp (H − G) | {interval(hs['ci97_5'],100)} pp | **Unresolved** |

H lowers fresh-panel perplexity by {-100*ce['relative_perplexity_change_h_over_g']:.4f}%. Its CE is lower on {ce['h_sequence_wins']:,}/4,096 sequences, with {ce['ties']} ties. Descriptive 95% intervals: CE {interval(ce['ci95'])} nats; HellaSwag {interval(hs['ci95'],100)} pp. The adjusted CE lower bound exceeds the historical 0.0001-nat small-effect reference; that reference does not establish value for the extra compute.

Paired HellaSwag correctness: both wrong 6,523; G wrong/H right 475; G right/H wrong 414; both right 2,630. Thus H gets 61 more examples correct. Its adjusted interval crosses zero, so these results **do not support a blanket overall-superiority claim**.

Bootstrap: 50,000 paired resamples, isolated seeds 20260918 (CE) and 20260919 (HellaSwag), NumPy 2.5.2, linear percentiles. Bonferroni 97.5% marginal intervals govern the two endpoint classifications. This is one trajectory per recipe, not seed replication or a causal isolation of the gate. H uses additional multipass compute.

## Matched intermediate comparisons

{table}
These use the original G aggregate records; no paired confidence intervals are available for intermediate G weights. The final historical-style H monitoring CE is {evals['ce:19072']['score']:.8f}. The saved G original parallel evaluator reproduced its historical 3.0750441551208496 CE exactly in preflight. The fresh-panel paired scores above use the same incremental evaluation for both models and are the primary endpoint.

## Runtime, compute, and recovery

The exact assigned four-A100-80GB pod is **verified stopped**. Total cumulative billed time, including all startup/repair attempts, was **{resource['cumulative_billed_seconds']/3600:.4f} pod-hours / {resource['cumulative_gpu_hours']:.4f} GPU-hours**, below 24 hours. At the quoted $6.36/pod-hour, estimated compute cost is **${resource['estimated_compute_dollars']:.2f}**, excluding storage.

Retained scientific training: {resource['training_seconds']/3600:.4f} hours. Scheduled scientific evaluations: {resource['evaluation_seconds']/60:.2f} minutes. Successful disposable preflight: {resource['preflight_seconds']/60:.2f} minutes. A controller network failure caused a verified pod stop and recovery from u6000, discarding 104 later updates / {resource['discarded_recovery_training_seconds']:.2f} training seconds; the raw tail remains archived. All billed attempts are included. The remaining {resource['residual_excluding_discarded_training_seconds']/60:.2f} minutes combine earlier failed setup/preflight, saves, transfer tail, shutdown and idle time; those categories cannot be separately reconstructed reliably and are not presented as pure idle time.

Final GPU completion to verified provider stop took {resource['final_gpu_finished_to_verified_stop_seconds']/60:.2f} minutes, used for independent checkpoint/artifact export and shutdown. Transfers overlapped training. Some earlier transfer acknowledgements were refreshed after recovery, so their durations do not reconstruct all historical transfer time. The billed-time chart uses the correct segment on each side of the restart and excludes stopped gaps.

The recovery restored verified model/optimizer/RNG/loader state and fixed identities. The first replayed update's loss and gradient norm matched; later repeated-step losses differed slightly (maximum observed over 23 steps: 7.3433e-5). We do not claim multi-step bitwise reproduction. The preregistered disposable next-step parameter/optimizer resume audit passed on all four ranks with zero maximum parameter error. Details and raw recovery evidence are retained.

Same-hardware disposable update timing: G one-pass {pf['measurements']['g_update_seconds']:.3f}s; H two-pass {pf['measurements']['two_pass_seconds']:.3f}s; H three-pass {pf['measurements']['three_pass_seconds']:.3f}s. Incremental B4 inference (768-token prime, 256 timed tokens), minimum rank throughput: G {hardware['inference_min_rank_tokens_per_second']['G']:.2f} tokens/s; H {hardware['inference_min_rank_tokens_per_second']['H']:.2f} tokens/s. These are disposable preflight-weight measurements, not final-weight throughput. Historical G ran on one A100, so a four-GPU H wall-time comparison is not architecture speedup.

H has 124,697,386 registered / 124,697,382 active parameters; G has 124,475,904. Persistent BF16 state per sequence: H 33,289,728 bytes, G 37,711,872 bytes. Rank-0 peak training allocation: {hardware['h_peak_allocated_bytes']/1e9:.3f} GB. See resource CSV/JSON and figures for evaluation throughput and measured training/GPU-hour coordinates.

## Integrity and artifacts

All {len(exports['independently_hashed_artifacts'])} exported science-artifact hashes were rechecked locally. Every update's loader cursor and pass cadence match the frozen plan; all 24 monitoring CE panels, seven HellaSwag panels and final paired panels have complete unique ownership. Initial tensor identity, distributed gradient checks, checkpoint optimizer/RNG audits, terminal cursor, budget and provider stop all pass. Actual input token hashes are retained per update. CUDA determinism and controller fixes are documented separately from the frozen scientific identity.

Final H checkpoint (outside Git): `{run/'checkpoints/u19072.pt'}`.
Persistent copy: `{manifest['persistent_path']}` on network volume yhzyb27fb5 (190 GB).
SHA-256: `{manifest['sha256']}`; {manifest['bytes']:,} bytes.

The scientific bundle identity remains commit `54845a06bac1361e518c6729ed46f547553d21e3` plus patch SHA `f43e4de57af847bb67f3f23f17f7551e4afa90ca4db0f30e0cc0c665311497ce`. Operational recovery patch SHA: `ea58cb231892b91e2d06b26eb64549b1185b7c6a56ace16ab1c3b8b3264d259c`. The final result commit/tag bind the packaged evidence and local analysis code. Model weights and datasets are excluded from Git. No further training is authorized.

See `AUDIT.json`, `PAIRED_ENDPOINTS.json`, `RESOURCE_ACCOUNTING.json`, `HARDWARE_SUMMARY.json`, CSV/JSON data, PNG/PDF figures, and `TECHNICAL_DETAILS.md` here; complete run evidence is packaged alongside this analysis.

{INTERPRETATION}
'''
    atomic_bytes(out/'FINAL_REPORT.md',report.encode())
    destination=REPO/'experiment_2d11/results';destination.mkdir(exist_ok=True)
    for p in run.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.md','.log','.csv','.png','.pdf']:
            target=destination/'run'/p.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ['RECOVERY_REPLAY_CHECK.json','NETWORK_RECOVERY_READY.json','USER_MILESTONE_CONFIRMATION.json']:
        if (archive/name).exists():shutil.copy2(archive/name,destination/name)
    for label in ['startup_attempt_01','startup_attempt_02']:
        for p in (archive/label).glob('*.json'):
            if p.name in ['BILLING_ACCOUNTING.json','STOP_VERIFICATION.json','CONTROLLER_FAILURE.json','INDEPENDENT_LOCAL_STOP_VERIFICATION.json','INDEPENDENT_LOCAL_STOP_REASON.json']:
                target=destination/'earlier_attempts'/label/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    atomic_json(destination/'RESULT_MANIFEST.json',dict(outcome='COMPLETED',time=time.time(),
       files={str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='RESULT_MANIFEST.json'}))
    print('FINAL_LOCAL_AUDIT_PASS',len(exports['independently_hashed_artifacts']),flush=True)

if __name__=='__main__':
    import sys
    finalize(sys.argv[1])
