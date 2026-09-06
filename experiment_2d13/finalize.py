"""Local complete audit and report; prohibited before provider shutdown."""
import argparse,csv,json,shutil,time
import torch
from .common import *
from .checkpoints import validate
from .model import tensor_identity
from .analysis import analyze,plots
from .closeout import audit_closeout

def finalize(archive):
    stop=read_json(archive/'STOP_VERIFICATION.json');assert stop['passed'] and stop['provider']['desiredStatus']=='EXITED' and stop['provider']['runtimeStatus']=='stopped'
    billing=read_json(archive/'RUNTIME_ACCOUNTING.json');assert billing['within_ceiling']
    raw=archive/'raw';out=PACKAGE/'results';out.mkdir(exist_ok=True)
    export=read_json(archive/'EXPORT_VERIFICATION.json');assert export['passed']
    for name,row in export['files'].items():assert sha256(raw/name)==row['sha256'] and (raw/name).stat().st_size==row['bytes']
    result=read_json(raw/'GPU_WORK_FINISHED.json');assert result['passed'] and result['completed_updates']==1000
    plan=[json.loads(l) for l in (FROZEN/'stream_plan.jsonl').read_text().splitlines()];expected=[json.loads(l) for l in (FROZEN/'expected_batches.jsonl').read_text().splitlines()]
    metrics=[json.loads(l) for l in (raw/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==1000
    for u,(a,b) in enumerate(zip(metrics,expected),1):
        assert a['completed_updates']==u and a['data']==b['data'] and a['lr']==b['lr'] and a['lr_index']==u-1
        for k,v in accounting(u).items():assert a[k]==v
    audits={};torch.set_num_threads(4)
    for u in [0,250,500,750,1000]:
        path=raw/f'checkpoints/u{u:05d}.pt';p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        audits[str(u)]=validate(p);assert sha256(path)==read_json(str(path)+'.manifest.json')['sha256']
    assert p['loader']==plan[-1]['after'] and p['next_lr_index']==1000
    original=torch.load(archive/'local_initial.pt',map_location='cpu',mmap=True,weights_only=False)
    changed={n:not torch.equal(v,original['model'][n]) for n,v in p['model'].items()}
    active=read_json(FROZEN/'TRAINABILITY.json')['active_names'];inactive=read_json(FROZEN/'TRAINABILITY.json')['frozen_names']
    assert all(changed[n] for n in active) and not any(changed[n] for n in inactive)
    for _,entry in read_json(FROZEN/'SOURCE_IDENTITIES.json')['inputs'].items():assert sha256(entry['path'])==entry['sha256']
    panel=read_json(FROZEN/'PANEL.json');l=read_json(raw/'evaluations/L524M_COMPLETE.json');h=read_json(raw/'evaluations/H524M_COMPLETE.json')
    closeout=audit_closeout(archive,audits,metrics)
    a=analyze(l,h,panel);atomic_json(out/'ANALYSIS.json',a)
    lmon=[]
    for u in SCHEDULE:
        v=read_json(raw/f'evaluations/L_monitor_u{u:05d}_COMPLETE.json');assert v['count']==1280*1024
        assert v['binding']['panel_identity']==read_json(FROZEN/'MONITOR.json')['identity']
        lmon.append(dict(update=u,ce=v['ce']))
    hmon=read_json(FROZEN/'H_MONITOR.json');plots(out,a,lmon,hmon,metrics)
    with (out/'endpoint_sequences.csv').open('w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['sequence_id','canonical_group','token_sha256','CE_L','CE_H','difference'])
        for r,s in zip(l['rows'],h['rows']):w.writerow([r['id'],r['canonical_group'],r['token_sha256'],r['ce'],s['ce'],r['ce']-s['ce']])
    train_seconds=sum(r['training_seconds'] for r in metrics);historical=read_json(FROZEN/'H_PREFIX.json')
    audit=dict(passed=True,completed_at=time.time(),checkpoints=audits,stream_all_1000_hashes_matched=True,terminal_cursor=p['loader'],optimizer_all_active_steps=1000,next_lr_index=1000,active_tensors_all_changed=True,inactive_tensors_unchanged=True,source_artifacts_unchanged=True,initial_base_identity=BASE_SHA,registered_parameters=124697386,active_parameters=124475904,trainable_difference_from_H=221478,backup_files=len(export['files']),budget_compliant=True,provider_stopped=True,one_scientific_arm=True,new_training_seconds=train_seconds,new_training_gpu_hours=train_seconds/3600,historical_training_seconds=historical['training_seconds'],historical_training_gpu_hours=historical['gpu_hours'],preflight=read_json(raw/'PREFLIGHT.json'),billing=billing)
    audit['closeout']=closeout
    atomic_json(out/'FINAL_AUDIT.json',audit)
    for n in ['STOP_VERIFICATION.json','RUNTIME_ACCOUNTING.json','EXPORT_VERIFICATION.json','STAGING.json','PROVIDER_BEFORE_CUDA.json','REMOTE_LAUNCH.json']:shutil.copyfile(archive/n,out/n)
    for n in ['GPU_WORK_FINISHED.json','PREFLIGHT.json','CUDA_OBJECTIVE.json','CUDA_ACCUMULATION.json','CUDA_RESUME.json','CUDA_STRUCTURAL.json','metrics.jsonl']:shutil.copyfile(raw/n,out/n)
    shutil.copytree(raw/'evaluations',out/'evaluations',dirs_exist_ok=True)
    for source in raw.glob('*.json'):shutil.copy2(source,out/source.name)
    (out/'checkpoint_manifests').mkdir(exist_ok=True)
    for source in (raw/'checkpoints').glob('*.json'):shutil.copy2(source,out/'checkpoint_manifests'/source.name)
    s=a['primary'];g=a['group_sensitivity'];meaning='H has lower endpoint CE' if s['lo']>0 else 'L has lower endpoint CE' if s['hi']<0 else 'The paired interval does not establish an endpoint benefit for either model'
    recommendation='A longer matched local-only comparison is worth testing, subject to separate authorization.' if s['hi']<.0001 else 'This screen does not support replacing H with the local-only recipe; a longer run is not implied.'
    timing_table='\n'.join(f"| {k.replace('_',' ')} | {v:.3f} |" for k,v in closeout['stage_seconds'].items())
    flag_table='\n'.join(f"| {k.replace('_',' ')} | {v} | {g['flags'][k]} |" for k,v in s['flags'].items())
    cpu_probes=read_json(out/'CPU_AUDIT.json')['objective']['probes'];cuda_probes=read_json(out/'CUDA_OBJECTIVE.json')['probes']
    text=f'''# Experiment 2D13 — fresh local-only at 524M targets

{meaning}. {recommendation} No continuation or additional experiment was launched.

Exactly one new scientific training arm completed 1,000 updates / 524,288,000 logical targets. L started only from the exact saved untrained H backbone tensors (identity `{BASE_SHA}`). H u01000 was reused only as an evaluation comparator, never as L initialization or optimizer state. L retained W2/W32/W64 and bypassed all recurrent reads/routers throughout training and inference.

| Endpoint | CE | Perplexity |
|---|---:|---:|
| L524M | {a['L_CE']:.10f} | {a['L_PPL']:.8f} |
| H524M | {a['H_CE']:.10f} | {a['H_PPL']:.8f} |

Primary paired difference CE(L)−CE(H): **{s['mean']:.10f}**, sequence-bootstrap 95% interval **[{s['lo']:.10f}, {s['hi']:.10f}]**. Positive favors H. Relative PPL difference {s['relative_ppl_percent']:.7f}% with transformed interval [{s['relative_ppl_interval_percent'][0]:.7f}%, {s['relative_ppl_interval_percent'][1]:.7f}%]. Both conditions used one fresh paired 4,096-sequence panel and the same B{l['binding']['batch_size']} true incremental BF16 forward, FP32 token CE and FP64 NLL accumulation. Coverage: 8,192 sequence-condition rows / 8,388,608 final target predictions; five L monitors add 6,553,600. Every sequence has 1,024 scored targets.

Flags at reference ΔCE=0.0001:

| Flag | Primary sequence CI | Group sensitivity CI |
|---|---|---|
{flag_table}

Failure to establish benefit is not equivalence. The reference is a small-effect convention, not a deployment-value threshold.

Canonical B64 group sensitivity: [{g['lo']:.10f}, {g['hi']:.10f}]. The sampling unit does not change any flag for this run. Both bootstraps used 50,000 FP64 resamples, seeds 20260925/20260926 and linear percentiles. Neither interval estimates independent training-seed variability. Sequence fractions: L wins {a['L_win_fraction']:.6f}, H wins {a['H_win_fraction']:.6f}, ties {a['tie_fraction']:.6f}.

![Monitoring](monitoring.png)

![Paired effect](paired_effect.png)

The monitoring curves reuse the sealed 1,280-sequence panel and historical H incremental scores. H's multipass weighted training objective is not overlaid with L's single-pass training objective. The separate `L_training.png` plots only L training CE.

| Exposure | Historical H524M | New L524M |
|---|---:|---:|
| Updates | 1,000 | 1,000 |
| Logical targets | 524,288,000 | 524,288,000 |
| CE pass-equivalent updates | 2,031 | 1,000 |
| CE pass-target evaluations | 1,064,828,928 | 524,288,000 |
| Mean actual CE passes | 2.031 | 1 |

H used 969 two-pass and 31 three-pass updates. Activation-checkpoint recomputation is excluded from exposure but included in measured compute. L registered 124,697,386 parameters, with 124,475,904 trainable backbone parameters and 221,482 frozen compatibility/router parameters. H had 221,478 more trainable parameters; equal registered counts do not imply equal active capacity.

Training-only measured time: L {train_seconds:.6f} seconds on one A100-SXM4-80GB ({train_seconds/3600:.6f} GPU-hours); historical H {historical['training_seconds']:.9f} seconds on four GPUs ({historical['gpu_hours']:.9f} GPU-hours), independently recovered from its first 1,000 metric rows. This compares different hardware counts and scheduling and is not a FLOP-matched or isolated same-configuration speed benchmark.

Cumulative new billed interval including user-requested held preparation, input verification, disposable preflight, training, monitoring/final scoring, checkpoint/export and stop latency: {billing['gpu_hours']:.6f} GPU-hours, ${billing['compute_cost']:.6f} at the verified $1.59/hour quote. Within both ceilings. Provider invoice unavailable; storage charges are separate. Historical H compute is sunk and excluded. Stage measurements, projection, and residual overhead are in the audit and runtime records; no GPU was restarted for analysis.

The first attempt was invalidated after 82 updates (42,991,616 targets; 357.520 seconds), because restoring the CPU template had overwritten the fresh CUDA optimizer's fused flag with False. All 82 updates were discarded. Its repeated u0 monitor (1,310,720 targets), preflight, checkpoint and raw logs are retained separately. The corrected run restored the exact saved untrained tensors and initial stream/RNG state, reran the required checks, and completed the full 1,000 valid updates with fused=True enforced on restoration and every training step. No trained weights from the discarded attempt were reused.

A historical implementation mismatch was discovered before endpoint scoring: H's saved initial and u1000 optimizers serialize fused=False, whereas this protocol explicitly requires fresh fused CUDA AdamW for L. L follows that explicit requirement. Hyperparameters, data, effective batch and LR schedule match, but optimizer implementation does not; the comparison therefore cannot be described as differing only in recurrence or as bitwise execution matched. This limitation is recorded in `H_OPTIMIZER_IMPLEMENTATION.json`.

| Measured stage | Seconds |
|---|---:|
{timing_table}

Valid evaluation time comprises five monitors ({sum(v for k,v in closeout['evaluation_seconds'].items() if 'monitor' in k):.3f} seconds), L final ({l['seconds']:.3f}) and H final ({h['seconds']:.3f}). The two preflights each used six disposable complete updates plus objective-specific probes, all excluded from scientific counters. Checkpoint serialization/reopen/hash work totaled {closeout['checkpoint_serialization_seconds_overlapping_other_stages']:.3f} seconds and overlaps other activity; it must not be added again to the billed interval. The residual includes local preparation with the pod held at the user's request, input/identity checks, snapshot/transfer work, idle gaps and shutdown. These components cannot all be separately identified from the available telemetry. Individual resumable transport timeout counts were not persisted; their time remains included in total billing.

Other recorded repairs were the CPU reference fixture's mutable optimizer-state aliasing, a literal floating-point LR test assertion, and the transfer controller heartbeat/timeout handling. One staging controller was aborted before launching CUDA during the optimizer regression repair. The user expanded the assigned volume from 190 to 200 GB during training; strict pod/volume identity checks were preserved while capacity became informational. Mocked 190/200 responses and read-only live status passed. Both active local supervision processes were replaced and their loaded executable code fingerprints checked; remote training PID 3287 continued unchanged. The corrected controller subsequently issued the real completion stop and verified EXITED/stopped. No pod stop was used to test that repair.

All 1,000 per-update hashes/cursors matched H before their updates. Terminal cursor `{p['loader']}`; original LR prefix and all optimizer step counters verified through 1,000. Every active tensor changed, inactive tensors retained their exact initial identities. Checkpoints u0/u250/u500/u750/u1000 were reopened and validated, and independent persistent/local byte hashes matched. u1000 contains the completed training state and a pending endpoint evaluation ledger; identity-bound completed raw outputs reconcile that ledger on resume without rescoring. All required checkpoints and raw evaluations are preserved in `{archive}` and `{export['persistent_root']}` on volume `yhzyb27fb5`.

Provider pod `{stop['provider']['id']}` verified `EXITED` / `stopped`; volume retained at {closeout['volume_size_GB']} GB (resized by the user). Source H SHA `{H_SHA}`. New checkpoint hashes: {json.dumps({u:read_json(raw/f'checkpoints/u{int(u):05d}.pt.manifest.json')['sha256'] for u in audits},sort_keys=True)}.

The scientific CUDA bundle remained fixed at SHA `{sha256(archive/'bundle.tar.gz')}`; its frozen code identity is `{closeout['scientific_code_identity']}`. Later changes affected local supervision and post-shutdown reporting only. Both earlier sealed worktree HEADs and their tracked files were rechecked unchanged.

CPU FP32 and full-size CUDA BF16 tests covered one versus explicit two/three pass objectives at initial and disposable noninitial weights with checkpointing on/off, every active gradient, absent inactive gradients, clipping, AdamW effects, original global normalization and schedule-boundary save/reload. CPU FP32 maximum absolute gradient discrepancy was {max(r['gradient_max_abs'] for r in cpu_probes):.9g}. Across the eight CUDA BF16 objective probes, loss discrepancy was {max(r['loss_max_abs'] for r in cuda_probes):.9g}, maximum global gradient relative L2 error {max(r['gradient_relative_l2'] for r in cuda_probes):.9g}, minimum cosine {min(r['gradient_cosine'] for r in cuda_probes):.12f}, and maximum post-AdamW parameter discrepancy {max(r['parameter_max_abs'] for r in cuda_probes):.9g}. The accumulation comparison had relative L2 error {read_json(out/'CUDA_ACCUMULATION.json')['relative_l2']:.9g}; schedule-boundary resume reproduced parameters with zero error. Per-tensor gradients, finite/zero patterns, clipping norms and optimizer state differences are archived against the prespecified tolerances. Independent incremental references, row isolation, causality and reset checks passed before final scoring. Disposable work is recorded separately from scientific updates.

This result compares the full H training recipe with a fresh local-only control at 524M. It does not uniquely attribute gains to attached temporal gradients, settle the 10B comparison, or erase the distinction from 2D12's frozen H10B OFF experiment. No architecture adoption or further training is authorized by this report.

Final Git archive: branch `codex/experiment-2d13-local-only-from-scratch-500m`; tag `experiment-2d13-local-only-from-scratch-500m-final`. Large checkpoints remain in the verified durable archives, outside Git. The local archive receipt `GIT_ARCHIVAL.json` records the verified remote commit and tag identities.
'''
    atomic_bytes(out/'FINAL_REPORT.md',text.encode());print(dict(passed=True,L_CE=a['L_CE'],H_CE=a['H_CE'],difference=s['mean'],CI=[s['lo'],s['hi']]),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args();finalize(a.archive)
