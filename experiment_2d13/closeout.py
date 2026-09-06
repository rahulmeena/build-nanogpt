"""Additional local provenance, coverage, retry and resource audit after stop."""
import hashlib,json,shutil,subprocess,tarfile,time
import torch
from .common import *
from .evaluate import collect

def audit_closeout(archive,audits,metrics):
    stop=read_json(archive/'STOP_VERIFICATION.json')
    assert stop['passed'] and stop['provider']['desiredStatus']=='EXITED' and stop['provider']['runtimeStatus']=='stopped'
    assert stop['provider']['id']=='y96gntb89tzuvj'
    assert stop['provider']['networkVolumeId']==stop['provider']['networkVolume']['id']=='yhzyb27fb5'
    raw=archive/'raw';out=PACKAGE/'results';sources=read_json(FROZEN/'SOURCE_IDENTITIES.json')
    assert audits['0']['model_tensor_identity']==sources['L_initial_tensor_identity']
    code=identity(read_json(FROZEN/'CODE_IDENTITY.json'));config=identity(read_json(FROZEN/'EXECUTION.json'))
    with tarfile.open(archive/'bundle.tar.gz','r:gz') as bundle:
        for name,digest in read_json(FROZEN/'CODE_IDENTITY.json')['files'].items():
            assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest()==digest,name
    coverage={};evaluation_seconds={}
    labels=[(f'L_monitor_u{u:05d}',u,'MONITOR.json') for u in SCHEDULE]+[('L524M',1000,'PANEL.json'),('H524M',None,'PANEL.json')]
    for label,u,panel_file in labels:
        summary=read_json(raw/f'evaluations/{label}_COMPLETE.json');b=summary['binding'];panel=read_json(FROZEN/panel_file)
        assert summary['passed'] and summary['checkpoint_unchanged'] and summary['tensors_unchanged']
        assert b['code_identity']==code and b['configuration_identity']==config and b['panel_identity']==panel['identity']
        assert b['batch_size']==128 and b['mode']=='true_incremental_bf16_fp32_ce_fp64_nll'
        expected_model=sources['H_model_identity'] if u is None else audits[str(u)]['model_tensor_identity']
        expected_checkpoint=H_SHA if u is None else sha256(raw/f'checkpoints/u{u:05d}.pt')
        assert b['model_tensor_identity']==expected_model and b['checkpoint_sha256']==expected_checkpoint
        assert b['condition']==('H_ON' if u is None else 'H_ALL_OFF')
        actual=collect(raw/'evaluations',label,panel,b,128)
        for k,v in actual.items():assert summary[k]==v,(label,k)
        for r,s in zip(actual['rows'],panel['sequences']):assert r['canonical_group']==s.get('canonical_group',s['id']//64)
        coverage[label]=dict(sequences=len(actual['rows']),targets=actual['count'],checkpoint_sha256=expected_checkpoint,model_tensor_identity=expected_model)
        evaluation_seconds[label]=summary['seconds']
    assert sum(v['targets'] for v in coverage.values())==14942208
    panel=read_json(FROZEN/'PANEL.json')
    for group in range(64):assert all(r['canonical_group']==group for r in panel['sequences'][group*64:(group+1)*64])
    assert len({r['input_start'] for r in panel['sequences']})==4096
    hpath=REPO/'experiment_2d11/results/run/metrics-attempt01.jsonl';historical=read_json(FROZEN/'H_PREFIX.json')
    assert sha256(hpath)==historical['metric_source_sha256']
    hmetrics=[json.loads(l) for l in hpath.read_text().splitlines()][:1000]
    assert len(hmetrics)==1000 and [r['completed_updates'] for r in hmetrics]==list(range(1,1001))
    assert sum(r['training_seconds'] for r in hmetrics)==historical['training_seconds']
    assert sum(r['pass_count'] for r in hmetrics)==2031 and sum(r['pass_count']==3 for r in hmetrics)==31
    for a,b in zip(metrics,hmetrics):assert a['data']==b['data'] and a['lr']==b['lr']
    hflags={}
    for name in ('H','initial'):
        p=torch.load(sources['inputs'][name]['path'],map_location='cpu',mmap=True,weights_only=False)
        hflags[name]=[g['fused'] for g in p['optimizer']['param_groups']]
        assert hflags[name]==[False,False]
    old=archive/'attempts/attempt01';discard=read_json(old/'INVALID_ATTEMPT.json')
    discarded_metrics=[json.loads(l) for l in (old/'raw/metrics.jsonl').read_text().splitlines()]
    assert len(discarded_metrics)==discard['completed_updates']==82
    assert sum(r['training_seconds'] for r in discarded_metrics)==discard['training_seconds']
    assert discard['scientific_updates_retained']==0 and discard['logical_targets_computed']==82*524288
    billing=read_json(archive/'RUNTIME_ACCOUNTING.json')
    stages=dict(valid_training=sum(r['training_seconds'] for r in metrics),valid_evaluations=sum(evaluation_seconds.values()),valid_preflight=read_json(raw/'PREFLIGHT.json')['seconds'],discarded_training=discard['training_seconds'],discarded_monitor=read_json(old/'raw/evaluations/L_monitor_u00000_COMPLETE.json')['seconds'],discarded_preflight=read_json(old/'raw/PREFLIGHT.json')['seconds'])
    stages['remaining_preparation_io_idle_and_shutdown']=billing['cumulative_billed_seconds']-sum(stages.values())
    assert stages['remaining_preparation_io_idle_and_shutdown']>=0
    checkpoint_seconds=sum(read_json(raw/f'checkpoints/u{u:05d}.pt.manifest.json')['seconds'] for u in SCHEDULE)
    originals=[]
    for folder,expected in [('2d12_h10b_ablation','622f34b29ec54816dcf293aef9228939c72a6dd8'),('2d11_from_scratch','ca0103cd3aee5326855f4e72d1d6db93ce073550')]:
        path=REPO.parent/folder
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
        dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no']).decode()
        assert head==expected and not dirty,(folder,head,dirty)
        originals.append(dict(path=str(path),head=head,tracked_changes=False))
    tags={}
    for tag,expected in [('experiment-2d12-h10b-recurrence-ablation-final','622f34b29ec54816dcf293aef9228939c72a6dd8'),('experiment-2d11-from-scratch-h-vs-gpt2-10b-final','ca0103cd3aee5326855f4e72d1d6db93ce073550')]:
        target=subprocess.check_output(['git','rev-parse',tag+'^{}'],cwd=REPO).decode().strip()
        assert target==expected;tags[tag]=target
    for n in ['H_OPTIMIZER_IMPLEMENTATION.json','OPERATIONS_CAPACITY_FIX.json','INDEPENDENT_STOP_RECHECK.json','CAPACITY_FIX_BEFORE.json','OPTIMIZER_INTEGRITY_HOLD.json']:
        shutil.copy2(archive/n,out/n)
    shutil.copy2(old/'INVALID_ATTEMPT.json',out/'INVALID_ATTEMPT.json')
    shutil.copytree(old/'raw',out/'discarded_attempt01',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.transfer'))
    report=dict(passed=True,time=time.time(),coverage=coverage,evaluation_seconds=evaluation_seconds,stage_seconds=stages,checkpoint_serialization_seconds_overlapping_other_stages=checkpoint_seconds,historical_optimizer_fused_flags=hflags,L_optimizer_fused=True,discarded_attempt=discard,original_worktrees=originals,scientific_code_identity=code,volume_size_GB=stop['provider']['networkVolume']['size'],transfer_retry_telemetry='Transport timeouts were retried by the bounded resumable controller; individual timeout counts were not persisted. Billing includes their full elapsed interval.',operational_reload='Controller and watchdog replaced locally with loaded-code fingerprints; remote scientific PID 3287 continued unchanged.')
    report.update(scientific_bundle_sha256=sha256(archive/'bundle.tar.gz'),scientific_bundle_all_frozen_files_verified=True,original_tags=tags)
    atomic_json(out/'CLOSEOUT_AUDIT.json',report)
    return report
