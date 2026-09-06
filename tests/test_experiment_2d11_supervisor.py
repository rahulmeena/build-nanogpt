import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_2d11.supervisor import StateMachine
from experiment_2d11.provider import Provider
from experiment_2d11.common import read_json
from experiment_2d11.budget import project_remaining


@pytest.mark.parametrize('terminal',['complete','futility'])
def test_automatic_success_and_futility_without_chat(terminal):
    s=StateMachine(86400)
    for event,now in [('preflight_started',5),('preflight_passed',10),('training_started',11),
                      ('evaluation_started',100),('continue',101),(terminal,200),
                      ('exports_verified',210),('stop_verified',215)]:s.event(event,now)
    assert s.state=='LOCAL_ANALYSIS'
    assert s.outcome==('COMPLETED' if terminal=='complete' else 'EARLY_STOPPED_FOR_FUTILITY')


def test_rank_failure_bounded_export_delay_and_hard_deadline():
    s=StateMachine(10000);s.event('training_started',1);s.event('rank_failed',50)
    assert s.event('export_poll',949)=='FINAL_EXPORT_VERIFICATION'
    assert s.event('export_poll',950)=='STOP_POD'
    assert s.outcome=='INVALID_INCOMPLETE'
    s=StateMachine(1000);s.event('training_started',1);s.event('soft_deadline',500)
    assert s.state=='SAFE_CHECKPOINT'
    assert s.event('export_poll',1000)=='STOP_POD' and s.outcome=='BUDGET_STOPPED'


def test_shutdown_retries_and_prominent_failure(tmp_path):
    class Fake(Provider):
        def __init__(self):self.count=0
        def status(self):
            self.count+=1
            if self.count<3:raise OSError('synthetic API outage')
            return dict(desiredStatus='EXITED',runtimeStatus='stopped')
        def stop(self):pass
    path=tmp_path/'stop.json';result=Fake().stop_verified(path,retry_delay=0,max_attempts=4)
    assert result['passed'] and len(result['attempts'])==2
    class Failed(Fake):
        def status(self):raise OSError('synthetic')
    with pytest.raises(RuntimeError,match='BILLING STOP NOT VERIFIED'):
        Failed().stop_verified(path,retry_delay=0,max_attempts=2)
    assert not read_json(path)['passed'] and read_json(path)['alert']=='POD BILLING STOP NOT VERIFIED'


def test_projection_includes_all_evaluations_exports_and_previous_billing():
    m=dict(two_pass_seconds=2.,three_pass_seconds=3.,monitor_ce_seconds=60.,h_hellaswag_seconds=120.,
           h_fresh_ce_seconds=200.,g_fresh_ce_seconds=150.,g_hellaswag_seconds=90.,
           checkpoint_write_seconds=10.,checkpoint_bytes=1500000000,download_bytes_per_second=100000000,
           g_update_seconds=1.)
    p=project_remaining(m,1800)
    assert p['monitor_ce_seconds']==24*60 and p['hellaswag_seconds']==7*120
    assert p['remaining_allowance_less_two_hour_contingency']==77400
    assert p['fits']
    assert not project_remaining(m,40000)['fits']


@pytest.mark.parametrize('scenario',['success','futility','rank_failure','export_delay','deadline'])
def test_real_subprocess_supervisor_rehearsal(tmp_path,monkeypatch,scenario):
    """Run the production supervisor around synthetic child processes; never use a pod."""
    import time
    from types import SimpleNamespace
    from experiment_2d11 import supervisor
    from experiment_2d11.common import atomic_json
    run=tmp_path/scenario;run.mkdir();binding=tmp_path/'binding.json'
    atomic_json(binding,dict(hard_deadline=time.time()+1000))
    outcome='EARLY_STOPPED_FOR_FUTILITY' if scenario=='futility' else 'COMPLETED'
    def command(args,module,extra=()):
        script='''import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);module=sys.argv[2];scenario=sys.argv[3];outcome=sys.argv[4]
if module=='preflight':
 (p/'PREFLIGHT.json').write_text(json.dumps({'passed':True}))
else:
 if scenario=='deadline':time.sleep(20)
 if scenario=='rank_failure':sys.exit(7)
 (p/'SCIENTIFIC_OUTCOME.json').write_text(json.dumps({'outcome':outcome}))
 if scenario!='export_delay':(p/'FINAL_EXPORTS_VERIFIED.json').write_text(json.dumps({'passed':True}))
'''
        return [sys.executable,'-c',script,str(run),module,scenario,outcome]
    monkeypatch.setattr(supervisor,'stage_command',command)
    if scenario=='deadline':
        # Hard-deadline safety reserve leaves 0.1 s for the child stage.
        atomic_json(binding,dict(hard_deadline=time.time()+300.1))
    args=SimpleNamespace(run=str(run),binding=str(binding),data='',initial='',hella='',baseline='')
    before=time.time();supervisor.supervise(args,poll_seconds=.01,export_seconds=.05,terminate_seconds=.05)
    assert time.time()-before<4
    state=read_json(run/'SUPERVISOR_STATE.json')
    assert state['state']=='STOP_POD' and (run/'STOP_POD_NOW.json').exists()
    assert (run/'GPU_WORK_FINISHED.json').exists()
    if scenario=='futility':assert state['outcome']=='EARLY_STOPPED_FOR_FUTILITY'
    if scenario in ('rank_failure','deadline'):assert state['outcome']=='INVALID_INCOMPLETE'
    if scenario in ('success','export_delay'):assert state['outcome']=='COMPLETED'


def test_independent_watchdog_calls_verified_provider_stop(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import time
    from experiment_2d11 import supervisor
    from experiment_2d11.common import atomic_json
    atomic_json(tmp_path/'binding.json',dict(hard_deadline=time.time()+1000,pod_id='synthetic',billing_start=time.time()))
    (tmp_path/'STOP_POD_NOW.json').touch();called=[]
    class FakeProvider:
        def __init__(self,binding):assert binding['pod_id']=='synthetic'
        def stop_verified(self,path):
            called.append(path);atomic_json(path,dict(passed=True,synthetic=True))
    monkeypatch.setattr(supervisor,'Provider',FakeProvider)
    supervisor.watchdog(SimpleNamespace(run=str(tmp_path),binding=str(tmp_path/'binding.json')))
    assert len(called)==1 and read_json(called[0])['passed']


def test_analysis_flags_after_json_and_paired_bootstrap():
    import numpy as np
    from experiment_2d11.analysis import flag,bootstrap
    flags=json.loads(json.dumps(dict(a=True,b=False,c=None)))
    assert [flag(x) for x in flags.values()]==['PASS','FAIL','UNAVAILABLE']
    for invalid in [1,0,'false','true']:
        with pytest.raises(TypeError):flag(invalid)
    benefit=bootstrap(np.ones(20)*.01,20260918,500)
    harm=bootstrap(np.ones(20)*-.01,20260919,500)
    unresolved=bootstrap(np.zeros(20),20260918,500)
    assert [x['classification'] for x in [benefit,harm,unresolved]]==['benefit','harm','unresolved']
    assert benefit==bootstrap(np.ones(20)*.01,20260918,500)


def test_real_torchrun_parser_keeps_experiment_run_argument():
    from types import SimpleNamespace
    from torch.distributed.run import parse_args
    from experiment_2d11.supervisor import stage_command
    args=SimpleNamespace(run='/run',binding='/binding.json',data='/data',initial='/initial.pt',hella='/hella.json',baseline='/G.pt')
    for module in ['preflight','train']:
        command=stage_command(args,module)
        parsed=parse_args(command[3:])
        assert parsed.module and parsed.training_script=='experiment_2d11.'+module
        assert parsed.training_script_args[:2]==['--run','/run']
        assert '--baseline' in parsed.training_script_args


def test_projection_models_serial_export_queue_without_double_counting_overlap():
    m=dict(two_pass_seconds=2.3,three_pass_seconds=3.5,monitor_ce_seconds=60.,h_hellaswag_seconds=120.,
           h_fresh_ce_seconds=200.,g_fresh_ce_seconds=150.,g_hellaswag_seconds=90.,
           checkpoint_write_seconds=10.,checkpoint_bytes=1650000000,download_bytes_per_second=3600000,
           g_update_seconds=1.)
    p=project_remaining(m,1100)
    assert p['fits'] and p['checkpoint_export_worker_seconds']>p['checkpoint_export_seconds']
    events=p['export_queue_events'];duration=m['checkpoint_bytes']/(m['download_bytes_per_second']*.5)
    for previous,current in zip(events,events[1:]):
        assert current['export_finished_seconds']>=previous['export_finished_seconds']+duration-1e-7
    m['download_bytes_per_second']=100000
    assert not project_remaining(m,1100)['fits']


def test_monitor_does_not_import_stale_remote_controller_heartbeat(tmp_path,monkeypatch):
    import time
    from experiment_2d11 import controller
    from experiment_2d11.common import atomic_json
    before=time.time();downloaded=[]
    class Remote:
        def download(self,source,target,timeout):
            downloaded.append(source)
            atomic_json(target,dict(time=0))
        def put_json(self,path,value):pass
    class Provider:
        def stop_verified(self,path):atomic_json(path,dict(passed=True))
    monkeypatch.setattr(controller,'inventory',lambda *_:[
        dict(path='LOCAL_CONTROLLER_HEARTBEAT.json',size=20,mtime=0),
        dict(path='SUPERVISOR_HEARTBEAT.json',size=20,mtime=0)])
    controller.monitor(Remote(),Provider(),dict(hard_deadline=before-1),tmp_path,'/remote')
    assert downloaded==['/remote/SUPERVISOR_HEARTBEAT.json']
    assert read_json(tmp_path/'LOCAL_CONTROLLER_HEARTBEAT.json')['time']>=before


@pytest.mark.parametrize('stale',[True,False])
def test_local_guard_records_timeout_and_respects_live_heartbeat(tmp_path,monkeypatch,stale):
    import time
    from experiment_2d11 import controller
    from experiment_2d11.common import atomic_json
    now=time.time();called=[]
    atomic_json(tmp_path/'binding.json',dict(hard_deadline=now+1000,billing_start=now-1000))
    atomic_json(tmp_path/'LOCAL_CONTROLLER_HEARTBEAT.json',dict(time=now-700 if stale else now))
    class Provider:
        def __init__(self,binding):pass
        def stop_verified(self,path):called.append(path)
    monkeypatch.setattr(controller,'Provider',Provider)
    monkeypatch.setattr(controller.time,'sleep',lambda _:atomic_json(tmp_path/'STOP_VERIFICATION.json',dict(passed=True)))
    controller.local_guard(tmp_path/'binding.json',tmp_path)
    assert bool(called)==stale
    if stale:assert read_json(tmp_path/'INDEPENDENT_LOCAL_STOP_REASON.json')['reason']=='stalled_local_controller'


def test_monitor_survives_simultaneous_ssh_and_provider_network_outage(tmp_path,monkeypatch):
    import time
    from experiment_2d11 import controller
    from experiment_2d11.common import atomic_json
    before=time.time();calls=[]
    class Remote:
        def put_json(self,*args):pass
    class FakeProvider:
        def status(self):raise OSError('temporary provider network failure')
        def stopped(self,p):return False
        def stop_verified(self,path):calls.append('stop');atomic_json(path,dict(passed=True))
    def inventory(*args):
        calls.append('poll')
        if calls.count('poll')==1:raise OSError('temporary SSH network failure')
        # End the synthetic monitor only after demonstrating another poll.
        binding['hard_deadline']=before-1
        return []
    binding=dict(hard_deadline=before+1000)
    monkeypatch.setattr(controller,'inventory',inventory)
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    controller.monitor(Remote(),FakeProvider(),binding,tmp_path,'/remote')
    assert calls==['poll','poll','stop']
    events=[json.loads(line) for line in (tmp_path/'controller_events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['poll_error','status_probe_error']


def test_resume_reuses_preflight_and_verifies_checkpoint(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from experiment_2d11 import supervisor
    from experiment_2d11.common import atomic_json,sha256
    run=tmp_path;checkpoint=run/'state.pt';checkpoint.write_bytes(b'verified state')
    atomic_json(str(checkpoint)+'.manifest.json',dict(audit=dict(passed=True),sha256=sha256(checkpoint)))
    atomic_json(run/'PREFLIGHT.json',dict(passed=True))
    atomic_json(run/'binding.json',dict(hard_deadline=time.time()+1000))
    calls=[]
    def command(args,module,extra=()):
        calls.append((module,extra))
        return [sys.executable,'-c',"import json,sys;from pathlib import Path;p=Path(sys.argv[1]);(p/'SCIENTIFIC_OUTCOME.json').write_text(json.dumps({'outcome':'COMPLETED'}));(p/'FINAL_EXPORTS_VERIFIED.json').write_text('{}')",str(run)]
    monkeypatch.setattr(supervisor,'stage_command',command)
    args=SimpleNamespace(run=str(run),binding=str(run/'binding.json'),resume=str(checkpoint),attempt='attempt02')
    supervisor.supervise(args,poll_seconds=.01)
    assert calls==[('train',['--attempt','attempt02','--resume',str(checkpoint)])]
    assert read_json(run/'PREFLIGHT_REUSED.json')['checkpoint_sha256']==sha256(checkpoint)
    assert read_json(run/'SUPERVISOR_STATE.json')['outcome']=='COMPLETED'


def test_checkpoint_export_retries_transport_but_rejects_corruption(tmp_path,monkeypatch):
    from experiment_2d11 import controller
    from experiment_2d11.common import sha256
    source=tmp_path/'source';source.write_bytes(b'checkpoint');calls=[]
    class Remote:
        def download(self,remote,target,timeout):
            calls.append('download')
            if len(calls)==1:raise OSError('temporary network')
            Path(target).write_bytes(source.read_bytes())
        def put_json(self,*args):calls.append('ack')
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    manifest=dict(sha256=sha256(source),bytes=source.stat().st_size)
    assert controller.export_checkpoint(Remote(),'/state',tmp_path/'result.pt',manifest,'/run')['verified']
    assert calls==['download','download','ack']
    manifest['sha256']='wrong'
    with pytest.raises(ValueError,match='SHA mismatch'):
        controller.export_checkpoint(Remote(),'/state',tmp_path/'bad.pt',manifest,'/run')


def test_recovery_preserves_raw_tail_and_removes_stale_stop_markers(tmp_path):
    from experiment_2d11.recover_network import preserve,HISTORY
    from experiment_2d11.common import atomic_json
    atomic_json(tmp_path/'STOP_VERIFICATION.json',dict(passed=True))
    rows=[dict(completed_updates=u,training_seconds=2.5) for u in [5999,6000,6001,6002]]
    raw=tmp_path/'metrics-attempt01.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    preserve(tmp_path)
    history=tmp_path/'recovery_history'/HISTORY
    assert not (tmp_path/'STOP_VERIFICATION.json').exists()
    assert read_json(history/'STOP_VERIFICATION.json')['passed']
    assert len((history/raw.name).read_text().splitlines())==4
    assert [json.loads(x)['completed_updates'] for x in raw.read_text().splitlines()]==[5999,6000]
    assert read_json(history/'ROLLED_BACK_TAIL.json')['discarded_training_seconds']==5
