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
