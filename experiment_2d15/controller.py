"""Mac export/verification and combined stop authority. Never stops on exceptions."""
import argparse,os,shlex,subprocess,time,traceback
from .common import *
from .provider import status,stop_completed,check_binding
from .state import required_labels,complete
from .evaluate import collect

class Controller:
    def __init__(self,archive):
        self.archive=archive;self.b=read_json(archive/'binding.json');check_binding(self.b)
        self.ssh_args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',self.b['ssh_key'],'-p',str(self.b['ssh_port']),f"root@{self.b['ssh_host']}"]
        self.verified=read_json(archive/'VERIFIED_FILES.json') if (archive/'VERIFIED_FILES.json').exists() else {}
    def ssh(self,command):return subprocess.check_output(self.ssh_args+[command],timeout=180).decode()
    def put_json(self,path,value):
        code='from pathlib import Path; import json,os; p=Path('+repr(str(path))+'); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(".writing"); t.write_text('+repr(__import__('json').dumps(value))+'); os.replace(t,p)'
        self.ssh('python -c '+shlex.quote(code))
    def synchronize(self):
        transport=shlex.join(self.ssh_args[:-1])
        subprocess.run(['rsync','-a','--exclude=*.pt','--exclude=*.writing','--exclude=shared_inputs/code/','-e',transport,self.ssh_args[-1]+':'+self.b['remote_root']+'/',str(self.archive)+'/'],check=True,timeout=180)
        code='import pathlib,json; roots='+repr([self.b['remote_root'],self.b['scratch_root']])+ '; print(json.dumps([str(p) for r in roots for p in pathlib.Path(r).glob("**/u*.pt.manifest.json") if p.with_suffix("").with_suffix("").exists()]))'
        # Explicitly enumerate manifests, then transfer only fully published states.
        paths=__import__('json').loads(self.ssh('python -c '+shlex.quote(code)))
        for path in paths:
            if '/preflight' in path:continue
            mf=__import__('json').loads(self.ssh('cat '+shlex.quote(path)));arm=mf['arm'];assert arm in ARMS
            filename=mf['file'];assert filename==f"u{mf['completed_updates']:05d}.pt"
            dest=self.archive/arm/'checkpoints'/filename;dest.parent.mkdir(parents=True,exist_ok=True)
            key=arm+'/'+filename
            if self.verified.get(key)!=mf['sha256'] or not dest.exists():
                remote=path[:-len('.manifest.json')]
                assert remote.startswith(self.b['remote_root']+'/') or remote.startswith(self.b['scratch_root']+'/')
                temp=dest.with_suffix('.downloading')
                subprocess.run(['rsync','-a','--partial','-e',transport,self.ssh_args[-1]+':'+remote,str(temp)],check=True,timeout=600)
                assert sha256(temp)==mf['sha256'];os.replace(temp,dest)
                atomic_json(str(dest)+'.manifest.json',mf)
                # Independently re-open complete state on this Mac, never trusting only remote audit.
                import torch
                from .checkpoints import validate
                torch.set_num_threads(4);audit=validate(torch.load(dest,map_location='cpu',mmap=True,weights_only=False))
                atomic_json(self.archive/arm/'checkpoint_audits'/(filename+'.json'),audit)
                self.verified[key]=mf['sha256'];atomic_json(self.archive/'VERIFIED_FILES.json',self.verified)
            self.put_json(Path(self.b['remote_root'])/arm/'export_acks'/(filename+'.json'),dict(passed=True,sha256=mf['sha256'],local_path=str(dest),verified_at=time.time()))
        # Retain required milestones plus the latest two verified rolling checkpoints locally.
        for arm in ARMS:
            paths=sorted((self.archive/arm/'checkpoints').glob('u*.pt'));rolling=[p for p in paths if int(p.stem[1:]) not in (0,*MILESTONES)]
            for p in rolling[:-2]:
                if self.verified.get(arm+'/'+p.name)==sha256(p):p.unlink()
    def arm_evidence(self,arm):
        root=self.archive/arm;done=root/'GPU_COMPLETE.json'
        if not done.exists():return dict(updates=0)
        v=read_json(done);assert v['passed'] and v['completed_updates']==5000
        labels=required_labels(arm)
        for label in labels:
            summary=read_json(root/'evaluations'/(label+'_COMPLETE.json'));panel=read_json(FROZEN/('MONITOR.json' if '_monitor_' in label else 'PANEL.json'))
            binding=summary['binding'];u=int(label.rsplit('_u',1)[1])
            assert summary['passed'] and summary['checkpoint_unchanged'] and summary['tensors_unchanged']
            assert binding['label']==label and binding['arm']==arm and binding['milestone']==u
            assert binding['panel_identity']==panel['identity'] and binding['configuration_identity']==identity(read_json(FROZEN/'EXECUTION.json'))
            assert binding['code_identity']==identity(read_json(FROZEN/'CODE_IDENTITY.json'))
            assert binding['batch_size']==128 and binding['world_size']==4 and binding['partition']=='B128_group_index_modulo_4'
            assert binding['mode']=='true_incremental_bf16_fp32_ce_fp64_nll' and binding['dataset_sha256']==VAL_SHA
            if label.startswith('H_'):expected_sha,model_id=H_CHECKPOINTS[u]
            else:
                manifest=read_json(root/'checkpoints'/f'u{u:05d}.pt.manifest.json')
                expected_sha=manifest['sha256'];model_id=manifest['audit']['model_tensor_identity']
            assert binding['checkpoint_sha256']==expected_sha and binding['model_tensor_identity']==model_id
            condition='H_ALL_OFF' if (label.startswith('L_') or '_ALL_OFF_' in label) else 'H_ON'
            assert binding['condition']==condition
            check=collect(root/'evaluations',label,panel,binding,128)
            assert check['count']==len(panel['sequences'])*1024
        for u in (0,*MILESTONES):
            f=root/'checkpoints'/f'u{u:05d}.pt';assert self.verified.get(arm+'/'+f.name)==sha256(f)
            mf=read_json(str(f)+'.manifest.json');assert mf['audit']['completed_updates']==u
        evidence=dict(updates=5000,gpu_complete=True,evaluations=labels,exports_verified=True)
        self.put_json(Path(self.b['remote_root'])/'controller'/(arm+'_DURABLY_COMPLETE.json'),dict(passed=True,time=time.time(),evidence=evidence))
        return evidence
    def run(self):
        while True:
            try:
                self.synchronize();state={arm:self.arm_evidence(arm) for arm in ARMS}
                all_ready=all(a.get('exports_verified') for a in state.values())
                state.update(joint_identity_coverage_verified=False,remaining_gpu_work=1)
                if all_ready and (self.archive/'controller/REMOTE_GPU_COMPLETE.json').exists():
                    for u in MILESTONES:
                        summaries={c:read_json(self.archive/('L_nf4' if c in ('L_LOCAL','H_ON','H_ALL_OFF') else 'R_nf4')/'evaluations'/f'{c}_u{u:05d}_COMPLETE.json') for c in CONDITIONS}
                        bindings=[v['binding'] for v in summaries.values()]
                        assert len({b['panel_identity'] for b in bindings})==len({b['code_identity'] for b in bindings})==1
                        assert summaries['H_ON']['binding']['checkpoint_sha256']==summaries['H_ALL_OFF']['binding']['checkpoint_sha256']==H_CHECKPOINTS[u][0]
                        assert summaries['R_ON']['binding']['checkpoint_sha256']==summaries['R_ALL_OFF']['binding']['checkpoint_sha256']
                    state.update(joint_identity_coverage_verified=True,remaining_gpu_work=0)
                atomic_json(self.archive/'COMBINED_STATE.json',state)
                if complete(state):
                    stop_completed(self.b,state,self.archive/'STOP_VERIFICATION.json');return
                atomic_json(self.archive/'CONTROLLER_HEARTBEAT.json',dict(time=time.time(),state=state,pod_stop_requested=False))
            except Exception as e:
                append_json(self.archive/'CONTROLLER_ERRORS.jsonl',dict(time=time.time(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),pod_stop_requested=False))
            time.sleep(20)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);Controller(p.parse_args().archive).run()
