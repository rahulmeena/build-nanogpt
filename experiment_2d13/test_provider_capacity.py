"""Operational regressions: no live mutation or CUDA work."""
import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from . import provider,controller
from .common import atomic_json,read_json

def response(size=200):
    return dict(id=provider.POD,gpuCount=1,networkVolumeId=provider.VOLUME,
                networkVolume=dict(id=provider.VOLUME,size=size),costPerHr=1.59,
                desiredStatus='RUNNING',runtimeStatus='running')

class CapacityTests(unittest.TestCase):
    def test_status_and_stop_at_both_capacities(self):
        for size in (190,200):
            p=response(size);done={**p,'desiredStatus':'EXITED','runtimeStatus':'stopped'}
            with self.subTest(size=size),tempfile.TemporaryDirectory() as tmp:
                with patch.object(provider,'call',return_value=p):
                    self.assertEqual(provider.status()['networkVolume']['size'],size)
                with patch.object(provider,'call',side_effect=[p,{},done]) as call,patch.object(provider.time,'sleep'):
                    self.assertTrue(provider.stop(Path(tmp)/'stop.json','mock only')['passed'])
                    self.assertEqual(call.call_args_list[1].args[0],['pod','stop',provider.POD,'-o','json'])
                    self.assertEqual(call.call_count,3)

    def test_identity_mismatches_rejected_before_stop(self):
        bad=[]
        for key,value in [('id','other-pod'),('networkVolumeId','other-volume'),('gpuCount',4)]:
            p=response();p[key]=value;bad.append(p)
        p=response();p['networkVolume']['id']='other-volume';bad.append(p)
        for p in bad:
            with self.subTest(p=p),patch.object(provider,'call',return_value=p) as call:
                with self.assertRaises(AssertionError):provider.status()
                self.assertEqual(call.call_count,1)
                self.assertEqual(call.call_args.args[0][1],'get')

    def test_attach_skips_staging_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp);root='/workspace/exp2d13/local_scratch_20260906_attempt02'
            atomic_json(a/'binding.json',dict(rate=1.59,billing_start=time.time(),hard_deadline=time.time()+10000))
            atomic_json(a/'GUARD_ARMED.json',{})
            atomic_json(a/'REMOTE_LAUNCH.json',dict(root=root,pid=3287,command=['python3','-m','experiment_2d13.runner','--run',root+'/run']))
            atomic_json(a/'raw/GPU_WORK_FINISHED.json',{})
            remote=Mock();remote.run.side_effect=[b'same runner alive',b'{}'];remote.json.return_value={'stage':'FINISHED'}
            code_identity=provider.loaded_code_identity()
            with patch.object(provider,'loaded_code_identity',return_value=code_identity),patch.object(controller,'connect',return_value=remote),patch.object(controller,'export_all',return_value={}),patch.object(provider,'status',return_value=response()),patch.object(provider,'stop',return_value=dict(verified_at=time.time())):
                controller.launch(a,attach=True)
            remote.transfer.assert_not_called();remote.put_json.assert_not_called()
            self.assertEqual(remote.run.call_count,2)
            active=read_json(a/'CONTROLLER_ACTIVE.json')
            self.assertEqual(active['remote_pid'],3287)
            self.assertEqual(active['mode'],'attach')
            self.assertEqual(active['provider_loaded_code'],provider.loaded_code_identity())

    def test_invalid_attachment_does_not_stop_pod(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)
            atomic_json(a/'binding.json',dict(rate=1.59))
            atomic_json(a/'GUARD_ARMED.json',{})
            atomic_json(a/'REMOTE_LAUNCH.json',dict(root='wrong-run'))
            with patch.object(provider,'status',return_value=response()),patch.object(provider,'stop') as stop:
                with self.assertRaises(AssertionError):controller.launch(a,attach=True)
            stop.assert_not_called()

if __name__=='__main__':unittest.main()
