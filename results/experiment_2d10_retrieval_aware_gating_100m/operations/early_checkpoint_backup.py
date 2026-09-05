"""Overlap verified checkpoint exports with final GPU evaluation."""
import importlib.util,json,pathlib,subprocess,time
spec=importlib.util.spec_from_file_location('control',pathlib.Path(__file__).with_name('control.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
for arm in ('H','T'):
 name='scientific_cumulative_001300758528.pt'
 source=c.REMOTE+'/checkpoints/'+arm+'/'+name
 while not c.remote('test -f '+source+'.verification.json && echo yes || true').strip():
  status=c.remote('cat '+c.REMOTE+'/workflow.status 2>/dev/null || true').strip()
  if status:raise RuntimeError('workflow finished before checkpoint was available: '+arm)
  time.sleep(20)
 dest=c.ARCHIVE/arm;dest.mkdir(exist_ok=True)
 for suffix in ('.verification.json','.sha256'):
  subprocess.run(c.SCP+['root@154.54.102.30:'+source+suffix,str(dest/(name+suffix))],check=True,timeout=120)
 v=json.loads((dest/(name+'.verification.json')).read_text());assert v['strict_reopen']['passed']
 target=dest/name
 if not target.exists():
  tmp=dest/(name+'.early-transfer')
  subprocess.run(c.SCP+['root@154.54.102.30:'+source,str(tmp)],check=True,timeout=900)
  assert c.sha(tmp)==v['sha256'];tmp.replace(target)
 assert c.sha(target)==c.remote('sha256sum '+source).split()[0]==v['sha256']
 print('EARLY_CHECKPOINT_BACKUP_VERIFIED '+arm+' '+v['sha256'],flush=True)
