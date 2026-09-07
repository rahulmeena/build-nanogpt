import pathlib,json,hashlib,time
root=pathlib.Path('/workspace/exp2d15/lr_nf4_20260907')
assert [v.split(b'=',1)[1].decode() for v in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0') if v.startswith(b'RUNPOD_POD_ID=')]==['l55wgmjewejiv2']
plan={'L_nf4': ['L_nf4_monitor_u00000', 'L_nf4_monitor_u00250', 'L_nf4_monitor_u00500', 'L_nf4_monitor_u00750', 'L_nf4_monitor_u01000', 'L_nf4_monitor_u02000', 'L_nf4_monitor_u03000', 'L_nf4_monitor_u04000', 'L_nf4_monitor_u05000', 'L_LOCAL_u01000', 'H_ON_u01000', 'H_ALL_OFF_u01000', 'L_LOCAL_u02000', 'H_ON_u02000', 'H_ALL_OFF_u02000', 'L_LOCAL_u05000', 'H_ON_u05000', 'H_ALL_OFF_u05000'], 'R_nf4': ['R_nf4_monitor_u00000', 'R_nf4_monitor_u00250', 'R_nf4_monitor_u00500', 'R_nf4_monitor_u00750', 'R_nf4_monitor_u01000', 'R_nf4_monitor_u02000', 'R_nf4_monitor_u03000', 'R_nf4_monitor_u04000', 'R_nf4_monitor_u05000', 'R_ON_u01000', 'R_ALL_OFF_u01000', 'R_ON_u02000', 'R_ALL_OFF_u02000', 'R_ON_u05000', 'R_ALL_OFF_u05000']}
files={}
def record(p):
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 files[str(p.relative_to(root))]=dict(sha256=h.hexdigest(),bytes=before.st_size)
for arm in plan:
 for u in [0,1000,2000,5000]:
  p=root/arm/'checkpoints'/f'u{u:05d}.pt';manifest=json.loads(pathlib.Path(str(p)+'.manifest.json').read_text());record(p);assert files[str(p.relative_to(root))]['sha256']==manifest['sha256']
pending=[(arm,label) for arm,labels in plan.items() for label in labels]
while pending:
 for arm,label in pending[:]:
  d=root/arm/'evaluations';p=d/(label+'_COMPLETE.json')
  if not p.exists():continue
  value=json.loads(p.read_text());count=1280 if '_monitor_' in label else 4096
  assert value['passed'] and value['checkpoint_unchanged'] and value['tensors_unchanged'] and len(value['rows'])==count
  record(p)
  for i in range(0,count,128):record(d/label/f'batch_{i:04d}.json')
  pending.remove((arm,label))
 if pending:
  failures=[str(p) for arm in ['L_nf4','R_nf4','controller'] for p in (root/arm).glob('FAILURE*')]
  assert not failures,failures
  time.sleep(2)
assert len(files)==701
print(json.dumps(dict(passed=True,time=time.time(),pod_id='l55wgmjewejiv2',required_checkpoint_files=8,evaluation_complete_files=33,raw_batch_files=660,files=files)))
