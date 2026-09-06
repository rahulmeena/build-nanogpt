"""Conservative shared-volume reservation; never modifies another namespace."""
import subprocess,time
from .common import *

def size(path):return path.stat().st_size if path.exists() else 0

def capacity(root,l_run):
    root=Path(root);l_run=Path(l_run)
    used=int(subprocess.check_output(['du','-sb','/workspace'],timeout=120).split()[0])
    own=int(subprocess.check_output(['du','-sb',str(root)],timeout=120).split()[0]) if root.exists() else 0
    # Published and in-progress bytes are already included in used; reserve only the remaining bytes.
    l_remaining=150_000_000
    evidence=[]
    for u in (0,250,500,750,1000):
        path=l_run/f'checkpoints/u{u:05d}.pt';writing=path.with_suffix('.writing')
        estimate=654_000_000 if u==0 else 1_652_000_000
        published=path.exists() and Path(str(path)+'.manifest.json').exists()
        already=max(size(path),size(writing))
        missing=0 if published else max(0,estimate-already)
        l_remaining+=missing;evidence.append(dict(update=u,published=published,already_counted_bytes=already,remaining_bytes=missing))
    own_remaining=max(0,8_500_000_000-own);safety=2_000_000_000
    projected=used+l_remaining+own_remaining+safety
    return dict(time=time.time(),passed=projected<=200_000_000_000,used_bytes=used,own_existing_bytes=own,
        quota_bytes=200_000_000_000,own_peak_reserve=8_500_000_000,own_remaining_bytes=own_remaining,
        l_remaining_reserve_bytes=l_remaining,l_checkpoint_evidence=evidence,safety_bytes=safety,
        projected_bytes=projected,headroom_bytes=200_000_000_000-projected,other_files_modified=False,
        df_is_cluster_capacity_not_volume_quota=True)
