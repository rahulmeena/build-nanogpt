"""One bounded residual input verification on the existing persistent volume."""
import argparse
from pathlib import Path
import time
import subprocess
import shutil
import numpy as np
import torch
from .common import *


def run(args):
    s=time.time();data=Path(args.data);inputs=Path(args.inputs);shards=read_json(FROZEN/'shards.json')
    assert torch.__version__=='2.8.0+cu128' and np.__version__=='2.1.2'
    rows=[]
    for row in [*shards['train'],shards['validation']]:
        p=data/row['filename'];assert p.stat().st_size==row['bytes'];digest=sha256(p)
        assert digest==row['sha256'];rows.append(dict(filename=p.name,sha256=digest,bytes=p.stat().st_size))
    assert sha256(inputs/'initial.pt')==read_json(FROZEN/'initial_identity.json')['sha256']
    assert sha256(inputs/'baseline.pt')==BASELINE_SHA
    assert sha256(inputs/'hellaswag.json')==read_json(FROZEN/'hellaswag_manifest.json')['tokenized_sha256']
    atomic_json(args.output,dict(passed=True,shards=rows,seconds=time.time()-s,
                torch_version=torch.__version__,numpy_version=np.__version__,actual_content_hashes=True,
                nvidia_smi=subprocess.check_output(['nvidia-smi']).decode(),
                topology=subprocess.check_output(['nvidia-smi','topo','-m']).decode(),
                filesystem_free_bytes=shutil.disk_usage(inputs).free,filesystem_free_is_shared_not_quota=True,
                gpu_models=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                cuda_runtime=torch.version.cuda))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','inputs','output']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
