"""Atomic batches with strict identities, unique ownership, and resumability."""
import math
from .common import *

def validate_batch(value,condition,selected,binding):
    assert value['condition']==condition and value['binding']==binding
    assert value['disabled_destinations']==list(CONDITIONS[condition])
    assert len(value['rows'])==len(selected)
    for r,s in zip(value['rows'],selected):
        assert r['id']==s['id'] and r['input_start']==s['input_start']
        assert r['token_sha256']==s['token_sha256'] and r['count']==s['targets']
        assert math.isfinite(r['nll']) and r['nll']>=0 and r['ce']==r['nll']/r['count']
    assert value['cache_audit']['passed']
    return value

def existing(path,condition,selected,binding):
    path=Path(path)
    if not path.exists():return None
    return validate_batch(read_json(path),condition,selected,binding)

def collect(raw,panel):
    raw=Path(raw);config=read_json(raw/'RUN_CONFIG.json');binding=config['binding']
    assert binding['panel_identity']==panel['identity'] and binding['checkpoint_sha256']==CHECKPOINT_SHA
    b=config['batch_size'];data={}
    for c in CONDITIONS:
        files=sorted((raw/c).glob('batch_*.json'));assert len(files)==4096//b
        rows=[]
        for start in range(0,4096,b):
            p=raw/c/f'batch_{start:04d}.json'
            v=validate_batch(read_json(p),c,panel['sequences'][start:start+b],binding)
            rows.extend(v['rows'])
        assert [r['id'] for r in rows]==list(range(4096))
        data[c]=rows
    assert sum(len(v) for v in data.values())==20480
    assert sum(r['count'] for v in data.values() for r in v)==20971520
    return data
