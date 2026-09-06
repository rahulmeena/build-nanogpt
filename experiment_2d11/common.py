"""Frozen scientific constants and durable, non-secret artifact utilities."""
from pathlib import Path
import hashlib
import json
import math
import os
import time

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
FROZEN = PACKAGE / 'frozen'
ENDPOINT = 19072
TARGETS = 524288
CE_SCHEDULE = (0, 250, 500, 750, *range(1000, 19001, 1000), 19072)
HELLA_SCHEDULE = (0, 1000, 2000, 5000, 10000, 15000, 19072)
PRESERVE = (0, 1000, 2000, 5000, 10000, 15000, 19072)
BASELINE_SHA = '924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871'
VAL_SHA = '8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f'
SEALED_COMMIT = '1eda20fc8ee66b4bbfcccae598c8e1354b2d86aa'
SOURCE_COMMIT = '0f7766f9a0c24e9c2854e698862f0a22c19174c7'
INTERPRETATION = ('A poor early H trajectory may reflect the complete H-from-scratch recipe, '
                  'including the preregistered 50/50 initial branch mixture; this experiment '
                  'does not prove that every possible H initialization would fail.')
REFERENCES = {1000: {'ce': 4.13529109954834, 'hella': .2550288787094204},
              2000: {'ce': 3.6628191471099854, 'hella': .26070503883688506}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('wb') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())


def accounting(u):
    if not isinstance(u, int) or not 0 <= u <= ENDPOINT:
        raise ValueError('completed update outside scientific endpoint')
    passes = 2*u + u//32
    return dict(completed_updates=u, logical_targets=TARGETS*u,
                ce_pass_equivalent_updates=passes, ce_pass_targets=TARGETS*passes,
                mean_ce_passes_per_logical_target=passes/u if u else None,
                matched_g_ce_pass_targets=TARGETS*u)


def pass_count(u):
    if not 1 <= u <= ENDPOINT:
        raise ValueError('scientific updates start at one and stop at 19072')
    return 3 if u % 32 == 0 else 2


def learning_rate(index):
    if not 0 <= index < 19073:
        raise ValueError('LR index outside original schedule')
    if index < 715:
        return 6e-4 * (index+1)/715
    maximum=6e-4
    minimum=maximum*.1
    ratio=(index-715)/(19073-715)
    coefficient=.5*(1.+math.cos(math.pi*ratio))
    return minimum+coefficient*(maximum-minimum)


def quality_decision(u, ce, hella=None, g500=None):
    if u not in REFERENCES or not math.isfinite(ce):
        raise ValueError('invalid review')
    gap = ce - REFERENCES[u]['ce']
    if u == 1000:
        stop = gap >= .50
        inputs = dict(g500=gap, severe_deficit_threshold=.50)
    else:
        if g500 is None or hella is None or not all(map(math.isfinite, (g500, hella))):
            raise ValueError('1B rule requires both CE gaps and full HellaSwag')
        advantage = hella - REFERENCES[u]['hella']
        stop = g500 >= .15 and gap >= .15 and gap >= .75*g500 and advantage <= .005
        inputs = dict(g500=g500, g1=gap, a1=advantage)
    return dict(outcome='EARLY_STOPPED_FOR_FUTILITY' if stop else 'CONTINUE',
                stop=stop, inputs=inputs, interpretation=INTERPRETATION,
                historical_comparison_has_paired_ci=False, **accounting(u))


def heartbeat(run, rank, state, progress, stage_deadline, **extra):
    atomic_json(Path(run)/f'heartbeat-rank{rank}.json', dict(
        state=state, progress=progress, last_successful_write=time.time(),
        stage_deadline=stage_deadline, pid=os.getpid(), **extra))


def configure_cuda_determinism():
    """Set before CUDA initialization; preserve frozen BF16/FP32 computation rules."""
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
