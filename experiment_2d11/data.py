"""Reconstruct the original B64/world-one stream before physical partitioning."""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
import torch
from .common import ENDPOINT, TARGETS, identity, pass_count


@dataclass
class Cursor:
    shard: int = 0
    position: int = 0
    wraps: int = 0
    logical_batches: int = 0


def advance(cursor, lengths, b=64, t=1024):
    """Exactly the original next_batch post-read shard condition (strict >)."""
    n = b*t
    if cursor.position+n+1 > lengths[cursor.shard]:
        raise ValueError('invalid logical source cursor')
    cursor.position += n
    cursor.logical_batches += 1
    if cursor.position+(n+1) > lengths[cursor.shard]:
        cursor.shard = (cursor.shard+1) % len(lengths)
        cursor.wraps += int(cursor.shard == 0)
        cursor.position = 0


def full_plan(shards, endpoint=ENDPOINT):
    lengths = [x['tokens'] for x in shards]
    cursor = Cursor()
    rows = []
    for u in range(1, endpoint+1):
        before = asdict(cursor)
        spans = []
        for _ in range(8):
            spans.append([cursor.shard, cursor.position])
            advance(cursor, lengths)
        rows.append(dict(update=u, before=before, spans=spans, after=asdict(cursor),
                         pass_count=pass_count(u), logical_targets=TARGETS))
    return rows


def array_hash(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(np.ascontiguousarray(value, dtype='<i8').tobytes())
    return h.hexdigest()


class LogicalLoader:
    def __init__(self, root, shards, cursor=None, b=64, t=1024):
        self.root, self.shards = Path(root), shards
        self.b, self.t = b, t
        self.cursor = Cursor(**cursor) if cursor else Cursor()
        self._array = None
        self._shard = None

    def state_dict(self):
        return asdict(self.cursor)

    def load_state_dict(self, value):
        self.cursor = Cursor(**value)
        if not 0 <= self.cursor.shard < len(self.shards):
            raise ValueError('bad shard')

    def next_batch(self):
        index = self.cursor.shard
        if self._shard != index:
            path = self.root/self.shards[index]['filename']
            self._array = np.load(path, mmap_mode='r', allow_pickle=False)
            assert self._array.ndim == 1 and len(self._array) == self.shards[index]['tokens']
            assert str(self._array.dtype) == self.shards[index]['dtype']
            self._shard = index
        pos, n = self.cursor.position, self.b*self.t
        # Two shifted views of ONE canonical stream; sequence ends never shift independently.
        x = np.array(self._array[pos:pos+n], dtype=np.int64).reshape(self.b,self.t)
        y = np.array(self._array[pos+1:pos+n+1], dtype=np.int64).reshape(self.b,self.t)
        advance(self.cursor, [s['tokens'] for s in self.shards], self.b, self.t)
        return x, y

    def next_global(self, expected=None):
        before = self.state_dict()
        batches = [self.next_batch() for _ in range(8)]
        x, y = [np.concatenate([b[i] for b in batches]) for i in (0,1)]
        if expected is not None:
            assert before == expected['before'] and self.state_dict() == expected['after']
        return x, y, dict(before=before, after=self.state_dict(),
                          actual_token_sha256=array_hash(x,y), target_count=int(y.size))


def physical_batches(x, y, rank, world=4, micro_b=32):
    if len(x) != micro_b*world*4 or rank not in range(world):
        raise ValueError('physical global batch geometry mismatch')
    for micro in range(4):
        start = (micro*world+rank)*micro_b
        yield torch.from_numpy(x[start:start+micro_b]), torch.from_numpy(y[start:start+micro_b])


def partition_examples(rows, workers=4):
    """Length-sorted, deterministic least-loaded assignment; preserve every ID."""
    queues, costs = [[] for _ in range(workers)], [0]*workers
    for row in sorted(rows, key=lambda r: (-r.get('length',1024), r['id'])):
        owner = min(range(workers), key=lambda rank: (costs[rank], rank))
        queues[owner].append(row['id'])
        costs[owner] += row.get('length',1024)
    actual = [x for q in queues for x in q]
    if len(actual) != len(set(actual)) or set(actual) != {x['id'] for x in rows}:
        raise ValueError('partition coverage failure')
    return dict(owners=queues, estimated_lengths=costs, identity=identity(queues))


def collect_historical_spans(value, spans):
    """Recover the sealed formats and explicit inclusive sequence target spans."""
    if isinstance(value, dict):
        for row in value.get('canonical_target_spans_half_open', []):
            spans.add(tuple(map(int,row)))
        for i in value.get('batch_indices_in_evaluation_order', []):
            spans.add((int(i)*65536+1,(int(i)+1)*65536+1))
        selection = value.get('selection')
        if isinstance(selection,dict) and {'start_token_offset','batch_count'} <= selection.keys():
            for i in range(int(selection['batch_count'])):
                start = int(selection['start_token_offset'])+i*65536
                spans.add((start+1,start+65537))
        if 'dataset_target_span' in value:
            a,b = value['dataset_target_span']
            spans.add((int(a),int(b)+1))
        for child in value.values():
            collect_historical_spans(child,spans)
    elif isinstance(value,list):
        for child in value:
            collect_historical_spans(child,spans)
