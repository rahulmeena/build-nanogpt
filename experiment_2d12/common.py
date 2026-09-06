from pathlib import Path
from experiment_2d11.common import sha256, identity, canonical, read_json, atomic_json, atomic_bytes, append_json
PACKAGE=Path(__file__).resolve().parent
REPO=PACKAGE.parent
FROZEN=PACKAGE/'frozen'
CONDITIONS={'H_ON':(), 'H_B1_OFF':(0,), 'H_B3_OFF':(2,), 'H_B5_OFF':(4,), 'H_ALL_OFF':(0,2,4)}
CHECKPOINT_SHA='bdcbe4ceede0af61b0cea177d677c31b4b26d3f002a3aabdba1f3bde6810011b'
VAL_SHA='8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f'
SOURCE_COMMIT='ca0103cd3aee5326855f4e72d1d6db93ce073550'
WINDOWS={0:2,2:32,4:64}
