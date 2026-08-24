# Experiment 2D0B — B11 Micro-Window Sensitivity Sweep

## Outcome

The six requested windows were evaluated with the exact frozen 2D0A path. This was evaluation-only: no optimizer, backward pass, update, recurrence, or completion module was used.

## New micro-window points

| Window | Validation loss | Damage vs W1024 | KV fraction | B11 cosine | B11 RMS | B12 cosine | B12 RMS | Logit argmax agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.1129368308 | +0.0378930554 | 0.062500 | 0.9875277161 | 0.3398053877 | 0.9881379763 | 0.3751777165 | 0.9047485352 |
| 32 | 3.1445640774 | +0.0695203021 | 0.031250 | 0.9791996060 | 0.4583077571 | 0.9802385236 | 0.5061040693 | 0.8687507629 |
| 16 | 3.1845984731 | +0.1095546978 | 0.015625 | 0.9687206690 | 0.5757519928 | 0.9701577012 | 0.6368427531 | 0.8345626831 |
| 8 | 3.2240422755 | +0.1489985002 | 0.007812 | 0.9570631046 | 0.6835281800 | 0.9587351621 | 0.7578832007 | 0.8086990356 |
| 4 | 3.2579787844 | +0.1829350090 | 0.003906 | 0.9447999064 | 0.7803748874 | 0.9464983741 | 0.8674473738 | 0.7904541016 |
| 2 | 3.2940255915 | +0.2189818162 | 0.001953 | 0.9312607841 | 0.8742901934 | 0.9326525330 | 0.9739860555 | 0.7761756897 |

## Complete measured curve

| Window | Validation loss | Damage vs W1024 | B11 KV fraction |
|---:|---:|---:|---:|
| 1024 | 3.0750437753 | +0.0000000000 | 1.000000 |
| 896 | 3.0753463744 | +0.0003025990 | 0.875000 |
| 768 | 3.0757388638 | +0.0006950885 | 0.750000 |
| 512 | 3.0773528399 | +0.0023090646 | 0.500000 |
| 384 | 3.0792040860 | +0.0041603106 | 0.375000 |
| 256 | 3.0827297657 | +0.0076859903 | 0.250000 |
| 128 | 3.0934758906 | +0.0184321153 | 0.125000 |
| 64 | 3.1129368308 | +0.0378930554 | 0.062500 |
| 32 | 3.1445640774 | +0.0695203021 | 0.031250 |
| 16 | 3.1845984731 | +0.1095546978 | 0.015625 |
| 8 | 3.2240422755 | +0.1489985002 | 0.007812 |
| 4 | 3.2579787844 | +0.1829350090 | 0.003906 |
| 2 | 3.2940255915 | +0.2189818162 | 0.001953 |
| 1 | 3.3540805459 | +0.2790367706 | 0.000977 |

## Adjacent micro-window increments

| Transition | Additional damage | Damage at shorter window |
|---|---:|---:|
| W128→W64 | +0.0194609401 | +0.0378930554 |
| W64→W32 | +0.0316272466 | +0.0695203021 |
| W32→W16 | +0.0400343957 | +0.1095546978 |
| W16→W8 | +0.0394438024 | +0.1489985002 |
| W8→W4 | +0.0339365089 | +0.1829350090 |
| W4→W2 | +0.0360468071 | +0.2189818162 |
| W2→W1 | +0.0600549544 | +0.2790367706 |

## Quality thresholds

| Allowed damage | Smallest measured window | KV fraction |
|---:|---:|---:|
| 0.010 | 256 | 0.250000 |
| 0.020 | 128 | 0.125000 |
| 0.050 | 64 | 0.062500 |
| 0.100 | 32 | 0.031250 |
| 0.200 | 4 | 0.003906 |
| 0.250 | 2 | 0.001953 |

## Interpretation

The complete 1024→1 curve is monotonic at every measured point.
These results isolate B11's direct history while B1-B10 and B12 remain full-context. They do not establish the optimal B11 window when other layers are shortened jointly.
The B12 RMS comparison shows whether the final block absorbs or preserves the B11 state disturbance; the exact per-window values are in the table and machine-readable drift artifacts.

## Integrity audit

- PASS — source checkpoint exact
- PASS — Standard GPT-2; Full AttnRes absent
- PASS — canonical validation exact
- PASS — W1024/W512 regression sentinels
- PASS — windows exactly 64/32/16/8/4/2
- PASS — B1-B10 and B12 remain W1024
- PASS — only B11 modified
- PASS — same precision and denominator
- PASS — losses and activations finite
- PASS — incoming h10 identical
- PASS — model tensors unchanged
- PASS — zero optimizer/backward/update/training
- PASS — no recurrence or completion
- PASS — independent processes; no DDP/NCCL
- PASS — all required artifacts present

## Performance

| Window/GPU | Wall s | Targets/s | Peak allocated MB | Peak reserved MB |
|---|---:|---:|---:|---:|
| W64/GPU0 | 47.391 | 27657.3 | 33869.0 | 41750.0 |
| W32/GPU1 | 45.651 | 28712.0 | 33869.0 | 41750.0 |
| W16/GPU2 | 48.723 | 26901.3 | 33869.0 | 41750.0 |
| W8/GPU3 | 48.828 | 26843.8 | 33869.0 | 41750.0 |
| W4/GPU0 | 37.579 | 34878.6 | 33869.0 | 41750.0 |
| W2/GPU1 | 37.172 | 35261.4 | 33869.0 | 41750.0 |

Wave 1 [64, 32, 16, 8] elapsed wall time: 48.929 seconds.

Wave 2 [4, 2] elapsed wall time: 37.626 seconds.

# EXPERIMENT 2D0B COMPLETE
