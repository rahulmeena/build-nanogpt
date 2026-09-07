# Experiment 2D15: partial H/L comparison at update 5000

L_nf4 has completed 5,000 updates (2,621,440,000 logical targets). R_ON and R_ALL_OFF are pending. The main endpoint remains update 5000; this is an incomplete three-model comparison.

All three conditions use the same fresh 4,096 sequences, true incremental BF16 forward, FP32 token CE, FP64 sums, and B128 groups on four GPUs. Raw batches were independently recollected on the Mac and checked against frozen panel, configuration, code, checkpoint and model identities.

| Condition | CE | Perplexity |
|---|---:|---:|
| L_LOCAL | 3.291562382 | 26.884835 |
| H_ON | 3.292104878 | 26.899424 |
| H_ALL_OFF | 3.292779387 | 26.917574 |

The sequence bootstrap uses 50,000 paired resamples with seed 20261001; the 64-group sensitivity uses 50,000 with seed 20261002. Pairing is preserved across available ages. Both retain the full 18-contrast family and adjusted 99.7222222222% intervals. Raw 95% intervals and separate classification flags are in the JSON.

| Contrast | Mean CE difference | Sequence adjusted interval | Group adjusted interval |
|---|---:|---|---|
| C = L_LOCAL − H_ON | -0.000542495 | [-0.001688770, +0.000598455] | [-0.001761595, +0.000729752] |
| G_H = H_ALL_OFF − H_ON | +0.000674509 | [+0.000601825, +0.000746297] | [+0.000594869, +0.000753397] |

Negative C favors L quality; positive G_H favors H recurrence ON. R quality and the CE1-free joint goal remain unclassified until the full comparison is available. These are checkpoints from one trajectory per trained model, with intentional architecture and compute differences. No recipe adjustment or early stop follows these results.
