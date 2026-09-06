# Experiment 2D15: partial H/L comparison at update 1000

L_nf4 has completed 1,000 updates (524,288,000 logical targets). This is an incomplete three-model comparison: R_nf4 has not yet reached this checkpoint. The main endpoint remains update 5000; training continues unchanged.

The same fresh 4,096 sequences were evaluated for all three available conditions using true incremental BF16, FP32 token CE, FP64 sums, B128 groups on four GPUs. Each raw batch was independently recollected on the Mac and checked against the frozen panel, configuration, code and checkpoint/model identities.

| Condition | CE | Perplexity |
|---|---:|---:|
| L_LOCAL | 3.999703392 | 54.581958 |
| H_ON | 4.004067100 | 54.820658 |
| H_ALL_OFF | 4.006596882 | 54.959518 |

Both bootstraps use 50,000 paired resamples. The sequence seed is 20261001 and the 64-group sensitivity seed is 20261002. Adjusted intervals retain the full prespecified 18-contrast family (99.7222222222%); later checkpoints and R contrasts do not shrink this family. Raw 95% intervals and separate classification flags are retained in the JSON.

| Contrast | Mean CE difference | Sequence adjusted interval | Group adjusted interval |
|---|---:|---|---|
| C = L_LOCAL − H_ON | -0.004363708 | [-0.005490383, -0.003229236] | [-0.005613294, -0.003077718] |
| G_H = H_ALL_OFF − H_ON | +0.002529782 | [+0.002383192, +0.002676729] | [+0.002313907, +0.002768368] |

At this checkpoint L has lower CE than H_ON; both adjusted C intervals lie below −0.0001. H performs better with recurrence enabled than with its recurrence bypassed; both adjusted G_H intervals lie above +0.0001. These results do not establish the CE1-free R goal, which requires R_ON/R_ALL_OFF and the complete paired comparison. They describe one checkpoint from one trajectory per trained model, with intentional architecture and compute differences.

All H inputs and completed 2D13/2D14 artifacts are preserved. No recipe adjustment or early stop follows these results.
