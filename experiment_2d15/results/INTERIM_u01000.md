# Experiment 2D15: complete five-condition comparison at update 1000

Each trained trajectory has reached 524,288,000 logical targets. Historical H is reused as the matched-age comparator. The prespecified main endpoint is update 5000; no recipe adjustment or early stop follows this result.

All five conditions use the same fresh 4,096 validation sequences, true incremental BF16 forward, FP32 token CE and FP64 sums, with fixed B128 groups on four GPUs.

| Condition | CE | Perplexity |
|---|---:|---:|
| H_ON | 4.004067100 | 54.820658 |
| L_LOCAL | 3.999703392 | 54.581958 |
| R_ON | 4.010816546 | 55.191919 |
| H_ALL_OFF | 4.006596882 | 54.959518 |
| R_ALL_OFF | 4.030191000 | 56.271658 |

Each bootstrap uses 50,000 paired resamples, preserving pairing across conditions and available ages. Sequence seed: 20261001. Canonical 64-group sensitivity seed: 20261002. The family remains all 18 contrasts, including at interim milestones; adjusted intervals are 99.7222222222%. The practical reference is delta_CE = 0.0001.

| Contrast | Mean | Raw 95% interval | Adjusted interval | Separate adjusted-interval flags |
|---|---:|---|---|---|
| A = H_ON − R_ON | -0.006749447 | [-0.007513870, -0.005984938] | [-0.007915989, -0.005581954] | negative; below −delta |
| B = L_LOCAL − R_ON | -0.011113154 | [-0.011955936, -0.010279365] | [-0.012401199, -0.009837364] | negative; below −delta |
| C = L_LOCAL − H_ON | -0.004363708 | [-0.005109714, -0.003625873] | [-0.005490383, -0.003229236] | negative; below −delta |
| G_H = H_ALL_OFF − H_ON | +0.002529782 | [+0.002433311, +0.002625852] | [+0.002383192, +0.002676729] | positive; above +delta |
| G_R = R_ALL_OFF − R_ON | +0.019374453 | [+0.018968531, +0.019779005] | [+0.018752853, +0.019987792] | positive; above +delta |
| I = G_R − G_H | +0.016844671 | [+0.016497009, +0.017188940] | [+0.016315062, +0.017372251] | positive; above +delta |

| Group sensitivity contrast | Raw 95% interval | Adjusted interval | Separate flags |
|---|---|---|---|
| A | [-0.007609511, -0.005910688] | [-0.008074947, -0.005465837] | negative; below −delta |
| B | [-0.012052792, -0.010174903] | [-0.012545668, -0.009654178] | negative; below −delta |
| C | [-0.005179330, -0.003520095] | [-0.005613294, -0.003077718] | negative; below −delta |
| G_H | [+0.002384754, +0.002680880] | [+0.002313907, +0.002768368] | positive; above +delta |
| G_R | [+0.018775859, +0.019986044] | [+0.018458853, +0.020333255] | positive; above +delta |
| I | [+0.016354506, +0.017343364] | [+0.016095586, +0.017607676] | positive; above +delta |

Positive A or B favors R quality; positive C favors H over L. Positive G values favor recurrence ON, and positive I means greater R OFF sensitivity. A flag of none does not establish practical equivalence.

CE1-free joint goal: **not met** under the primary sequence analysis; **not met** in the group sensitivity. This requires the adjusted lower bounds A > −0.0001, G_R > +0.0001 and I > +0.0001 together.

These checkpoints belong to one training trajectory per model. Intentional architecture, objective and pass-count differences remain. Larger OFF sensitivity alone does not establish better quality or isolated recurrent-content usefulness. No broad matched 10B/HellaSwag conclusion is made.
