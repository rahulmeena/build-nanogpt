# Experiment 2D15: complete five-condition comparison at update 2000

Each trained trajectory has reached 1,048,576,000 logical targets. Historical H is reused as the matched-age comparator. The prespecified main endpoint is update 5000; no recipe adjustment or early stop follows this result.

All five conditions use the same fresh 4,096 validation sequences, true incremental BF16 forward, FP32 token CE and FP64 sums, with fixed B128 groups on four GPUs.

| Condition | CE | Perplexity |
|---|---:|---:|
| H_ON | 3.568172181 | 35.451735 |
| L_LOCAL | 3.571814430 | 35.581094 |
| R_ON | 3.578177150 | 35.808208 |
| H_ALL_OFF | 3.568984059 | 35.480529 |
| R_ALL_OFF | 3.592040136 | 36.308074 |

Each bootstrap uses 50,000 paired resamples, preserving pairing across conditions and available ages. Sequence seed: 20261001. Canonical 64-group sensitivity seed: 20261002. The family remains all 18 contrasts, including at interim milestones; adjusted intervals are 99.7222222222%. The practical reference is delta_CE = 0.0001.

| Contrast | Mean | Raw 95% interval | Adjusted interval | Separate adjusted-interval flags |
|---|---:|---|---|---|
| A = H_ON − R_ON | -0.010004968 | [-0.010729056, -0.009280661] | [-0.011089628, -0.008917383] | negative; below −delta |
| B = L_LOCAL − R_ON | -0.006362720 | [-0.007119915, -0.005601189] | [-0.007525846, -0.005197573] | negative; below −delta |
| C = L_LOCAL − H_ON | +0.003642248 | [+0.002928001, +0.004370432] | [+0.002541312, +0.004756141] | positive; above +delta |
| G_H = H_ALL_OFF − H_ON | +0.000811877 | [+0.000755279, +0.000868270] | [+0.000726190, +0.000896804] | positive; above +delta |
| G_R = R_ALL_OFF − R_ON | +0.013862986 | [+0.013582483, +0.014142200] | [+0.013446556, +0.014301962] | positive; above +delta |
| I = G_R − G_H | +0.013051109 | [+0.012787644, +0.013313163] | [+0.012660485, +0.013458567] | positive; above +delta |

| Group sensitivity contrast | Raw 95% interval | Adjusted interval | Separate flags |
|---|---|---|---|
| A | [-0.010821497, -0.009189181] | [-0.011239896, -0.008767050] | negative; below −delta |
| B | [-0.007131098, -0.005614223] | [-0.007546224, -0.005234998] | negative; below −delta |
| C | [+0.002977210, +0.004328071] | [+0.002659381, +0.004700938] | positive; above +delta |
| G_H | [+0.000742139, +0.000883696] | [+0.000708175, +0.000921636] | positive; above +delta |
| G_R | [+0.013510435, +0.014203360] | [+0.013317688, +0.014383511] | positive; above +delta |
| I | [+0.012723032, +0.013367121] | [+0.012550858, +0.013531164] | positive; above +delta |

Positive A or B favors R quality; positive C favors H over L. Positive G values favor recurrence ON, and positive I means greater R OFF sensitivity. A flag of none does not establish practical equivalence.

CE1-free joint goal: **not met** under the primary sequence analysis; **not met** in the group sensitivity. This requires the adjusted lower bounds A > −0.0001, G_R > +0.0001 and I > +0.0001 together.

These checkpoints belong to one training trajectory per model. Intentional architecture, objective and pass-count differences remain. Larger OFF sensitivity alone does not establish better quality or isolated recurrent-content usefulness. No broad matched 10B/HellaSwag conclusion is made.
