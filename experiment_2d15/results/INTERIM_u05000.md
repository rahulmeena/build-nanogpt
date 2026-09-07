# Experiment 2D15: complete five-condition comparison at update 5000

Each trained trajectory has reached 2,621,440,000 logical targets. Historical H is reused as the matched-age comparator. The prespecified main endpoint is update 5000; no recipe adjustment or early stop follows this result.

All five conditions use the same fresh 4,096 validation sequences, true incremental BF16 forward, FP32 token CE and FP64 sums, with fixed B128 groups on four GPUs.

| Condition | CE | Perplexity |
|---|---:|---:|
| H_ON | 3.292104878 | 26.899424 |
| L_LOCAL | 3.291562382 | 26.884835 |
| R_ON | 3.295090746 | 26.979862 |
| H_ALL_OFF | 3.292779387 | 26.917574 |
| R_ALL_OFF | 3.337079561 | 28.136835 |

Each bootstrap uses 50,000 paired resamples, preserving pairing across conditions and available ages. Sequence seed: 20261001. Canonical 64-group sensitivity seed: 20261002. The family remains all 18 contrasts, including at interim milestones; adjusted intervals are 99.7222222222%. The practical reference is delta_CE = 0.0001.

| Contrast | Mean | Raw 95% interval | Adjusted interval | Separate adjusted-interval flags |
|---|---:|---|---|---|
| A = H_ON − R_ON | -0.002985869 | [-0.003745486, -0.002227479] | [-0.004127674, -0.001837555] | negative; below −delta |
| B = L_LOCAL − R_ON | -0.003528364 | [-0.004313538, -0.002748874] | [-0.004728584, -0.002301612] | negative; below −delta |
| C = L_LOCAL − H_ON | -0.000542495 | [-0.001291982, +0.000206669] | [-0.001688770, +0.000598455] | none |
| G_H = H_ALL_OFF − H_ON | +0.000674509 | [+0.000628103, +0.000720392] | [+0.000601825, +0.000746297] | positive; above +delta |
| G_R = R_ALL_OFF − R_ON | +0.041988815 | [+0.041561398, +0.042416443] | [+0.041343562, +0.042647603] | positive; above +delta |
| I = G_R − G_H | +0.041314306 | [+0.040894408, +0.041736445] | [+0.040675505, +0.041962493] | positive; above +delta |

| Group sensitivity contrast | Raw 95% interval | Adjusted interval | Separate flags |
|---|---|---|---|
| A | [-0.003731960, -0.002265418] | [-0.004129372, -0.001903088] | negative; below −delta |
| B | [-0.004419119, -0.002639265] | [-0.004886781, -0.002191819] | negative; below −delta |
| C | [-0.001355898, +0.000293495] | [-0.001761595, +0.000729752] | none |
| G_H | [+0.000623974, +0.000724901] | [+0.000594869, +0.000753397] | positive; above +delta |
| G_R | [+0.041414361, +0.042557107] | [+0.041110529, +0.042848914] | positive; above +delta |
| I | [+0.040745844, +0.041878434] | [+0.040447121, +0.042171402] | positive; above +delta |

Positive A or B favors R quality; positive C favors H over L. Positive G values favor recurrence ON, and positive I means greater R OFF sensitivity. A flag of none does not establish practical equivalence.

CE1-free joint goal: **not met** under the primary sequence analysis; **not met** in the group sensitivity. This requires the adjusted lower bounds A > −0.0001, G_R > +0.0001 and I > +0.0001 together.

These checkpoints belong to one training trajectory per model. Intentional architecture, objective and pass-count differences remain. Larger OFF sensitivity alone does not establish better quality or isolated recurrent-content usefulness. No broad matched 10B/HellaSwag conclusion is made.
