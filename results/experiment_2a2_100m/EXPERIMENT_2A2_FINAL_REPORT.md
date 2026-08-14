# Experiment 2A2 — Teacher-Feedback Memory Reader Scaling, 25M to 100M

## Outcome

**MEMORY SIGNAL ACCELERATING**

The same frozen-base teacher-feedback reader continued from 25,165,824 to
100,139,008 Experiment-2 student tokens. Sequence-specific recovery increased
from 0.041252 at 25M, to 0.081010 at 50M, to 0.141575 at 100M. Real aligned
feedback beat shuffled feedback on every one of the 20 canonical validation
batches at both new milestones.

The frozen continuation rule answers **YES**: there is enough evidence to
justify considering this same teacher-assisted reader for a separately
approved 100M-to-250M continuation. No such continuation was launched.

## Resume and integrity

```text
starting checkpoint:
/workspace/build-nanogpt-exp2a0/runs/experiment_2a1_25m/checkpoints/checkpoint_updates_000048.pt

starting SHA256:
d821b48a796b12bb489f5bc9bc1791c475c09a50de7d5b47c4a36cf766643ec2

starting completed updates: 48
starting student tokens: 25,165,824
last consumed schedule step: 1001
next schedule step: 1002
restart learning rate: 0.000599674418833957
```

Before training, the model, feedback-only AdamW moments, four replay loaders,
Python/NumPy/Torch CPU/Torch CUDA RNG, cumulative state, root-parent lineage,
and next-global-batch hash restored exactly. The first next-batch hash was
`1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5`.

Exactly 1,537 parameters were trainable throughout: query 768, RMSNorm scale
768, and scalar gate 1. The teacher remained frozen, in `eval()`, and under
`no_grad`; all memory was detached. All 143 new update rows had finite losses,
gradients, optimizer moments, and reader parameters; no frozen student-base or
teacher gradient was present. All global replay hashes matched before their
optimizer steps.

The update-96 forced fresh-object restart restored model, optimizer, four
loaders, all RNG, and next batch exactly. Causality passed at updates 48, 96,
and 191: each v16/v17/v20/v24 shifted memory test was bit-exact, position-zero
memory was zero, and prefix logits were bit-exact after suffix perturbation.

Checkpoints:

| Completed updates | Tokens | SHA256 |
|---:|---:|---|
| 96 | 50,331,648 | `ac209ea594795d604766b563a82a42d9f9be2bcc27ee3da1bfb39ebc12593402` |
| 191 | 100,139,008 | `6c206a89422470061d7997764fbd9a5708be3d9043f8fab930dd4b800bd5cb95` |

## Training

```text
final Experiment-2 updates: 191
final Experiment-2 tokens: 100,139,008
additional updates in 2A2: 143
additional student tokens: 74,973,184
completion invocation runtime: 11,005.09 s (3:03:25)
training-update wall time: 4,525.63 s (1:15:26)
peak allocated VRAM: 50,629.34 MiB
peak reserved VRAM: 74,778 MiB
```

| Interval | Updates | Mean loss | Last loss | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 25M→50M (49–96) | 48 | 5.731751 | 5.591563 | 5.591563 | 5.820368 |
| 50M→100M (97–191) | 95 | 5.605796 | 5.530703 | 5.462942 | 5.750841 |

These are frozen-base feedback-training losses on different global batches and
are not equal-training comparisons to the original GPT-2 pretraining curve.

## Validation trajectory

The fixed controls are full-context loss 4.0786544085 and masked/no-feedback
loss 5.9736744881, giving damage 1.8950200796.

| Tokens | Real val | Shuffled val | Total recovery % | Specific recovery % | Specific share % |
|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 5.953305 | 5.961796 | 1.0749 | 0.4481 | 41.6858 |
| 10,485,760 | 5.922936 | 5.939155 | 2.6775 | 0.8559 | 31.9653 |
| 15,204,352 | 5.894420 | 5.917988 | 4.1822 | 1.2437 | 29.7375 |
| 25,165,824 | 5.835339 | 5.876591 | 7.2999 | 2.1769 | 29.8204 |
| 50,331,648 | 5.714319 | 5.795329 | 13.6861 | 4.2749 | 31.2350 |
| 100,139,008 | 5.595705 | 5.737280 | 19.9454 | 7.4709 | 37.4567 |

The historical 5M/10M/15M/25M rows are imported unchanged from the audited
Experiment 2A1 result.

## 100M controls

```text
full context:           4.0786544085
masked/no feedback:     5.9736744881
real feedback:          5.5957053900
shuffled feedback:      5.7372799873
zero feedback:          5.9736744881

damage:                 1.8950200796
total recovery:         0.3779690981
total recovery fraction: 19.9454%

sequence-specific recovery:          0.1415745974
sequence-specific recovery fraction: 7.4709% of original damage
sequence-specific share:              37.4567% of total recovery
```

Gate-zero equaled masked/no-feedback on every batch. Full context and masked
controls reproduced their frozen references exactly.

## Paired real versus shuffled batches

The stored causal orientation is shuffled-minus-real; positive means aligned
history helps. The user-requested real-minus-shuffled mean and median are the
negative values below.

```text
real wins: 20
shuffled wins: 0
ties: 0

mean real-shuffled:   -0.1415745974
median:               -0.1439332962
standard deviation:    0.0085039980
minimum real-shuffled: -0.1552171707
maximum real-shuffled: -0.1214628220
```

For descriptive context only, the paired shuffled-minus-real 95% t interval on
the fixed 20-batch prefix is [0.137595, 0.145555]. These contiguous fixed
batches are not assumed IID, so this is not presented as a formal population
significance claim.

## Router evolution

| Tokens | Gate | Query norm | Entropy | Normalized entropy | v16 | v17 | v20 | v24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 0.005998 | 0.124895 | 1.289204 | 0.929964 | 0.271206 | 0.251649 | 0.219422 | 0.257723 |
| 10,485,760 | 0.011995 | 0.269662 | 1.037263 | 0.748227 | 0.354613 | 0.159561 | 0.198184 | 0.287643 |
| 15,204,352 | 0.017393 | 0.391837 | 0.850197 | 0.613287 | 0.411335 | 0.118092 | 0.179551 | 0.291022 |
| 25,165,824 | 0.028787 | 0.605271 | 0.737608 | 0.532072 | 0.406141 | 0.165030 | 0.169009 | 0.259819 |
| 50,331,648 | 0.057569 | 1.030883 | 0.613797 | 0.442761 | 0.420169 | 0.145839 | 0.204620 | 0.229372 |
| 100,139,008 | 0.114390 | 1.576196 | 0.584330 | 0.421505 | 0.422467 | 0.134264 | 0.216281 | 0.226988 |

Final `tanh(gate)` is 0.1138940752. Final RMSNorm displacement is
1.5883628130. Routing weight is not itself causal importance.

## Source ablations

These are renormalized leave-one-source-out interventions and are neither
additive contributions nor isolated sequence-specific effects.

| Source removed | Ablated loss | Delta versus real |
|---|---:|---:|
| v16 | 5.798497 | +0.202792 |
| v17 | 5.645828 | +0.050123 |
| v20 | 5.675327 | +0.079622 |
| v24 | 5.668870 | +0.073164 |

Every 100M delta was positive on all 20 batches. At 25M the ranking was
v16 > v24 > v20 > v17; at 100M it is v16 > v20 > v24 > v17. Dependence on
v16 grew most strongly.

## HellaSwag

The complete upstream validation set of 10,042 examples was evaluated with the
unchanged normalized-completion scorer. Each candidate received teacher states
only from itself; candidate perturbation, example reset, position-zero memory,
and RNG/model/loader isolation tests passed.

| System/control | Correct | Accuracy |
|---|---:|---:|
| Standard GPT-2 500M historical | 2,568 | 25.573% |
| Full AttnRes 500M historical | 2,532 | 25.214% |
| Full-context current | 2,532 | 25.214% |
| Masked/no feedback | 2,407 | 23.969% |
| Real teacher feedback | 2,457 | 24.467% |
| Gate zero | 2,407 | 23.969% |

Real feedback recovered 50 of the 125 examples lost by masking Layer-1 history,
without reaching the full-context anchor. Shuffled HellaSwag was not evaluated:
the existing scientific shuffle permutes batch rows, which are answer
alternatives in HellaSwag and would contaminate candidates. Scoring semantics
were not changed to manufacture this control.

These are not equal-token pretraining runs. The feedback system is a frozen
500M-token Full AttnRes base plus a reader trained on additional examples.

## Interpretation

The prespecified classifier returns **MEMORY SIGNAL ACCELERATING** because
shuffled-minus-real recovery increased 0.041252 → 0.081010 → 0.141575 from
25M → 50M → 100M, and the late gain (0.060565) exceeded the early gain
(0.039758). This supports a strengthening and increasingly sequence-dependent
teacher-feedback reader within the current frozen-teacher diagnostic.

There is enough evidence to justify a separately reviewed continuation of the
same reader from 100M to 250M: sequence-specific recovery is positive and still
increasing, real wins 20/20, total recovery is positive, all controls/invariants
pass, and real-feedback HellaSwag is 0.498 percentage points above both masked
and gate-zero (so it is not more than the preregistered one-point tolerance
below them).

This does not establish full replacement of Layer-1 history, deployable student
recurrence, or an equal-token pretraining advantage. The mechanism still uses a
detached full-context teacher and recovers only 19.95% of the validation damage.

# EXPERIMENT 2A2 100M COMPLETE
