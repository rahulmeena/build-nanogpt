# Experiment 2B2A — 4-GPU Scaling of Memory-Writer Learning

## Outcome

**WRITER MEMORY SIGNAL REVERSING**

The four-GPU migration passed every frozen equivalence bound and reduced mean
canonical training time from 4,241.43 to 784.17 seconds per update, a 5.41×
speedup. Writer-only learning continued to reduce real self-recurrent validation
loss strongly: 5.5900331020 at 5M, 5.3613477468 at 10M, and 5.0959878206 at
15M.

The canonical experiment nevertheless stopped at 15M because the controlled
sequence-specific signal reversed during the final interval. The shuffled-minus-
real gap rose to 0.0402606964 at 10M, then fell by 0.0112473011 to 0.0290133953
at 15M. The frozen gate allowed a decline of only 0.005. All integrity checks
passed and real memory still won 20/20 paired batches, so this is a scientific
gate failure rather than an implementation failure.

At the user's explicit request, a separately labeled, noncanonical exploratory
extension continued from update 29 to update 38 (19.92M tokens) to keep the pod
productive. Real loss improved again to 4.8197069407, but the specific gap
collapsed further to 0.0088150740 and real memory won 19/20. This reinforces the
canonical conclusion: continued writer-only optimization improves the raw loss
while progressively losing sequence-specific dependence on the real recurrent
memory.

No HellaSwag evaluation was run. No optimizer update after update 38 was run.

## Git

| Field | Value |
|---|---|
| 2B2 frozen tag | `experiment-2b2-writers-5m` |
| 2B2 commit | `be74383a15c1834d3b56a0767586f1a991fc5dbc` |
| 2B2A branch | `experiment-2b2a-writers-scaling-4gpu` |
| Four-GPU core commit | `b96a487` |
| Final terminal-diagnostics implementation commit | `6587500841871ea8eeb395e67f681e183391dc2b` |
| Results snapshot commit | `0e41b4590c565a5bf9901ca5aaa825345016f6fa` |
| Final-report commit | The commit containing this report; recorded in the handoff |

The immutable 2B2 tag resolves exactly to the audited 2B2 result commit. The
intervening 2B2A commits contain the annotated-tag dereference fix, migration
hash-wrapper fix, explicit exploratory authorization record, and distributed
terminal diagnostics.

## Hardware migration

| Field | Value |
|---|---|
| Source checkpoint | `/workspace/migration/experiment_2b2_5m/result/checkpoints/checkpoint_updates_000010.pt` |
| Source SHA-256 | `a125c81acb9e4ec3395bd8b38dee8fade62012c642b102a1b6c4c0e0997f0637` |
| Source next-batch SHA-256 | `e3289bee6ed5a5b2fa1d2c05a615cd3f10f07c51b71aa091ee40380ebeedc21b` |
| GPU model | 4 × NVIDIA A100-SXM4-80GB |
| 1-GPU reference loss | 5.580326527473517 |
| 4-GPU candidate loss | 5.580326527473517 |
| Absolute loss delta | 0.0 |
| Gradient cosine | 0.9999999997923387 |
| Gradient relative L2 | 2.0478076605e-5 |
| Gradient-norm relative difference | 2.0113683698e-6 |
| Gradient maximum absolute difference | 9.4994902611e-7 |
| Migration result | **PASS** |

All eight per-writer tensor comparisons passed. Their relative gradient L2
differences ranged from 1.6819e-5 to 7.0048e-5, below the frozen 1e-4 bound.

### Temporary Adam-step comparison

| Field | Value |
|---|---:|
| Parameter-update cosine | 0.9999999996807993 |
| Parameter-update relative L2 | 2.5267596709e-5 |
| Maximum parameter-update difference | 8.1956386566e-8 |
| Adam m1 relative L2 | 3.9234708612e-6 |
| Adam m2 relative L2 | 5.2619726104e-6 |

Both temporary stepped states and all temporary loader states were discarded.
The source checkpoint remained unchanged, then the result run reloaded it and
its original next-data position from scratch.

## Four-GPU geometry

| Field | Value |
|---|---|
| World size | 4 |
| Rank/GPU mapping | rank 0→GPU 0, rank 1→GPU 1, rank 2→GPU 2, rank 3→GPU 3 |
| Per-rank batch | B64 × T1024 |
| Microsteps per rank | 2 |
| Targets per rank/update | 131,072 |
| Global targets/update | 524,288 |
| Gradient synchronization | one flattened 49,152-element FP32 NCCL `all_reduce(SUM)` per global update |
| Loss scaling | each token-loss sum / 524,288; no division after SUM |
| Gradient clipping | global synchronized writer gradient, then clip to 1.0 |

The implementation uses independent local replicas and does not use automatic
DDP gradient synchronization. Frozen parameters are never communicated.

## Resume

| Field | Value |
|---|---|
| Starting writer updates | 10 |
| Starting writer-training tokens | 5,242,880 |
| Starting Adam steps | 10 for all eight writer tensors |
| Starting next-batch hash | `e3289bee6ed5a5b2fa1d2c05a615cd3f10f07c51b71aa091ee40380ebeedc21b` |
| Loader mapping | stored loader 0→rank 0, 1→rank 1, 2→rank 2, 3→rank 3 |
| RNG policy | no stochastic result-path operations found; deterministic rank seeds 2026082000–2026082003, with per-rank Python/NumPy/Torch CPU/CUDA states checkpointed |

The update-20 checkpoint was written atomically, strict-reopened, and then
loaded by four fresh worker processes before update 21. Its restored next-batch
SHA-256 was `ddbb966eff17ddabd102ce4706ccace0e23973f98803478b392b8c4e5f9d32f3`.

## Scaling trajectory

| Writer tokens | Updates | Real val | Shuffled val | Specific gap | Gain vs 2B1 | Incremental gain | Recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.24M | 10 | 5.5900331020 | 5.6259720802 | 0.0359389782 | 0.1112756252 | — | 20.2447% |
| 10.49M | 20 | 5.3613477468 | 5.4016084433 | 0.0402606964 | 0.3399609804 | 0.2286853552 | 32.3124% |
| 15.20M | 29 | **5.0959878206** | 5.1250012159 | **0.0290133953** | **0.6053209066** | **0.2653599262** | **46.3154%** |

The 25.17M canonical milestone was not reached because the 15M gate failed.

## Milestones

### 10M / update 20

| Field | Value |
|---|---:|
| Writer-training tokens | 10,485,760 |
| Real validation | 5.36134774684906 |
| Shuffled validation | 5.4016084432601925 |
| Specific gap | 0.04026069641113228 |
| Incremental real gain | 0.22868535518646205 |
| Real / shuffled wins / ties | 20 / 0 / 0 |
| Median paired difference | 0.040288686752319336 |
| Sample standard deviation | 0.00831863920930631 |
| Minimum / maximum | 0.02292490005493164 / 0.05113935470581055 |
| Continuation gate | **PASS** |

Checkpoint SHA-256:
`de5e04f817dcfa5dd8a4dcc6e503ec86d8545d558d837b517c7259917218dff3`.
Model, optimizer, four loader states, four rank RNG states, Adam steps, next
batch, and cross-rank hashes strict-verified.

### 15M / update 29 — canonical terminal

| Field | Value |
|---|---:|
| Writer-training tokens | 15,204,352 |
| Real validation | 5.0959878206253055 |
| Shuffled validation | 5.125001215934754 |
| Specific gap | 0.029013395309448242 |
| Incremental real gain | 0.2653599262237547 |
| Specific-gap change from 10M | -0.011247301101684037 |
| Real / shuffled wins / ties | 20 / 0 / 0 |
| Median paired difference | 0.029801130294799805 |
| Sample standard deviation | 0.006576442445044017 |
| Minimum / maximum | 0.017365455627441406 / 0.041249752044677734 |
| Continuation gate | **FAIL** |

The real-improvement, 18/20-win, and integrity conditions passed. Only the
specific-gap condition failed: the required minimum was 0.0352606964, while the
measured value was 0.0290133953.

Canonical terminal checkpoint:

```text
/workspace/runs/experiment_2b2a/checkpoints/checkpoint_updates_000029.pt
SHA-256: 86c66343141e24d0beffcf8bc98a558f25c82e1dc05582feade2300d30b2ba84
next batch: 8b9fe2fa1c2a10ce930caff4d527c48e4f14ab0e1a6f5e4b352e42f61b8b360d
```

## Terminal controls

| Condition | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked L1 / no feedback | 5.9736744881 |
| Frozen 2B1 self | 5.7013087273 |
| Terminal writer real self | **5.0959878206** |
| Terminal writer shuffled self | 5.1250012159 |
| Terminal writer bypass | 5.7013087273 |
| Recurrent gate zero | 5.9736480713 |
| Teacher sources + terminal writers | 5.2170396566 |

Writer bypass matched frozen 2B1 exactly, and gate zero matched the established
gate-zero control exactly. All frozen-control regressions were within tolerance.
Teacher training forward calls were exactly zero.

## Recovery

| Metric | Value |
|---|---:|
| Gain vs frozen 2B1 | 0.6053209066 |
| Recovery fraction | 46.3154% |
| 15M specific gap | 0.0290133953 |
| Specific-gap change from 5M | -0.0069255829 |
| Specific-gap change from 10M | -0.0112473011 |

## Writer behavior

| Source | Delta RMS | Source RMS | Delta/source | Maximum per-example ratio | Source/adapted cosine |
|---|---:|---:|---:|---:|---:|
| v16 | 0.009243829 | 0.194199243 | 0.049116402 | 0.165840104 | 0.998499447 |
| v17 | 0.018940374 | 0.200206386 | 0.097499142 | 0.166317090 | 0.995050228 |
| v20 | 0.009926124 | 0.197834985 | 0.052848685 | 0.177937269 | 0.998175701 |
| v24 | 0.007250902 | 0.192377900 | 0.039606757 | 0.159044698 | 0.998967868 |

All mean residual/source ratios remained below the 10% warning threshold and
well below the 25% hard stop at the canonical terminal. Recurrent-state norms
remained finite and stable.

## Writer evolution

| Tokens | v16 ratio | v17 ratio | v20 ratio | v24 ratio | v16 route | v17 route | v20 route | v24 route | Entropy | Feedback RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.24M | 0.008786 | 0.015700 | 0.009319 | 0.007584 | 0.626302 | 0.017401 | 0.141689 | 0.214608 | 0.479758 | 0.025694 |
| 10.49M | 0.026607 | 0.052944 | 0.027696 | 0.021906 | 0.601998 | 0.014763 | 0.170834 | 0.212405 | 0.468923 | 0.025805 |
| 15.20M | 0.049116 | 0.097499 | 0.052849 | 0.039607 | 0.553503 | 0.024143 | 0.218052 | 0.204301 | 0.465643 | 0.025985 |

The reader continued to emphasize v16, while v20 routing grew. v17 underwent
the largest relative representation change but remained lightly routed.

## Writer residual ablations

Only the learned residual was disabled; the raw source remained available.

| Writer | Ablated loss | Delta | Positive batches |
|---|---:|---:|---:|
| v16 | 5.3486131668 | +0.2526253462 | 20/20 |
| v17 | 5.1082276106 | +0.0122397900 | 20/20 |
| v20 | 5.1509248018 | +0.0549369812 | 20/20 |
| v24 | 5.1451255322 | +0.0491377115 | 20/20 |

All four writers remained useful on every canonical batch. v16 was dominant;
v20 and v24 were substantial; v17 was small but consistently favorable.

## Reset horizon

Only recurrent high→low memory was reset. Blocks 2–12 K/V and absolute position
continued normally, and each condition retained B64×T1024.

| Interval | Frozen 2B1 | 2B2 5M | Terminal 2B2A 15M |
|---:|---:|---:|---:|
| 1 | 5.9737106323 | 5.9737106323 | 5.9736480713 |
| 2 | 5.8261914968 | 5.7768767834 | 5.5396609545 |
| 4 | 5.7499492407 | 5.6721416473 | 5.3130240679 |
| 8 | 5.7218983650 | 5.6283501625 | 5.2043139219 |
| 16 | 5.7104103088 | 5.6082828522 | 5.1507718086 |
| 32 | 5.7059802055 | 5.5992109060 | 5.1231377602 |
| 64 | 5.7035196781 | 5.5941849947 | 5.1087827206 |
| 128 | 5.7021847963 | 5.5918534994 | 5.1015742540 |
| Never | 5.7013566494 | 5.5900463343 | **5.0959878206** |

The terminal curve remained monotonic: longer memory persistence consistently
improved validation loss.

## Representation diagnostics

Teacher-vs-raw and teacher-vs-adapted cosine, RMS difference, and norm ratio were
measured for v16/v17/v20/v24 over all seven preregistered token windows. The full
matrix is stored in `terminal/controls_15m.json`. The writer transform generally
made small changes relative to the raw recurrent state; v17 showed the clearest
late-token movement toward the teacher, while self-recurrent validation already
outperformed teacher-source evaluation. No teacher diagnostic entered the loss.

## Performance

| Execution | GPUs | sec/update | targets/sec | Speedup | Scaling efficiency |
|---|---:|---:|---:|---:|---:|
| 2B2 5M historical | 1 | 4241.43 | 123.61 | 1.00× | 100% |
| 2B2A canonical updates 11–29 | 4 | **784.17** | **668.68** | **5.41×** | **135.22%** |

Average rank-0 NCCL all-reduce timing, including any wait for slower ranks, was
18.2436 seconds/update, or 2.3265% of total update wall time. Update 11 took
797.57 seconds, with per-rank total step times of 797.49, 762.21, 755.33, and
756.47 seconds. Its all-reduce call took 0.000370 seconds on the timed rank.

The requested forward/backward/optimizer phase split was not persisted by the
training logger; only per-rank total step time, gradient-sync time, wall time,
VRAM, and throughput were recorded. No extra optimizer update was launched to
retroactively manufacture that diagnostic. This is an instrumentation limitation,
not a scientific acceptance criterion.

Peak allocated VRAM was 40,888.13 MiB per rank; peak reserved VRAM was 79,968
MiB per rank.

## Integrity

| Requirement | Result |
|---|---|
| Trainable writer parameters | PASS; exactly 49,152 |
| Base / Full-AttnRes frozen | PASS |
| 2B1 reader frozen | PASS |
| Teacher training calls | PASS; 0 |
| Loss(t+1) → writer(t) | PASS; present, finite, nonzero |
| Loss(t+2) → writer(t) | PASS; absent |
| Historical-K/V temporal gradient | PASS; absent |
| Block-1 historical K/V | PASS; absent |
| Blocks 2–12 K/V | PASS; finite and detached |
| Future causality / row isolation | PASS / PASS |
| Global targets/update | PASS; 524,288 |
| Replay hashes | PASS; exact |
| Cross-rank writer equality | PASS after every update |
| Cross-rank optimizer equality | PASS after every update |
| Update-20 checkpoint | PASS; strict reload and fresh four-process restart |
| Update-29 checkpoint | PASS; strict reload |
| Canonical validation hash | PASS; exact |
| Terminal controls / ablations / reset / drift | PASS |
| HellaSwag | Not run, as required |

The update-20, update-29, and exploratory update-38 checkpoint files were copied
from the pod to the local workstation. Their local SHA-256 values match the pod
and their verification sidecars exactly. The pod and its pod volume were left
running and untouched; nothing was deleted remotely.

## Exploratory 20M override — noncanonical

This section is not part of the canonical 2B2A stopping decision.

| Field | Value |
|---|---:|
| Writer update | 38 |
| Writer-training tokens | 19,922,944 |
| Real validation | 4.81970694065094 |
| Shuffled validation | 4.82852201461792 |
| Specific gap | 0.008815073966979448 |
| Incremental real gain from 15M | 0.27628087997436523 |
| Real / shuffled wins / ties | 19 / 1 / 0 |
| Mean v17 residual/source | 0.1479000606 |
| Checkpoint SHA-256 | `d65ff192e037862008d85253a215d3112922c9c8365a576671462f3eaf56a838` |
| Next-batch SHA-256 | `7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7` |

The exploratory checkpoint strict-verified with four loader states, four rank
RNG states, and Adam step 38 for all eight writer tensors. It is preserved for
analysis, but it must not be mistaken for a gate-authorized canonical milestone.

## Final classification

**WRITER MEMORY SIGNAL REVERSING**

The frozen labels do not include a mixed case in which real loss keeps improving
while sequence specificity reverses. The experiment's controlled memory-signal
measure is the shuffled-minus-real gap, and that measure materially worsened with
valid integrity during the final canonical interval. Therefore REVERSING is the
closest and scientifically conservative frozen classification. It is not
UNSTABLE: every numerical and systems-integrity check passed.

## Decisions

### Decision A — switch to joint writer + reader optimization?

**Yes, if a new experiment is approved.** Writer-only real loss had not saturated,
but its sequence-specific advantage was reversing. The reason to switch is not
loss saturation; it is that further writer-only optimization increasingly looks
like generic compensation rather than better use of the actual recurrent memory.

### Decision B — keep temporal credit exactly one token?

**Yes.** One-token temporal credit remained stable, causal, and highly effective.
This experiment supplies no evidence that a longer gradient horizon is needed.

### Decision C — is teacher-assisted optimization still useful as a target?

**No, not as the next training target.** The distinct systems are:

| System | Loss |
|---|---:|
| Raw teacher memory with the frozen reader | 5.5720659256 |
| Teacher sources + 5M learned writers | 5.5216585398 |
| Teacher sources + terminal 15M writers | 5.2170396566 |
| Self-recurrent terminal 15M writers | **5.0959878206** |

The self-recurrent system now beats teacher sources passed through the same
terminal writers by 0.1210518360. Teacher states remain valuable as an
evaluation diagnostic, but teacher-assisted optimization is no longer justified
as the primary training target by these results.

### Decision D — use this four-rank implementation in future recurrent experiments?

**Yes.** The migration matched the 1-GPU reference within all frozen loss,
gradient, optimizer-update, and state bounds, maintained exact replay and
cross-rank equality, and delivered a 5.41× measured speedup.

Do not launch joint writer-reader training, longer temporal credit, HellaSwag,
or any further writer optimizer update without a newly defined and explicitly
approved experiment.

# EXPERIMENT 2B2A COMPLETE
