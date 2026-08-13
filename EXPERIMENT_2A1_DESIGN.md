# Experiment 2A1 — exact continuation of teacher feedback to 25M tokens

## Status and immutable lineage

This document freezes the Experiment 2A1 continuation before its first new
optimizer step. Experiment 2A1 changes no model architecture. It continues the
feedback-only optimizer and state produced by the verified Experiment 2A0 5M
result.

The immutable root remains the Experiment 1B Full AttnRes checkpoint:

```text
commit: abecd3e91e89e1259f7198d72d15664943ad48bf
checkpoint SHA256: 6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91
```

The immediate continuation checkpoint is:

```text
/workspace/build-nanogpt-exp2a0/runs/experiment_2a0_5m/checkpoints/checkpoint_updates_000010.pt
SHA256: cf68b9765072e2403c16e935ba02e92f826d48600953f904e11f2bd4d266638e
completed Experiment-2 updates: 10
student tokens: 5,242,880
next global-batch SHA256: 01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a
```

Before any new optimizer step, the runner must validate both checkpoint
sidecars and bytes, strict-load the complete student, load the existing
feedback-only AdamW state, restore all four replay loaders and all RNG state,
and reproduce the stored next-batch hash. It must abort rather than reinitialize
the query, RMSNorm, gate, optimizer, loader positions, or RNG.

## Architecture and optimization boundary

The architecture is exactly Experiment 2A0:

- Block-1 Attention has current-token V access and no historical K/V;
- frozen full-context teacher sources are raw `v16`, `v17`, `v20`, and `v24`;
- each source is detached and shifted by exactly one token without wraparound;
- the independent top-down router uses shared RMSNorm keys, raw values, a
  depth-only softmax, and one scalar `tanh` gate;
- feedback is injected only at the Block-1 Attention input;
- the Full AttnRes/GPT-2 student base and the entire teacher remain frozen.

Exactly three tensors and 1,537 scalars remain trainable:

```text
query:          768
RMSNorm scale:  768
scalar gate:      1
```

No destination, memory source, distillation target, recurrence, loss, or
unfreezing is added. `train_gpt2.py` is unchanged from the completed 2A0 run.

## Cumulative training geometry and schedule

Experiment-update indices are cumulative. New metric rows are indices 10
through 47, not a new zero-based run.

```text
source completed updates: 10
target completed updates: 48
additional updates:       38
tokens per update:         524,288
additional tokens:         19,922,944
final cumulative tokens:   25,165,824
```

Every update serializes the original four-rank batch exactly:

```text
four replay loaders
B=64 per rank
T=1024
gradient accumulation=2
eight ordered one-GPU microbatches per update
```

The original 10B-token LR schedule continues without restart. The first new
row, update index 10, uses global schedule step 964. The last row, update index
47, uses step 1001.

The runner independently derives all 38 expected global-batch hashes from a
clone of the restored checkpoint loaders and compares each actual hash before
its optimizer step. A mismatch is a hard stop.

## Milestones and durable state

Complete atomic checkpoints are published before milestone evaluation:

| Completed updates | Cumulative tokens | Last schedule step | Purpose |
| ---: | ---: | ---: | --- |
| 20 | 10,485,760 | 973 | real/shuffled evaluation |
| 29 | 15,204,352 | 982 | real/shuffled evaluation |
| 48 | 25,165,824 | 1001 | final controls and ablations |

Each checkpoint retains the established schema: complete student, feedback
AdamW, four loader states, Python/NumPy/Torch CPU/Torch CUDA RNG, cumulative
update/token state, immutable root-parent SHA, new 2A1 metadata, and next-batch
hash. Metadata additionally binds the immediate update-10 source checkpoint
SHA and source configuration/code hashes. Checkpoints are saved to temporary
files, fsynced, strictly reopened, hashed, and atomically renamed; existing
artifacts are never overwritten.

After each nonfinal milestone, a fresh student, optimizer, and loader group is
constructed and the just-written checkpoint is strict-restored before training
continues. Model, optimizer, loader, RNG, and next-batch state must match at the
restart boundary.

Explicit crash recovery accepts only a verified milestone checkpoint. Metrics
belonging to 2A1 must be exactly update indices 10 through
`completed_updates-1`; duplicate, missing, or pre-10 rows are rejected, while
uncommitted trailing rows beyond a durable checkpoint may be truncated before
deterministic replay. A checkpoint published before a failed evaluation can be
loaded to rerun evaluation without repeating optimizer steps.

## Evaluation protocol

All milestones use the same canonical FineWeb validation prefix:

```text
20 serial batches
B=64
T=1024
1,310,720 target tokens
BF16 autocast
SHA256: 3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb
```

The update-10 source is evaluated read-only before continuation to reconstruct
the paired baseline. Updates 20 and 29 evaluate only aligned real feedback and
the fixed-point-free one-row-roll shuffled feedback. Update 48 evaluates:

1. full context;
2. masked L1 without feedback;
3. aligned real feedback;
4. shuffled feedback;
5. trained feedback with gate forced exactly to zero;
6. leave-one-source-out masks for `v16`, `v17`, `v20`, and `v24`.

Teacher memory is computed once for each validation batch and reused across
feedback controls. The shuffle permutation is deterministic, bijective,
fixed-point-free, and shared by all four sources. The source mask is reset in a
`finally` block. Evaluation must leave student tensors, optimizer, training
loaders, and RNG bit-exact.

Every milestone artifact is bound to the milestone checkpoint SHA, cumulative
state, code/config hashes, runtime, canonical validation hash, and twenty raw
real and shuffled batch losses. The final artifact retains raw per-batch losses
for every control and ablation.

## Metrics and paired descriptions

The final V-only masked baseline from the completed matched 2A0 run is the
pinned intermediate reference:

```text
L_full   = 4.078654408454895
L_masked = 5.973674488067627
```

At every milestone:

```text
damage                         = L_masked - L_full
total_recovery                 = L_masked - L_real
total_recovery_fraction        = total_recovery / damage
sequence_specific_recovery     = L_shuffled - L_real
sequence_specific_fraction     = sequence_specific_recovery / damage
```

For each paired batch, `q = shuffled_loss - real_loss`; positive `q` favors
correct-sequence memory. Artifacts report the mean, median, minimum, maximum,
positive/negative/tie counts, sample standard deviation, standard error, and a
descriptive 95% paired t interval with 19 degrees of freedom. These fixed
contiguous validation batches are not assumed IID, so the interval is preserved
for descriptive comparison and is not called a significance test.

Router statistics are collected only from unmasked real feedback and include
the raw and `tanh` gate, query norm, RMSNorm displacement, mean tokenwise
entropy, normalized entropy, effective source count, and the four mean weights.
Mean routing weight is descriptive routing behavior, not causal importance.
Final source ablations are renormalized leave-one-source-out effects.

## Continuous validity checks

The run aborts on any failure of:

- exactly 1,537 trainable parameters and exactly three optimizer states;
- teacher `eval()`, fully frozen, `no_grad`, detached memory;
- no gradient on a frozen student or teacher parameter;
- finite loss, gradient, gradient norm, new tensor, routing statistic, and Adam
  moment;
- expected cumulative token count, update index, schedule step, LR, replay hash,
  or optimizer step;
- frozen-base or teacher state hash equality;
- production future-token invariance and zero-at-position-zero memory;
- checkpoint SHA/sidecar/schema/lineage/state/reload checks;
- canonical validation hash, paired vector length/alignment, or routing-simplex
  checks;
- final full-context equality, matched masked baseline equality, and per-batch
  gate-zero equality to masked/no-feedback.

## End condition and interpretation

The only permitted terminal training state is 48 completed Experiment updates
and 25,165,824 cumulative student tokens. No 50M continuation, recurrent loop,
or follow-on experiment is launched.

The final report preserves the trajectory at 5M, 10M, 15M, and 25M and selects
exactly one user-requested qualitative label from:

```text
MEMORY SIGNAL STRENGTHENING
MEMORY SIGNAL STABLE
MEMORY SIGNAL SATURATING
MEMORY SIGNAL DISAPPEARING
```

The label is frozen before training from the directly remeasured
`shuffled-real` values `s` at updates 10, 20, 29, and 48:

- `DISAPPEARING` if the final value is nonpositive, or is below half of a
  positive update-10 value;
- `STABLE` if the full range is at most 25% of
  `max(abs(mean(s)), 1e-12)`;
- otherwise `SATURATING` if the final value is below an earlier maximum, or if
  a positive update-10-to-29 gain is followed by an update-29-to-48 gain no
  larger than 25% of it;
- otherwise `STRENGTHENING`.

Both `shuffled-real` (positive favors aligned memory) and the user-requested
`real-shuffled` orientation are reported explicitly, along with paired
uncertainty and batch sign counts, so the qualitative classification is
auditable. Regardless of outcome, the experiment remains a detached
full-context-teacher diagnostic and is not described as deployable student
recurrence.
