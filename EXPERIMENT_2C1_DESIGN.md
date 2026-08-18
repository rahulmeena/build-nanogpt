# Experiment 2C1 — Destination-Depth Sweep for Sequence-Specific High→Low Attention Residual Memory

Status: preregistered before implementation and result execution on 2026-08-18.

This repository document freezes the user-supplied Experiment 2C1 protocol.
It supersedes the narrower single-destination recommendation in the 2C0
handover. Experiment 2C0 is frozen and must not be rerun.

## 1. Scientific objective

Experiment 2C1 tests whether the usefulness of correct sequence-specific
high→low feedback increases when the feedback destination moves upward through
the Transformer. It is a teacher-assisted destination-depth experiment so that
destination requirements are isolated from self-recurrent representation
drift during training.

The experiment uses four independent single-reader models:

| ID | Destination | Zero-indexed block | GPU |
|---|---|---:|---:|
| D1 | Block 1 attention input | 0 | 0 |
| D5 | Block 5 attention input | 4 | 1 |
| D9 | Block 9 attention input | 8 | 2 |
| D12 | Block 12 attention input | 11 | 3 |

Each model contains exactly one new reader and one masked attention
destination. A model containing four readers is forbidden.

## 2. Immutable Git lineage

- frozen 2C0 branch: `experiment-2c0-separated-generic-sequence-b1`;
- frozen 2C0 commit:
  `677d711bc00dba0da1b80cb6369f33841ec29a51`;
- immutable tag: `experiment-2c0-separated-b1-final`, pointing exactly to the
  frozen commit;
- 2C1 branch: `experiment-2c1-destination-depth-sweep`, created from that tag.

No earlier experiment branch or tag may be modified.

## 3. Frozen base model

All destinations start from the same immutable Experiment 1B Full-AttnRes
checkpoint:

```text
/workspace/build-nanogpt/runs/exp1b_500m/full_attnres/checkpoints/checkpoint_tokens_000500170752.pt
SHA256 6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91
```

Expected state:

- 954 completed updates;
- 500,170,752 tokens;
- 124,514,304 parameters;
- original world size 4.

The checkpoint must strict-load. No Experiment 2 reader or writer weight is
part of the starting architecture. Every destination reader starts fresh.

## 4. Hardware and independent counterfactuals

Use exactly four NVIDIA A100-SXM4-80GB GPUs. Run four independent single-GPU
processes with the mapping in the destination table. DDP, gradient
synchronization, and cross-destination optimizer sharing are forbidden.

Each worker owns its own:

- model replica;
- frozen full-context teacher replica;
- reader;
- optimizer;
- private cloned replay-loader state;
- checkpoints and RNG state.

All destinations consume the same FineWeb global batches in the same order.
The global batch hash at every update must be identical across D1/D5/D9/D12.
A mismatch is a hard stop.

Use private clones of the four replay-loader states corresponding to the
original 500M Experiment 1B parent. Advancing one destination must not advance
another.

## 5. Frozen full-context teacher and source bank

For every destination, construct an immutable full-context teacher from the
same Experiment 1B checkpoint. The teacher is in `eval()` mode, runs under
`torch.no_grad()`, and has no trainable parameters or gradients.

Capture raw teacher residuals at the same four depths for every destination:

```text
v16 = Block 8 MLP output
v17 = Block 9 attention output
v20 = Block 10 MLP output
v24 = Block 12 MLP output
```

Shift the source bank by exactly one token:

```text
memory[:, :, 0, :] = 0
memory[:, :, t, :] = teacher_source[:, :, t-1, :]
```

There is no wraparound, future information, cross-row state, or temporal
gradient. The source bank cannot be customized by destination.

## 6. Destination mask semantics

For destination `d`, remove historical K/V only from Block `d`. Every other
attention block retains normal causal history:

```text
D1:  B1 history absent; B2–B12 normal
D5:  B5 history absent; B1–B4 and B6–B12 normal
D9:  B9 history absent; B1–B8 and B10–B12 normal
D12: B12 history absent; B1–B11 normal
```

The masked block retains the current token, input hidden state, attention
module, self/current-token attention, MLP, and all same-token computation. The
block is neither zeroed nor bypassed. Masking all lower layers is forbidden.

The complete Experiment 1B same-token Full-AttnRes architecture remains
unchanged, including its queries, RMSNorms, residual bank, lower→higher routing,
and final router. Temporal feedback is additive.

## 7. One top-down reader per destination

Each independent destination reader implements:

```text
key_j = RMSNorm(memory_j)
score_j = q_destination^T key_j
beta = softmax(score over the four source depths)
topdown = sum_j beta_j * memory_j
h_destination = h_normal + tanh(g_destination) * topdown
```

Use raw teacher residuals as values, one learned query, one learned RMSNorm
scale, no bias, no `1/sqrt(768)`, and softmax only over source depth.

Initialize every reader identically:

- query exactly zero;
- RMSNorm scale exactly one;
- scalar gate exactly zero;
- exactly 1,537 trainable parameters (`768 + 768 + 1`).

All base-model and teacher parameters are frozen.

## 8. Mandatory deterministic preflight

Before result training, test all four destinations in FP32. Require:

- only the target layer lacks history;
- every non-target cache has the exact expected length;
- current-token path unchanged;
- future suffix cannot alter a prefix;
- row isolation passes;
- fresh sequence reset passes;
- gate-zero equals masked/no-feedback;
- all outputs are finite;
- D1 matches the previously validated Block-1 intervention;
- trainable parameter count is exactly 1,537 at every destination;
- base and teacher gradients are absent.

Any causality, cache-policy, row-isolation, reset, zero-gate, finiteness, load,
or parameter-count failure is a hard integrity stop.

## 9. Frozen damage measurement

Before training, evaluate the canonical validation prefix for:

- full context;
- each destination masked with no feedback.

Use the frozen full-context reference:

```text
L_full = 4.0786544085
damage[d] = L_masked[d] - L_full
```

If `damage[d] < 0.02`, label the destination `LOW MASK DAMAGE`. Continue to
report raw real-vs-shuffled gaps, but do not over-interpret normalized recovery.
Do not amplify the mask.

No initial HellaSwag evaluation is authorized.

## 10. Optimizer and global geometry

Each destination gets a fresh independent AdamW optimizer over only its 1,537
reader parameters:

```text
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0
```

All destinations use the same original teacher-reader learning-rate schedule:
the original 10B schedule position associated with the 500M parent, beginning
at the same reader-training schedule step as historical Experiment 2A. LR may
not be tuned independently.

Each single GPU independently reconstructs the original global update using
eight serialized `B64 × T1024` slices:

```text
524,288 targets per optimizer update
```

All destinations receive identical batches, tokens, schedules, and update
counts.

## 11. Training budget and forced restart

Train every destination for exactly 48 reader updates / 25,165,824
reader-training tokens unless a hard integrity failure occurs.

Frozen evaluation milestones:

| Update | Tokens |
|---:|---:|
| 10 | 5,242,880 |
| 20 | 10,485,760 |
| 29 | 15,204,352 |
| 48 | 25,165,824 |

After update 20, write a complete atomic checkpoint for every destination,
terminate each worker, and continue only in fresh processes on the same GPUs.
Strictly restore reader, AdamW, four cloned loader states, Python/NumPy/Torch
CPU/CUDA RNG, schedule position, and next-batch hash.

Each checkpoint records:

- schema and destination block;
- base-model SHA;
- reader and AdamW states;
- reader update count and processed tokens;
- all four cloned replay-loader states;
- all RNG states;
- next-global-batch hash;
- source depths and mask semantics;
- Git/config lineage.

Write atomically, strict-reopen, compute SHA256, and never overwrite another
destination's checkpoint.

## 12. D1 historical regression

D1 is the protocol regression. At approximately 25M historical Block-1
teacher-reader results were:

```text
real = 5.8353391409
shuffled = 5.8765912533
specific_gap = 0.0412521124
```

D1 must reproduce the historical trajectory within established deterministic
and numerical tolerance. If it materially fails, stop interpreting D5/D9/D12
as comparable destination results and diagnose the mismatch first.

## 13. Milestone evaluation controls

At 5M, 10M, 15M, and 25M, evaluate every destination on the same canonical
`20 × B64 × T1024` BF16 validation prefix, aggregate SHA256:

```text
3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb
```

Controls:

1. target masked / no feedback;
2. target masked + real aligned teacher feedback;
3. target masked + coherent shuffled teacher feedback;
4. target masked + gate zero.

For every canonical B64 batch, use one deterministic fixed-point-free row
permutation. The same donor row supplies v16/v17/v20/v24 at all positions. Use
the same permutation at all destinations. This preserves source, position,
distribution, and temporal coherence while destroying receiver/donor identity.

## 14. Primary and recovery metrics

For each destination and milestone:

```text
specific_gap[d] = L_shuffled[d] - L_real[d]
recovery[d] = L_masked[d] - L_real[d]
recovery_fraction[d] = recovery[d] / damage[d]       if damage[d] > 0
specific_fraction[d] = specific_gap[d] / damage[d]   if damage is nontrivial
specific_share[d] = specific_gap[d] / recovery[d]    if recovery[d] > 0
```

Do not interpret these ratios as additive decompositions.

Save all 20 real and shuffled losses and their paired differences. Report real
wins, shuffled wins, ties, mean, median, sample standard deviation, minimum,
and maximum gap.

At every milestone report:

- gate and `tanh(gate)`;
- query norm;
- RMSNorm displacement;
- routing entropy;
- mean v16/v17/v20/v24 routing weights;
- top-down RMS;
- feedback RMS.

Small raw improvement is not an early-stop condition. Real versus shuffled is
the primary scientific metric.

## 15. Final generic-template diagnostic

At 25M only, reuse the exact disjoint Experiment 2B4 calibration dataset to
compute frozen full-context-teacher shifted-source means `mu16`, `mu17`,
`mu20`, and `mu24`. Because teacher and sources are identical, reuse the same
means across destinations.

For each trained reader, evaluate the destination-specific generic memory:

```text
position 0 = zero
position t > 0 = (mu16, mu17, mu20, mu24)
```

Record `L_generic[d]` and:

```text
generic_recovery[d] = L_masked[d] - L_generic[d]
generic_vs_real_delta[d] = L_generic[d] - L_real[d]
```

Compare generic, real, and shuffled without changing training.

## 16. Frozen support criteria

Define `HIERARCHICAL SEQUENCE-MEMORY SUPPORT` if at least one of B5/B9/B12
satisfies all of:

```text
specific_gap >= 0.050
real wins >= 18 / 20
recovery > 0
L_real + 0.010 <= L_generic
all integrity checks pass
```

Perfect monotonicity is not required. Also report, without changing the primary
threshold:

```text
specific_gap[d] - specific_gap[B1]
```

for B5/B9/B12. A value at least `0.020` is strong secondary depth evidence.

## 17. Forbidden training changes

The following are forbidden in 2C1:

- writer adapters;
- a generic or constant branch during training;
- auxiliary, margin, contrastive, KL, reconstruction, teacher-matching, or RL
  objectives;
- self-recurrent training;
- BPTT or temporal gradients;
- destination-specific source banks;
- DDP or cross-destination gradient synchronization;
- HellaSwag;
- training beyond 48 updates.

Teacher memory is detached. Training uses next-token CE only.

## 18. Conditional self-recurrent zero-shot transfer

After 25M teacher-assisted evaluation, a destination independently qualifies
for a self-recurrent test only if:

```text
teacher specific_gap >= 0.020
teacher real wins >= 18 / 20
teacher recovery > 0
all integrity checks pass
```

For qualifying destinations only, replace teacher memory with the student's own
previous-token v16/v17/v20/v24 states using the exact trained reader. Do not
train, retrain, rescale, or change the gate.

The target block still has no historical K/V; every non-target block keeps its
normal KV cache. Memory at position zero is zero. There is no cross-row leakage
or future information.

Evaluate canonical self-real, coherent self-shuffled, and self-gate-zero. Save:

```text
self_specific_gap = L_self_shuffled - L_self_real
self / teacher recovery ratio
```

For failed destinations, record `SELF TEST NOT TRIGGERED` and spend no recurrent
evaluation compute.

## 19. Hard integrity rules

Every destination must satisfy:

- exactly 1,537 trainable parameters;
- no base or teacher gradients;
- teacher in eval/no-grad;
- future causality and row isolation;
- exactly the target block lacks historical K/V;
- all non-target caches are normal;
- gate zero equals masked/no-feedback;
- exactly 524,288 targets/update;
- identical cross-destination batch hashes;
- finite losses, gradients, optimizer state, and parameters;
- checkpoint strict reload and exact continuation.

Use `DESTINATION-DEPTH EXPERIMENT UNSTABLE` only for NaN/Inf, causality
failure, mask failure, batch-stream mismatch, checkpoint mismatch, or cross-row
leakage.

## 20. Final classification

Choose exactly one:

```text
SEQUENCE MEMORY NEED INCREASES WITH DESTINATION DEPTH
SEQUENCE MEMORY IS STRONG ONLY AT SPECIFIC DEPTHS
GENERIC CORRECTION DOMINATES ACROSS DEPTHS
DESTINATION MASK DAMAGE TOO SMALL TO RESOLVE HIERARCHY
DESTINATION-DEPTH RESULT IS MIXED
DESTINATION-DEPTH EXPERIMENT UNSTABLE
```

Use the first when the frozen support criterion passes and higher-destination
gaps show an overall increasing trend relative to B1. Use the second when
support passes but the trajectory is clearly non-monotonic. Use generic
dominance when no higher destination reaches a `0.050` gap and the generic
template is comparable to or better than real memory at all meaningful-damage
destinations. Use low damage when most B5/B9/B12 damages are below `0.02`.
Otherwise use mixed.

## 21. Artifacts and reporting

Save under:

```text
results/experiment_2c1_destination_depth/
```

Required top-level artifacts:

```text
EXPERIMENT_2C1_FINAL_REPORT.md
result_summary.json
FINAL_AUDIT.json
destination_damage.json
destination_trajectories.json
destination_paired_losses.json
destination_router_stats.json
generic_template_controls.json
self_transfer.json
performance.json
checkpoint_manifest.json
```

Required destination subdirectories:

```text
D1_block1/
D5_block5/
D9_block9/
D12_block12/
```

The final report must include frozen provenance, damage, trajectories,
sequence-specific results, generic controls, router specialization,
conditional self transfer, performance, and every integrity invariant. It must
answer Q1–Q7 and recommendations A–F from the supplied protocol, select exactly
one frozen classification, and end exactly with:

```text
# EXPERIMENT 2C1 COMPLETE
```

Commit and push implementation/tests before result training. Then commit result
artifacts, run the final audit, commit the final report, push all commits, and
leave local and pod worktrees clean. Do not launch writers, continuation beyond
25M, multiple simultaneous feedback destinations in one model, BPTT, iterative
loops, auxiliary objectives, or HellaSwag without explicit approval.
