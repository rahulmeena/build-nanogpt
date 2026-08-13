# Experiment 2A0 Design — Teacher-Assisted Top-Down Feedback into Block 1

## Status and scope

This document freezes the Experiment 2A0 design before any optimizer step is
run. The parent is Experiment 1B commit
`abecd3e91e89e1259f7198d72d15664943ad48bf`. Experiment 2A0 asks only whether
detached higher-layer state from token `t-1` can compensate for removing Block
1's access to token positions `< t`.

The experiment retains sequence length 1024, learned GPT-2 positional
embeddings, normal causal attention in Blocks 2–12, and the complete trained
lower-to-higher Full AttnRes computation. It does not add student recurrence,
persistent memory, additional feedback destinations, longer context, or an
auxiliary objective.

## Corrected Experiment 1B geometry

The finalized Experiment 1B harness used four GPUs with per-rank `B=64`,
`T=1024`, and gradient accumulation 2. The global batch was 524,288 tokens.
The earlier handoff's `B=32`/accumulation-4 description was incorrect, but it
does not change the recorded global token counts or learning-rate milestones.

## Models and immutable initialization

The source checkpoint is:

```text
/workspace/build-nanogpt/runs/exp1b_500m/full_attnres/checkpoints/checkpoint_tokens_000500170752.pt
SHA256 6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91
```

It is read-only. Two models are constructed from its model state:

1. A frozen teacher with the original Experiment 1B architecture and strict
   state-dict loading.
2. A student with the same base architecture plus exactly one independent
   top-down router and gate. Loading must narrowly whitelist only the three new
   tensors.

The teacher is always in evaluation mode, always runs under `torch.no_grad()`,
and every captured source is detached. The student base and every existing
lower-to-higher AttnRes parameter are frozen. Only the new top-down query,
top-down RMSNorm scale, and feedback gate are trainable.

## Exact timing and causality

For input tokens `x[:, 0:T]`, the full-context teacher produces the raw Full
AttnRes residual values at every position for exactly these depths:

```text
v16 = Block 8 MLP output
v17 = Block 9 Attention output
v20 = Block 10 MLP output
v24 = Block 12 MLP output
```

The memory bank has shape `[4, B, T, C]`, where source depth is the first
dimension and `C=768` for GPT-2 124M. It is shifted without wraparound:

```text
memory[:, :, 0, :] = 0
memory[:, :, t, :] = teacher_sources[:, :, t-1, :]  for t > 0
```

Thus the student at `t` can receive information computed only after the teacher
processed `x_t-1`. There is no gradient through time, through the shift, or
into the teacher. Changing `x_t...x_T-1` must leave `memory[:, :, t, :]`
identical. A failing future-token invariance test is a hard stop.

Memory is regenerated from the current input row for every forward call. It is
zero at position 0 and never persists across rows, batches, validation
examples, or HellaSwag alternatives. FineWeb rows retain the baseline's packed
stream semantics: an end-of-text token inside a row does not reset any
Transformer layer's causal context and therefore does not reset teacher
feedback either. No BPTT or truncated BPTT is present in 2A0.

## Block-1 intervention

Only Block 1 Attention changes in masked modes. Restricting attention to the
diagonal makes its result exactly the current position's projected value, so
the implementation computes only the V slice of the frozen Q/K/V projection;
this is mathematically identical to computing Q/K/V and applying a one-element
softmax, while avoiding unused Q/K activations. The current token embedding and
learned position embedding remain intact. Blocks 2–12 continue to use the
unchanged full causal attention path.

The four explicit modes are:

```text
full_context
masked_l1_no_feedback
masked_l1_topdown_teacher
masked_l1_shuffled_feedback
```

`full_context` is the original Experiment 1B computation and does not require
memory. `masked_l1_no_feedback` applies only the diagonal Block-1 mask.
`masked_l1_topdown_teacher` adds real shifted memory. The shuffled control uses
one deterministic fixed-point-free permutation shared by all four sources, so
each row receives another sequence's memory; it is invalid for `B=1`.

## Top-down router and gate

The independent router uses raw shifted residuals as values, one shared learned
RMSNorm as the key transform, one learned query, no `1/sqrt(C)` factor, and a
softmax only across the four source depths:

```text
key_j = RMSNorm(memory_j)
score_j = q_topdown^T key_j
beta = softmax(score, dim=source_depth)
topdown = sum_j beta_j * memory_j
```

It is injected only into the input of Block 1 Attention:

```text
h_block1 = h_normal + tanh(g) * topdown
```

The query is initialized to zero, the RMSNorm scale to one, and scalar `g` to
zero. The new parameter count is `768 + 768 + 1 = 1,537`. Zero gate must make
the top-down mode bit-identical to the corresponding no-feedback mode. At
`g=0`, the gate can receive a gradient but query and RMSNorm gradients are
mathematically zero. Once the gate moves, the query can receive a gradient;
once the query also moves away from zero, the RMSNorm scale can receive a
gradient. The 10-update smoke must observe all three stages.

## Frozen diagnostics and comparable evaluation

Before optimization, the final checkpoint is evaluated on the same canonical
FineWeb validation prefix used by Experiment 1B: 20 serial batches with
`B=64`, `T=1024` (1,310,720 target tokens), BF16 autocast, and the mean of the
20 equal-size batch losses.

Required frozen results are:

1. `full_context` validation loss;
2. `masked_l1_no_feedback` validation loss and damage delta;
3. zero-gate teacher-feedback equality with no feedback.

The Experiment 1B source-ablation loss of 4.031627 used a smaller `B=8`
validation sample and must not be mixed with these values. Those source
ablations also measured same-token depth routing, not temporal memory or
Block-1 history.

## Single-GPU optimizer protocol (approval required)

No optimizer command may run until the user explicitly approves it.

The disposable architecture smoke uses one A100, 10 updates, `T=1024`, a small
microbatch, and a fresh optimizer. It checks finite loss, timing and memory,
the expected first-step gate-only gradient, later query/RMSNorm gradients, and
the complete absence of teacher/base gradients.

If the smoke passes, the result-bearing run starts again from the immutable
500M weights and zero-initialized feedback tensors. It processes 10 global
updates of 524,288 student tokens each, for 5,242,880 tokens. On one GPU, the
eight original Experiment 1B rank/microstep slices are replayed sequentially
per update. This preserves the original global batch and continues the exact
four-rank FineWeb stream from the saved loader states. The checkpoint's
`next_global_batch_sha256` must match before the first update.

The feedback-only AdamW optimizer keeps the original betas `(0.9, 0.95)`,
epsilon `1e-8`, and no weight decay for the two vectors and scalar gate. Its
learning rate continues the original 10B-token schedule at global update 954;
the first 2A0 result update therefore uses the schedule value for step 954.
Controls share the same frozen base, validation batches, token budget, and
evaluation code.

## Result controls and measurements

At the end of the 5M-token run, evaluate:

```text
full_context
masked_l1_no_feedback
masked_l1_topdown_teacher
masked_l1_shuffled_feedback
masked_l1_topdown_teacher with gate forced to zero
```

Then mask each of v16, v17, v20, and v24 separately in the top-down router.
Report mean per-token source weights, mean per-token routing entropy, query
norm, and gate value. Entropy is averaged before batches; it is not computed
from already-averaged weights.

For `damage > 0`:

```text
damage = L_masked_no_feedback - L_full
recovery = L_masked_no_feedback - L_real_feedback
recovery_fraction = recovery / damage
```

No claim is confirmed unless real feedback improves over masked/no-feedback
and zero/shuffled controls remove at least some of that improvement.

## Checkpoint and resume contract

Experiment 1B's four-rank optimizer and RNG state are not converted to a
one-rank optimizer resume. Only its model weights and four loader positions are
consumed. Experiment 2A0 creates a fresh checkpoint schema containing:

- full student model state and the exact parent checkpoint SHA;
- feedback-only AdamW state;
- completed 2A0 updates and processed student tokens;
- all four replay-loader states and their original geometry;
- Python, NumPy, Torch CPU, and current-device CUDA RNG states;
- the next global-batch hash;
- modes, source depths, data geometry, schedule position, Git commit/status,
  and environment metadata.

Checkpoints are written to a temporary file, reopened and verified, hashed,
then atomically published without overwriting an existing artifact. The
10-update smoke is deliberately interrupted at update 5: a fresh model,
optimizer, loader, and RNG state must reload bit-exactly at the boundary before
updates 6–10 proceed. The result-bearing run repeats this forced restart.

## Inference/cache semantics

The repository currently recomputes a complete prefix and has no KV-cache API.
Experiment 2A0 does not add one. Its intended future incremental semantics are:

- no historical K/V storage for Block 1 (current-token value only);
- normal historical K/V caches for Blocks 2–12;
- a four-vector detached teacher memory from the immediately preceding token,
  reset at each independently decoded sequence boundary. Packed training rows
  retain their existing end-of-text behavior.

Because 2A0 relies on an external full-context teacher, it is a causal
diagnostic rather than a deployable recurrent decoder. Student-generated
memory and its cache semantics belong to a later experiment.

## Compute stop points

Implementation, CPU/unit tests, checkpoint transfer, exact regression, and
frozen validation diagnostics are authorized preparation. The 10-update GPU
optimizer smoke requires explicit approval. A future 4-GPU 10–25M-token run
requires separate explicit approval and is never launched by this protocol.
