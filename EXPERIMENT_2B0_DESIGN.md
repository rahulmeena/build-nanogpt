# Experiment 2B0 — Zero-Shot Self-Recurrent High→Low Feedback

## Frozen question

Can the frozen Experiment-2A3 reader consume the student's own higher-layer
states from token `t-1` at Block 1 of token `t` without training, leakage, or
instability? The only experimental change is replacing the external teacher
memory with the recurrent student's memory. The four sources remain v16, v17,
v20, and v24; the only destination remains the Block-1 attention input.

## Immutable inputs

- Parent commit/tag: `75bb4e571e4356cacce76c89f352123a40254b5b` /
  `experiment-2a3-teacher-reader-250m`.
- Reader checkpoint:
  `/workspace/build-nanogpt-exp2a0/runs/experiment_2a3_250m/checkpoints/checkpoint_updates_000477.pt`.
- Required checkpoint SHA-256:
  `0702dc09c74b01eee8be504a7f5f89ca61fcc504cda8f34f30865d4ff9653d76`.
- Hardware/runtime: one A100-SXM4-80GB and
  `/workspace/venvs/exp1b/bin/python`.
- Canonical validation: the existing 20 batches at B=64, T=1024 with global
  payload hash `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

No optimizer is constructed or stepped. All model tensors must hash identically
before and after evaluation.

## Explicit recurrent state

`RecurrentState` contains:

1. an integer next-position index;
2. explicit preallocated KV buffers and logical lengths for every cached block;
3. a `[4, B, 1, 768]` bank containing the student's detached v16/v17/v20/v24
   values from the immediately preceding token; and
4. the fixed inference mode.

State is compactly serializable: only each valid KV prefix, the current memory,
position, mode, and schema are emitted. Restoring allocates fresh fixed-capacity
buffers and copies those prefixes. Modules contain no hidden mutable history.

## Attention and recurrence semantics

- Full-context equivalence mode caches Blocks 1–12 and disables feedback.
- Masked-L1 modes never allocate or retain a Block-1 cache. Block 1 uses only
  the current value projection, exactly matching the existing diagonal/self-only
  path.
- Blocks 2–12 append the current normalized attention input's K/V and attend to
  their complete prefix including the current token.
- Full AttnRes continues to route the current token's residual values with the
  frozen Experiment-1B equations at every attention, MLP, and final destination.
- Self feedback at position `t` reads only the detached memory produced at
  position `t-1`. Position zero reads an exactly zero bank. Current v16/v17/v20/
  v24 become the next state's memory only after current logits are computed.
- Shuffled self feedback applies one fixed-point-free permutation across batch
  examples to the previous memory, never across time or answer alternatives.

The zero-shot recurrence is detached. Experiment 2B0 implements neither full
BPTT nor TBPTT and does not define a training update. Any later adaptation must
be separately designed and approved.

## Boundaries and resets

Every independent sequence or validation batch starts from a freshly initialized
state: position zero, zero feedback memory, and empty caches. There is no document
metadata in the pinned contiguous validation loader, so no additional in-row
document reset is inferred. HellaSwag is excluded from this run.

The reset-horizon diagnostic resets only the four-vector feedback memory before
positions divisible by the interval. Blocks 2–12 KV caches and absolute position
continue unchanged. Interval 1 therefore removes all recurrent feedback; `never`
retains it for the whole sequence.

## Required gates

1. Short full-context and masked-L1 incremental logits/loss must be numerically
   equivalent to the existing parallel paths.
2. Future-suffix invariance, batch-row isolation, fresh-sequence reset, and exact
   serialize/resume tests must pass.
3. T=8/16/32/64 smoke runs must have finite logits, memory, feedback, and caches
   with correct lengths and no increasing non-finite state.
4. Run two fixed B=64, T=1024 validation batches for masked, teacher, and self.
5. Run all 20 canonical batches only if self feedback is finite and its two-batch
   mean loss is no greater than masked/no-feedback. Otherwise classify the
   recurrence as unstable and stop the expensive expansion.
6. Run reset-horizon diagnostics only if the canonical self loss is strictly
   below masked/no-feedback. Do not run HellaSwag without new approval.

## Classification

- **SELF-RECURRENT MEMORY TRANSFERS STRONGLY**: self recovery is at least 80% of
  teacher recovery and self is no worse than teacher by 0.05 loss.
- **SELF-RECURRENT MEMORY TRANSFERS PARTIALLY**: masked loss > self loss > teacher
  loss, without satisfying strong transfer.
- **SELF-RECURRENT MEMORY DOES NOT TRANSFER ZERO-SHOT**: self and masked differ by
  at most 0.01 loss without instability.
- **SELF-RECURRENT MEMORY IS UNSTABLE**: self loss exceeds masked loss by more
  than 0.01, or any recurrent numeric/cache invariant fails.

The 0.01 equality band prevents floating-order noise from being called transfer.
If self improves by more than 0.01 but falls outside the ordered partial-transfer
case, the report must state the exact anomaly and choose the closest conservative
classification.
