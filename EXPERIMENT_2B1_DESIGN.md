# Experiment 2B1 — Detached Self-Recurrent Reader Adaptation

## Frozen question

Can the existing 1,537-parameter top-down reader adapt to the student's own
previous-token recurrent states while every GPT-2 / Full-AttnRes base tensor,
the recurrence architecture, and the FineWeb target geometry remain fixed?

The model initializes from the verified Experiment-2A3 250M checkpoint. A new
AdamW optimizer trains only the reader query, RMSNorm scale, and scalar gate.
Teacher states and teacher losses never participate in optimizer training.

## Unchanged recurrence

- Sources are exactly v16, v17, v20, and v24.
- The destination is exactly the Block-1 attention input.
- Block 1 retains the current token and has no historical K/V.
- Blocks 2–12 retain their ordinary causal K/V prefixes.
- Every row starts at absolute position zero with zero recurrent memory and
  empty K/V caches. Packed in-row end-of-text tokens do not cause resets.
- The lower-to-higher Full-AttnRes system is unchanged.

The existing Experiment-2B0 incremental kernel is the production training
kernel. It stores `values[source].detach()` as next-token memory and copies
`k.detach()` / `v.detach()` into the historical caches. No BPTT or TBPTT is
present. The current token is not wrapped in `no_grad`: its loss differentiates
through frozen current-token operations to the three reader tensors.

## Optimization boundary

The source model state is loaded strictly and is not reinitialized. Source
teacher-memory Adam moments are ignored. The new optimizer is AdamW with
constant LR 1e-4, betas (0.9, 0.95), epsilon 1e-8, zero weight decay, and global
reader-gradient clipping at 1.0.

One result update replays the original four ranks and two accumulation steps as
eight serialized B64×T1024 slices. Each slice has fresh recurrent state. Every
one of the resulting 524,288 targets has weight 1/524,288 in the update loss.
Backward may occur after 16 independent token graphs; this changes neither the
loss definition nor the optimizer boundary.

The result run restores the four loader states and RNG state from the 2A3
checkpoint. It performs exactly 10 local updates (5,242,880 targets). Update 5
is atomically checkpointed and the process exits. A distinct invocation creates
fresh model, optimizer, and loader objects, verifies the recorded next batch,
and performs updates 6–10.

## Gates before result training

1. Verify the reader checkpoint SHA, strict model state, expected reader norms,
   2A3 data counters, and exact next-global-batch hash.
2. Preserve the 2B0 FP32/BF16 incremental equivalence, future-suffix causality,
   row isolation, fresh reset, cache policy, serialization, and short-horizon
   stability checks.
3. On an FP32 unit model, compare reader gradients from per-token backward with
   gradients from the configured chunked backward.
4. On the production model, backpropagate only a later-token loss and prove that
   retained prior-token source and K/V diagnostics receive no gradients, stored
   memory and cache tensors have no `grad_fn`, current reader gradients are
   finite/nonzero, and frozen parameters receive no gradients.
5. Reproduce the canonical 20-batch zero-shot self, masked, and gate-zero losses
   from the untouched source reader.
6. Run a disposable B2×T64 three-update smoke with a new optimizer. Save after
   update 2, exit, restore into fresh objects, verify the next batch, and run
   update 3. Discard its state.

Any gate failure prevents the result optimizer from being constructed or
stepped.

## Final evaluation

The same pinned 20 B64×T1024 validation batches evaluate full context, masked
Block 1, the source zero-shot self reader, trained self recurrence, trained
shuffled-self recurrence, trained gate-zero recurrence, and the trained reader
with detached teacher memory. Batch losses, routing, source RMS, and matched
teacher/student drift are retained.

If trained self improves on zero-shot self, run the nine reset horizons and
renormalized leave-one-source-out self-recurrent ablations. HellaSwag is not run.

## Classification rule

Let `adaptation_gain = L_zero_self - L_trained_self` and let
`specific_gap_gain` be the change in shuffled-minus-real loss from the frozen
2B0 value. Use the 0.01 loss equality band already used in 2B0.

- Any numeric, cache, gradient, replay, or checkpoint failure:
  `SELF-ADAPTATION IS UNSTABLE`.
- Adaptation gain below -0.01: `SELF-ADAPTATION DEGRADES`.
- Adaptation gain within ±0.01: `SELF-ADAPTATION IS NEUTRAL`.
- Adaptation gain above 0.01 and specific-gap gain above zero:
  `SELF-ADAPTATION IMPROVES SEQUENCE MEMORY`.
- Adaptation gain above 0.01 without positive specific-gap gain:
  `SELF-ADAPTATION IMPROVES GENERIC COMPENSATION ONLY`.

No optimizer update beyond local update 10 is authorized in this experiment.
