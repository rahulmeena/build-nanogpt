# Experiment 2C0 — Separated Generic and Sequence Branches at Block 1

Experiment 2C0 freezes the final Experiment 2B3/2B5 generic correction as one
sequence-independent 768-vector and trains a separate 1,537-parameter reader on
centered raw previous-token source states. The active recurrent path contains no
writer adapters and Block 1 retains no historical K/V. Blocks 2–12 preserve
their normal detached historical K/V.

The generic vector is constructed once from the frozen final-2B5 source means,
the final-2B3 reader, and its gate. New source means are calibrated under the
generic-only recurrence on the exact disjoint 2B4/2B5 calibration set. The
sequence reader receives only `raw_source - calibrated_mean`, computed in FP32.

Only the independent sequence query, RMSNorm scale, and scalar gate train. The
query and norm are copied from the final 2B1 direct self-memory reader. Its gate
is initialized with `atanh(0.5 * tanh(old_gate))`. A fresh AdamW optimizer uses
constant `1e-4` learning rate, `(0.9, 0.95)` betas, zero weight decay, and global
gradient clipping at 1.0.

Result training is gated on the frozen generic regression, positive zero-shot
aligned-vs-shuffled specificity, at least 12/20 paired wins, and all causality,
row-isolation, replay, and distributed-equivalence checks. The result budget is
exactly ten 524,288-target updates with a forced fresh-process restart after
update five. No HellaSwag, writer, auxiliary loss, BPTT, destination-depth, or
mask-depth experiment is part of 2C0.
