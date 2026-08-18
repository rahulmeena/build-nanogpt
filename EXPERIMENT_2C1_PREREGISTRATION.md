# Experiment 2C1 — Separated Generic and Sequence Branches at Block 9

Status: preregistered before implementation or execution on 2026-08-18.

## Scope and scientific question

Experiment 2C1 is a single-variable destination test derived from Experiment
2C0. It asks whether the exact frozen generic correction and independently
mean-centered previous-token sequence branch that failed the Block-1
specificity gate show positive sequence specificity when moved to a higher
Transformer destination.

The sole experimental destination is the **Block 9 attention input**:

- one-indexed Transformer block: `9`;
- zero-indexed implementation block: `8`;
- Full-AttnRes destination: the output of `attnres[16]`, immediately before
  Block 9 `ln_1` and attention;
- Block 9 attends only to its current token and has no historical K/V;
- Blocks 1–8 and 10–12 retain normal detached historical K/V.

Block 9 is frozen because it is a high destination, directly tests the depth
hypothesis raised by 2C0, and still leaves Blocks 9–12 to integrate the injected
signal. No other destination will be evaluated in 2C1.

## Frozen lineage

- parent branch: `experiment-2c0-separated-generic-sequence-b1`;
- parent/final 2C0 commit:
  `677d711bc00dba0da1b80cb6369f33841ec29a51`;
- 2C1 branch: `codex/experiment-2c1-destination-b9`;
- frozen model/source checkpoint: final Experiment 2B3 joint-update-9
  checkpoint, SHA256
  `7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b`;
- sequence-reader initialization checkpoint: final Experiment 2B1 update-10
  checkpoint, SHA256
  `5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a`;
- source depths: `v16`, `v17`, `v20`, and `v24`;
- canonical validation aggregate SHA256:
  `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`;
- training cursor/next-global-batch SHA256:
  `7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7`.

The 2C0 artifacts are inputs, not recalibrated outputs:

- `results/experiment_2c0_separated_generic_sequence_b1/generic_correction.pt`;
  file SHA256
  `a7db6a7c4007b516f0c74448508087408df15ed3e12ae55886e031b6b4dc141b`,
  tensor SHA256
  `cc169dc9b2cddd88657f6e88edffbdd01ccf50edd3279a94619d1b02216a703d`;
- `results/experiment_2c0_separated_generic_sequence_b1/sequence_source_means.pt`;
  file SHA256
  `144e04993a1afaa0d2a05b08ad27e1d3adbeb645168787ac4fa7be11dca8fbd8`,
  tensor SHA256
  `7f988159ce3fdb89242b18226c10957eb2cefa6b8d8621a6c58d39fce548b992`.

## Frozen architecture

For token `t=0`, both branches inject exactly zero. For every later token, the
Block-9 attention input receives:

```text
fixed_generic_G
+
tanh(sequence_gate) * SequenceReader(
    FP32(raw_previous_token_sources - frozen_source_means)
)
```

The FP32 centered bank is cast once to the reader input dtype. The same coherent
row supplies all four source depths in both aligned and shuffled controls.

The sequence reader is an independent copy of the 2C0 initialization:

- copied 2B1 query and RMSNorm scale;
- gate initialized with `atanh(0.5 * tanh(old_gate))`;
- exactly 1,537 trainable parameters (`768 + 768 + 1`);
- all base-model, old-reader, and writer parameters frozen;
- old dynamic reader bypassed;
- writer adapters never active;
- raw previous-token source memory detached;
- historical K/V detached;
- no BPTT, temporal gradient, auxiliary loss, or recurrent writer gradient.

No source, initialization, mask count, feedback scale, writer, objective, data
stream, optimizer, or training-budget change is allowed. The only scientific
change from 2C0 is moving the single masked/injected attention destination from
Block 1 to Block 9.

## Zero-shot controls

Evaluate the exact canonical `20 × B64 × T1024` validation prefix in BF16 with
the initialized reader and no optimizer. Save all paired batch losses for:

1. `generic`: frozen `G` only;
2. `real`: frozen `G` plus aligned centered previous-token sources;
3. `shuffle`: frozen `G` plus a deterministic coherent fixed-point-free row
   permutation of centered previous-token sources;
4. `sequence_only`: aligned sequence branch without `G`;
5. `gate_zero`: frozen `G` with sequence contribution forced to zero.

The primary preregistered metric is:

```text
specific_gap_0 = mean(L_shuffle - L_real)
```

Positive values mean correct sequence identity is useful. Report mean, median,
sample standard deviation, minimum, maximum, real wins, shuffled wins, and ties
over the 20 paired batches.

## Mandatory pre-training integrity

Before any optimizer construction, require all of the following:

- exact parent, model, initialization, artifact, validation, and tensor SHAs;
- exactly one missing historical cache: Block 9;
- Blocks 1–8 and 10–12 cache lengths equal the current recurrent position;
- current-token Block-9 attention path retained;
- direct feedback applied only at Block 9 attention input;
- exactly 1,537 trainable sequence-reader parameters;
- no base, old-reader, or writer gradients;
- writers never active;
- frozen `G` and frozen source means bit-identical before/after evaluation;
- frozen model bit-identical before/after evaluation;
- FP32 mean-centering before the single cast;
- future-suffix causality passes bit-exactly;
- row isolation passes bit-exactly;
- fresh-sequence state has zero memory, Block-9 cache absent, and every other
  cache present at length zero;
- zero centered sources produce bitwise-zero top-down and sequence feedback;
- `gate_zero` and `generic` agree within deterministic BF16 tolerance;
- all states and losses finite;
- HellaSwag not run.

Any integrity failure is a hard stop and classifies the experiment as unstable;
it does not authorize repair by changing scientific semantics.

## Frozen zero-shot training gate

Constructing an optimizer, calling backward, or consuming training targets is
authorized only if **all** conditions pass:

```text
specific_gap_0 > 0
real wins >= 12 / 20
L_real <= L_generic + 0.10
all mandatory integrity checks pass
```

If any condition fails, stop 2C1 before training. Optimizer constructions,
optimizer updates, backward calls, parameter updates, and additional training
targets must all remain exactly zero. Finalize and audit the zero-shot result;
do not weaken the gate or try another destination.

## Training protocol if and only if the gate passes

Preserve the complete 2C0 training protocol:

- four A100-SXM4-80GB GPUs with deterministic four-rank DDP;
- identical replay-loader states and exact next-batch hash lineage;
- two `B64 × T1024` microsteps per rank per update;
- 524,288 global next-token targets per update;
- exactly 10 sequence-reader updates / 5,242,880 targets;
- fresh AdamW over only the 1,537 reader parameters;
- constant learning rate `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero
  weight decay, and global gradient clip `1.0`;
- aligned next-token CE only;
- deterministic fixed rank-slot FP32 gradient reduction;
- disposable smoke test and 1-GPU-to-4-GPU migration equivalence before the
  result run;
- atomic strict-reopen checkpoint after update 5, mandatory process exit, then
  fresh-process resume to update 10;
- no continuation beyond update 10.

Smoke or migration-equivalence failure is a hard stop. A NaN/Inf, replay hash
mismatch, cross-rank reader/optimizer mismatch, checkpoint mismatch, causality
failure, cache-policy failure, or gradient leak is also a hard stop.

## Final evaluation and classification if training occurs

Re-evaluate the canonical 20 batches with the unchanged initial reader and the
trained reader. Preserve the 2C0 final controls: generic, initial aligned,
initial shuffled, trained aligned, trained shuffled, trained gate-zero, trained
sequence-only, leave-one-out batch-mean sequence, and calibration-mean
sequence. Save every paired loss and reader/router diagnostic.

Use the frozen strong direct sequence-memory criterion:

```text
training_real_gain >= 0.010
final_specific_gap >= 0.020
final real wins >= 18 / 20
final batchmean_gap >= 0.010
all integrity checks pass
```

Passing all five conditions is the only result that authorizes recommending
Block 9 for a later writer experiment. Writers are not implemented or run in
2C1. Stable threshold misses must be classified conservatively as generic-only,
neutral, degrading, or mixed according to the recorded raw metrics.

## Absolute stopping rules

- Do not rerun 2C0.
- Do not evaluate any destination other than Block 9.
- Do not recalibrate `G` or the source means.
- Do not add writers, BPTT, temporal gradients, mask-depth variants, auxiliary
  losses, multiple destinations, iterative loops, or HellaSwag.
- Stop before training on a failed zero-shot gate.
- Stop after exactly 10 updates if the gate passes.
- Do not launch any follow-up experiment without explicit approval.

