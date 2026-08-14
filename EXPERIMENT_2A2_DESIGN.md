# Experiment 2A2 — Teacher-Feedback Memory Reader Scaling, 25M to 100M

## Frozen objective and lineage

Experiment 2A2 changes no model architecture. It continues the exact feedback
reader, optimizer, RNG, and four FineWeb replay streams from the verified final
Experiment 2A1 checkpoint:

```text
checkpoint: /workspace/build-nanogpt-exp2a0/runs/experiment_2a1_25m/checkpoints/checkpoint_updates_000048.pt
SHA256: d821b48a796b12bb489f5bc9bc1791c475c09a50de7d5b47c4a36cf766643ec2
completed Experiment-2 updates: 48
student tokens: 25,165,824
next global-batch SHA256: 1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5
```

The immutable root parent remains Experiment 1B Full AttnRes checkpoint SHA256
`6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`.
Before optimizer access, checkpoint bytes, sidecars, complete student state,
feedback-only AdamW state, four loaders, Python/NumPy/Torch CPU/Torch CUDA RNG,
lineage, and next-batch hash must restore exactly.

## Architecture and optimization boundary

- Block-1 historical attention remains removed; the current-token embedding is retained.
- Blocks 2–12 retain normal full causal attention.
- The frozen full-context teacher supplies detached, one-token-shifted raw
  sources `v16`, `v17`, `v20`, and `v24`.
- Feedback enters only the Block-1 Attention input.
- The teacher is always frozen, `eval()`, and under `torch.no_grad()`.
- Every existing GPT-2 / Full-AttnRes student tensor remains frozen and
  bit-identical.
- Exactly 1,537 feedback-reader scalars train: query 768, RMSNorm scale 768,
  and one scalar gate.

No source, destination, recurrence, gate, slot, loss, distillation target,
mask, positional encoding, unfreezing, or RL mechanism is added.

## Cumulative geometry, data, and schedule

```text
source completed updates: 48
target completed updates: 191
additional updates:       143
tokens per update:         524,288
additional tokens:         74,973,184
final cumulative tokens:   100,139,008
metric update indices:     48 through 190
global schedule steps:     1002 through 1144
```

Each optimizer update serializes the original four-rank Experiment 1B global
batch: four B=64, T=1024 replay streams and gradient accumulation 2, for eight
ordered microbatches. The LR schedule and feedback AdamW moments continue
without reset or rescaling.

The 144 boundary hashes for updates 48 through 191 are derived read-only from
the source loader states and pinned by sequence digest
`3ea6e3a4833f14f57df109bfc9fca01798b5f1aaeb9b8d6b1fbc6035dd92d604`.
Every actual update hash must equal the corresponding oracle hash before its
optimizer step.

## Milestones and recovery

| Completed updates | Tokens | Last schedule step | Evaluation |
|---:|---:|---:|---|
| 48 | 25,165,824 | 1001 | continuity verification only |
| 96 | 50,331,648 | 1049 | canonical real and shuffled |
| 191 | 100,139,008 | 1144 | full controls, ablations, HellaSwag |

Complete atomic checkpoints are saved at updates 96 and 191. Each is
immediately reopened and checked for strict model, optimizer, loader, RNG,
lineage, and next-batch equality. A fresh-object restart is forced at update 96.
Resume accepts only a checkpoint bound to the same run directory and validates
the complete prior checkpoint/evaluation/restart/causality chain before any
new optimizer step.

## Canonical FineWeb validation

All milestones use the pinned 20 batches at B=64, T=1024 and validation digest
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.
At update 96 the measured controls are real and shuffled feedback. At update
191 the runner also measures full context, masked/no feedback, forced-zero
gate, and renormalized leave-one-source-out removal of v16/v17/v20/v24.

The audited 5M/10M/15M/25M trajectory is imported unchanged from the committed
2A1 result. The update-48 evaluation is repeated only as a checkpoint-bound
continuity test and must reproduce the audited real and shuffled losses exactly.
All 20 raw paired losses are stored.

## HellaSwag

The complete upstream validation split is pinned to 10,042 examples and file
SHA256 `0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588`.
At update 191 it evaluates normalized completion accuracy for:

- the full-context Full AttnRes teacher/base;
- masked Layer-1 with no feedback;
- real teacher feedback;
- the trained reader with gate forced to zero.

Each of the four answer candidates is an independent batch row. Teacher memory
is computed only from that candidate, shifted only within that candidate, and
discarded after the example. A preflight test changes the other candidates and
requires candidate-zero teacher memory and logits to remain bit-exact; it also
interposes another example and requires exact reset on repetition.

Shuffled HellaSwag is preregistered as skipped. The existing scientific shuffle
permutes batch rows; on HellaSwag those rows are answer alternatives, so using
it would exchange memory between candidates and violate strict alternative
isolation. Scoring semantics are not changed to manufacture this control.

Historical anchors are 2568/10042 (25.573%) for Standard GPT-2 at 500M and
2532/10042 (25.214%) for Full AttnRes at 500M. The current full-context result
must reproduce 2532/10042. These are not equal-token pretraining comparisons:
the 2A2 system is a frozen 500M base plus a reader trained on additional data.

## Safety invariants

At the source and every milestone, teacher-memory and end-to-end suffix
perturbation tests must show no future-token leakage. Every update must record
finite loss, all three finite feedback gradients, finite Adam moments, exactly
1,537 trainables, no teacher/base gradients, unchanged frozen-state hashes,
the correct cumulative tokens/schedule/LR, and the pinned replay hash. Any
mismatch is a hard stop; the runner must not repair semantics automatically.

## Frozen interpretation rules

Let `q25`, `q50`, and `q100` be shuffled-minus-real loss recovery:

- `MEMORY SIGNAL DISAPPEARING` if `q100 <= 0` or `q100 < 0.5*q25`;
- `MEMORY SIGNAL SATURATING` if `q100 > 0` and the 50M→100M gain is at most
  25% of `max(q50-q25, 1e-12)`;
- `MEMORY SIGNAL ACCELERATING` if `q100 > q50 > q25` and the 50M→100M gain
  exceeds the 25M→50M gain;
- otherwise `MEMORY SIGNAL STRENGTHENING`.

Continuing to 250M is justified only if all safety/control checks pass,
`q100 > 0`, real beats shuffled on at least 15/20 canonical batches,
`q100 >= q50`, total recovery remains positive, and real-feedback HellaSwag is
not more than one percentage point below both masked/no-feedback and gate-zero.

The hard stop is completed update 191. No 250M continuation, recurrence,
unfreezing, or follow-on experiment is launched automatically.
