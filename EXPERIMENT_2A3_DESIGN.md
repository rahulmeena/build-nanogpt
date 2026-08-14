# Experiment 2A3 — Teacher-Feedback Memory Reader Scaling: 100M → 250M

## Frozen question

Continue the exact Experiment 2A2 teacher-assisted reader from 100,139,008 to
250,085,376 Experiment-2 student tokens. Measure whether aligned high-layer
history keeps improving relative to shuffled history. No architecture, loss,
data-order, optimizer, or frozen-weight change is permitted.

## Immutable continuation

- Source state: 191 completed Experiment-2 updates / 100,139,008 tokens.
- Source checkpoint SHA-256: `6c206a89422470061d7997764fbd9a5708be3d9043f8fab930dd4b800bd5cb95`.
- Source next-global-batch SHA-256: `9f39510b105f068966ef6c052edc015d695827c422da37495fa7c244b965af0b`.
- Restore the student, feedback-only AdamW, four loaders, all RNG states, and
  lineage exactly. The first new schedule step is 1145.
- Train updates 191…476, reaching completed update 477 and 250,085,376 tokens.
- Save/evaluate at completed updates 286, 381, and 477. Fresh-object exact
  restarts are mandatory after 286 and 381.
- The committed update-191 evaluation is validated and reused, never rerun.

## Frozen mechanism

The Experiment 1B Full-AttnRes base and full-context teacher are immutable.
Block-1 historical self-attention is masked while the current embedding remains.
Detached, one-token-shifted teacher sources v16, v17, v20, and v24 are routed only
to the Block-1 attention input. Exactly the existing query (768), RMSNorm (768),
and scalar gate (1) are trainable: 1,537 parameters total.

## Geometry and evaluation

Each optimizer update replays the original four-rank B=64, T=1024,
gradient-accumulation=2 global batch as eight serialized one-GPU microbatches,
524,288 student tokens per update. Every actual batch hash must match the oracle.

At 150M, 200M, and 250M, evaluate real and fixed-derangement shuffled feedback
on the canonical 20×B64×T1024 validation prefix. At 250M also evaluate full
context, masked/no-feedback, gate-zero, and renormalized leave-one-source-out
v16/v17/v20/v24 controls. Save all paired batch losses. At 250M run all 10,042
HellaSwag validation examples for full, masked, real, and zero modes. Shuffled
HellaSwag remains skipped because candidate-row permutation would exchange answer
alternatives and violate candidate isolation.

## Predeclared interpretation

Let `q100,q150,q200,q250 = L_shuffled − L_real` and let the three successive
gains be `d1,d2,d3`.

- `MEMORY SIGNAL REVERSING` if q250 ≤ 0 or d3 < 0.
- `MEMORY SIGNAL SATURATING` if d3 ≥ 0 and d3 ≤ 25% of max(d1,d2,1e-12).
- `MEMORY SIGNAL STILL ACCELERATING` if q250>q200>q150>q100 and d3>d2>d1>0.
- Otherwise `MEMORY SIGNAL STRENGTHENING`.

Continue to 500M only if all invariants pass, q250 is positive and nondecreasing,
real wins at least 15/20 batches, total recovery is positive, and real HellaSwag
is no more than one percentage point below both masked and zero controls.

Begin a self-generated recurrence experiment only if the same invariants and
paired conditions pass, sequence-specific recovery is at least 10% of original
damage, every final source-ablation delta is positive, and real HellaSwag is at
least masked/no-feedback. This decision authorizes only a separately designed
experiment; it does not close the recurrent loop here.

## Hard stops

Abort on checkpoint, loader, RNG, replay, schedule, causality, finite-numeric,
trainability, frozen-base, teacher-gradient, optimizer, evaluation-isolation, or
candidate-isolation mismatch. Stop exactly at completed update 477. Do not launch
500M training, student recurrence, new sources/destinations, or any follow-on.
