# Experiment 2B2 — Memory-Only Writers with One-Step Temporal Credit

## Scientific objective

Experiments 2B0 and 2B1 established:

1. Student-generated recurrent high→low memory works zero-shot.
2. It remains stable through the full 1024-token context.
3. Longer memory persistence monotonically improves validation loss.
4. Reader-only adaptation produces only a small improvement and is formally neutral.
5. The self-memory reader becomes increasingly concentrated on `v16`, suggesting that reader capacity is no longer the main bottleneck.

Final Experiment 2B1 canonical values:

```text
Full context:
4.0786544085

Masked Block-1 / no feedback:
5.9736744881

Zero-shot self:
5.7074206829

2B1 trained self:
5.7013087273

2B1 trained shuffled self:
5.7310302973

2B1 gate-zero:
5.9736480713

2B1 reader + teacher memory:
5.5720659256
```

Derived 2B1 values:

```text
self adaptation gain:
0.0061119556

trained shuffled-minus-real:
0.0297215700

real wins vs shuffled:
20 / 20

trained self recovery / masking damage:
14.3727%
```

Experiment 2B2 asks:

> Can the model learn WHAT representation should be written into its recurrent memory so that the next token can use it more effectively?

This is the first experiment with explicit temporal credit assignment.

However, it must remain tightly controlled.

We will NOT unfreeze any GPT-2 or Full-AttnRes parameter.

Instead, add small memory-only writer adapters to the existing four recurrent memory sources.

The adapters affect ONLY the recurrent-memory branch.

They must NOT alter the normal same-token Transformer computation.

---

# 1. Freeze Experiment 2B1

Current completed branch:

```text
experiment-2b1-self-reader-adaptation
```

Final audited results commit:

```text
ad2eb56b1fdbf20dde515693f6d2c9bd9034f444
```

Verify:

```bash
git status
git rev-parse HEAD
git show --stat ad2eb56b1fdbf20dde515693f6d2c9bd9034f444
```

Working tree must be clean.

Create and push an immutable tag:

```text
experiment-2b1-self-reader-5m
```

pointing exactly to:

```text
ad2eb56b1fdbf20dde515693f6d2c9bd9034f444
```

Then create a new branch:

```text
experiment-2b2-memory-writers-1step
```

from that tag.

Record before modification:

```text
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
git status
```

Do not modify the frozen 2B0 or 2B1 branches/tags.

---

# 2. Starting model

Use the FINAL Experiment 2B1 model weights.

Do not initialize from the older 2A3 reader if a verified 2B1 result checkpoint exists.

Resolve the exact final 2B1 checkpoint path from the audited Experiment 2B1 artifacts.

Do NOT guess its path.

Expected final checkpoint SHA-256:

```text
5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a
```

Verify this SHA before loading.

Expected reader values:

```text
gate:
0.1594133824

tanh(gate):
0.1580765992

query norm:
1.9570868015

RMSNorm displacement:
1.9478669167
```

Expected self-memory router behavior:

```text
v16:
0.6340336651

v17:
0.0206797247

v20:
0.1305237375

v24:
0.2147629216

entropy:
0.4845177144

top-down RMS:
0.1623093784

feedback RMS:
0.0256773381
```

All existing parameters from this checkpoint must initially load exactly.

---

# 3. Freeze the existing reader

Unlike 2B1, the existing top-down reader is NOT trainable in 2B2.

Freeze:

```text
top-down query
top-down RMSNorm scale
top-down scalar gate
```

Also freeze:

```text
entire GPT-2 base
all Full-AttnRes parameters
token embeddings
position embeddings
all attention blocks
all MLP blocks
all layer norms
all lower→higher routers
```

The 1,537 reader parameters remain part of the model but:

```text
requires_grad = False
```

They must remain bit-identical throughout Experiment 2B2.

---

# 4. Existing recurrent architecture remains unchanged

Maintain the verified Experiment 2B0/2B1 inference semantics.

## Block 1

```text
historical K/V:
NONE

current token:
YES

high→low recurrent feedback:
YES
```

## Blocks 2–12

```text
normal historical K/V caches:
YES
```

## Memory sources

Exactly:

```text
v16 = Block 8 MLP
v17 = Block 9 Attention
v20 = Block 10 MLP
v24 = Block 12 MLP
```

## Reader destination

Exactly:

```text
Block 1 Attention input
```

Do not change any source or destination.

---

# 5. Add four MEMORY-ONLY writer adapters

Create one independent writer adapter for each recurrent source:

```text
writer_v16
writer_v17
writer_v20
writer_v24
```

Use rank:

```text
r = 8
```

For each raw source state:

```text
v_j(t)
```

first detach it from the ordinary Transformer computation:

```python
source = v_j.detach()
```

Then use a parameter-free RMS normalization:

```text
u = source / sqrt(mean(source^2) + eps)
```

with:

```text
eps = 1e-5
```

No learned scale or bias.

Then compute:

```text
z = W_down_j(u)

delta = W_up_j(silu(z))

memory_j = source + delta
```

Dimensions:

```text
W_down:
768 → 8
bias = False

W_up:
8 → 768
bias = False
```

Do not add any other parameters.

---

# 6. Exact writer parameter count

Per source:

```text
W_down:
768 × 8 = 6,144

W_up:
8 × 768 = 6,144

total/source:
12,288
```

Four sources:

```text
4 × 12,288
=
49,152 trainable parameters
```

Require exactly:

```text
trainable parameters = 49,152
```

No other trainable tensor is permitted.

---

# 7. Zero-effect initialization

The writer architecture must initialize as an EXACT functional identity.

Initialize:

```text
W_down:
deterministic Normal(mean=0, std=0.02)

W_up:
exactly zero
```

Use an explicit fixed initialization seed and record it.

Because:

```text
W_up = 0
```

initially:

```text
delta = 0

memory_j = source
```

Therefore the untrained 2B2 architecture must reproduce the frozen 2B1 recurrent result.

This is mandatory.

Do NOT initialize both matrices to zero because that would block all learning.

Expected gradient staging:

```text
first optimizer update:

W_up gradients:
nonzero

W_down gradients:
exactly zero or numerical zero
```

Once `W_up` moves:

```text
later updates:

W_down gradients:
nonzero
```

Add an explicit test for this staging.

---

# 8. Crucial separation: writers affect memory only

The ordinary same-token Transformer forward path must continue using:

```text
raw v16
raw v17
raw v20
raw v24
```

NOT their adapted writer outputs.

The adapted states:

```text
memory_v16
memory_v17
memory_v20
memory_v24
```

exist only in the recurrent high→low state passed to the NEXT token.

Conceptually:

```text
                ordinary model path
v16(t) ─────────────────────────────→ later layer computation
   │
   │ detach
   ▼
writer_v16
   │
   ▼
memory_v16(t)
   │
   │ one-token delay
   ▼
Block 1(t+1)
```

This distinction is fundamental.

The writers must not improve same-token prediction through a conventional residual path.

---

# 9. One-step temporal credit assignment

Experiment 2B2 introduces exactly ONE token of temporal gradient credit.

At token `t`:

```text
source v_j(t)
   ↓ DETACH FROM BASE
writer_j
   ↓
memory_j(t)
```

The writer-produced memory is NOT detached before token `t+1`.

At token `t+1`:

```text
memory(t)
   ↓
frozen Block-1 reader
   ↓
frozen Blocks 1–12
   ↓
loss(t+1)
```

Gradient from:

```text
loss(t+1)
```

must be allowed to reach:

```text
writer(t)
```

through the recurrent memory.

Therefore:

```text
loss(t+1)
→ Block 1(t+1)
→ memory(t)
→ writer(t)
```

is permitted.

---

# 10. But temporal gradient MUST stop after one transition

Experiment 2B2 does NOT permit multi-token BPTT.

The key mechanism is:

```text
v_j(t+1)
```

is detached before entering:

```text
writer_j(t+1)
```

Therefore although the FORWARD recurrent influence can propagate indefinitely:

```text
memory(t)
→ token t+1
→ high(t+1)
→ memory(t+1)
→ token t+2
→ ...
```

the BACKWARD graph stops at every writer input.

So:

```text
loss(t+2)
```

may train:

```text
writer(t+1)
```

but MUST NOT backpropagate through:

```text
high(t+1)
→ memory(t)
→ writer(t)
```

This gives an exact one-token temporal credit horizon.

---

# 11. Historical KV caches remain detached

Blocks 2–12 maintain their normal forward KV history.

However, stored K/V tensors must continue to be detached between timesteps:

```python
stored_k = current_k.detach()
stored_v = current_v.detach()
```

This prevents temporal gradients through ordinary Transformer history.

Block 1 continues to have no historical KV cache.

---

# 12. Explicit one-step-gradient unit test

Implement a dedicated 3-token or 4-token FP32 diagnostic.

Retain gradients on diagnostic writer memories:

```text
m0
m1
m2
```

Verify:

## Loss at token 1

```text
d loss1 / d m0 != 0
```

## Loss at token 2

```text
d loss2 / d m1 != 0
```

but:

```text
d loss2 / d m0 == 0 / None
```

because `v1` was detached before writer1.

Similarly:

```text
loss3 → m2
```

must exist while:

```text
loss3 → m1
```

must not exist through the recurrent chain.

Also verify no temporal gradient through historical K/V.

This test is a HARD prerequisite before GPU training.

---

# 13. Base and reader gradient isolation

During writer training require:

```text
writer gradients:
present

reader gradients:
none

base-model gradients:
none

lower→higher Full-AttnRes gradients:
none
```

Additionally require frozen parameter hashes before/after:

```text
GPT-2 / Full-AttnRes base hash:
unchanged

2B1 reader hash:
unchanged
```

---

# 14. Teacher is NOT used in training

Training uses only:

```text
STUDENT self-generated high states
```

The teacher may be used only for post-training evaluation.

Record:

```text
teacher forward calls during optimizer training = 0
```

---

# 15. Starting data position

Start from the exact NEXT FineWeb batch after Experiment 2B1.

Resolve the final 2B1 replay-loader states from its final verified checkpoint.

Expected final 2B1 next-global-batch SHA:

```text
3c4d0cd7905f16bfcfdd283cbc0799ff9f85c91ac3c521e886f0e403fc11ae57
```

Verify this before the first result-bearing update.

Do not restart the dataset.

Do not reuse the 2A3 starting data position.

Do not use any batch consumed by 2B1.

---

# 16. Optimizer

Create a NEW optimizer because the writer parameters are new.

Use AdamW over exactly 49,152 writer parameters:

```text
lr = 1.0e-4

betas = (0.9, 0.95)

eps = 1e-8

weight_decay = 0.0

gradient clipping = 1.0
```

Use a constant learning rate for this first experiment.

No LR warmup.

No reuse of the 2B1 reader optimizer.

No optimizer state for frozen parameters.

---

# 17. Hardware

Use:

```text
1 × NVIDIA A100-SXM4-80GB
```

No DDP.

---

# 18. Global training geometry

Preserve the established FineWeb global batch:

```text
524,288 target tokens / optimizer update
```

using the existing one-GPU serialization of the original four-rank stream.

Equivalent geometry:

```text
8 serialized B64 × T1024 slices
per global optimizer update
```

Do not alter data order.

Each B64 row begins with:

```text
position = 0
memory = zero
KV caches = empty
```

Do NOT reset on an in-row end-of-text token.

---

# 19. Loss semantics

Keep the exact causal language-model objective.

All valid 1024 targets per row contribute to CE.

The first position receives zero recurrent memory.

The final writer output in a row has no next in-row token from which to receive temporal writer credit; this is expected.

Do not modify token weighting to compensate.

---

# 20. Efficient backward implementation

Because the writer graph spans only one recurrent transition, do not retain a 1024-token autograd graph.

The preferred implementation is a streaming/pipelined backward.

Conceptually:

```text
token t:
  use memory(t−1)
  compute loss(t)
  compute new writer memory(t)

  backward loss(t)
    → writer from t−1

  retain only memory(t) graph
  detach stored KV
  continue
```

Ensure backward on the current loss does not destroy the graph required by the newly produced `memory(t)`.

Because writer input is detached from the base, `memory(t)` should own only the small writer graph necessary for the next token.

Add explicit tests.

Do not use `retain_graph=True` across the full sequence unless absolutely necessary.

---

# 21. Pre-training identity regression

Before constructing the result-bearing optimizer:

initialize writers with:

```text
W_up = 0
```

and evaluate the canonical self-recurrent model.

The 2B2 untrained writer model should reproduce Experiment 2B1:

```text
5.7013087273
```

within only the already-characterized BF16 incremental tolerance.

Also reproduce:

```text
masked:
5.9736744881

gate-zero:
5.9736480713
```

Require writer-disabled and zero-initialized-writer modes to agree.

If identity initialization does not reproduce the 2B1 model, STOP.

---

# 22. CPU/FP32 mathematical regression

Build a small deterministic FP32 test model.

Verify:

```text
zero writer:
exactly reproduces raw-memory recurrence
```

and verify one-step writer gradients against a reference explicit computation.

For writer parameters report:

```text
maximum absolute gradient difference
maximum relative gradient difference
```

Use strict FP32 tolerances.

---

# 23. Disposable GPU smoke

Before the full 5M run, perform a disposable smoke.

Use:

```text
1 × A100
B = 2
T = 64
3 optimizer updates
```

with the same writer optimizer configuration.

Verify:

```text
trainable parameter count = 49,152

update 1:
W_up gradient nonzero
W_down gradient zero/expected-zero

update 2+:
W_up gradient nonzero
W_down gradient nonzero

reader gradients = none
base gradients = none
teacher calls = 0

loss finite
gradients finite
optimizer moments finite

recurrent state finite
KV finite

one-step temporal-gradient test still passes
```

After smoke update 2:

1. save checkpoint;
2. terminate process;
3. load in a fresh process;
4. execute update 3;
5. verify next-batch identity.

Then discard all smoke changes.

The result-bearing run must restart from the original 2B1 model and untouched next-data position.

---

# 24. Result-bearing 5M run

If and only if all preflight and smoke tests pass:

initialize from:

```text
frozen 2B1 model weights
+
fresh zero-effect writer adapters
+
fresh writer AdamW
+
exact next data position after 2B1
```

Run exactly:

```text
10 global optimizer updates
```

at:

```text
524,288 target tokens/update
```

for:

```text
5,242,880 total 2B2 training tokens
```

Hard stop after update 10.

Do NOT automatically continue to 25M.

---

# 25. Forced exact restart

After update 5:

```text
2,621,440 2B2 tokens
```

save a complete checkpoint.

Terminate the process.

Start a fresh process.

Restore:

```text
writer parameters
writer AdamW moments
update count
processed 2B2 tokens
four replay loaders
Python RNG
NumPy RNG
Torch CPU RNG
Torch CUDA RNG
next-global-batch hash
model lineage
Git commit
configuration
```

Verify update 6 consumes the exact recorded next batch.

---

# 26. Per-update measurements

For every global update record:

```text
update
processed 2B2 tokens
training loss
learning rate

pre-clip gradient norm
post-clip gradient norm

W_up aggregate norm
W_down aggregate norm

writer delta RMS:
v16
v17
v20
v24

raw source RMS:
v16
v17
v20
v24

adapted memory RMS:
v16
v17
v20
v24

reader routing weights:
v16
v17
v20
v24

reader routing entropy

top-down RMS
feedback RMS

peak allocated VRAM
peak reserved VRAM
wall time
```

Also record maximum writer residual-to-source RMS ratio.

We want to detect writers becoming excessively large.

---

# 27. Canonical final validation

After update 10 use exactly:

```text
20 × B64 × T1024
BF16

validation hash:
3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb
```

Evaluate:

```text
A. full context

B. masked L1 / no feedback

C. frozen 2B1 self recurrence
   (writers bypassed)

D. trained 2B2 writer self recurrence

E. trained 2B2 writer shuffled recurrence

F. trained 2B2 writers disabled/bypassed

G. trained 2B2 writer model with recurrent gate forced zero
```

Also evaluate:

```text
H. trained writer adapters fed with teacher source states
```

only as a diagnostic of writer distribution compatibility.

Teacher remains evaluation-only.

---

# 28. Primary metrics

Let:

```text
L_masked
L_2b1
L_writer_real
L_writer_shuffled
```

Use the fixed 2B1 self baseline:

```text
L_2b1 = 5.7013087273
```

Calculate:

```text
writer_gain =
L_2b1 - L_writer_real
```

Sequence-specific writer gap:

```text
writer_specific_gap =
L_writer_shuffled - L_writer_real
```

2B1 baseline specific gap:

```text
0.0297215700
```

Therefore:

```text
specific_gap_gain =
writer_specific_gap
- 0.0297215700
```

Also calculate:

```text
writer_recovery =
L_masked - L_writer_real
```

```text
writer_recovery_fraction =
writer_recovery /
(L_masked - L_full_context)
```

and compare to:

```text
2B0 zero-shot recovery:
14.0502%

2B1 recovery:
14.3727%

teacher-reader recovery:
21.3001%
```

---

# 29. Paired real-vs-shuffled analysis

Save all 20 validation batch losses for:

```text
real trained-writer memory
shuffled trained-writer memory
```

Report:

```text
real wins
shuffled wins
ties

mean shuffled-real
median
sample std
minimum
maximum
```

2B1 reference:

```text
real wins:
20/20

mean shuffled-real:
0.0297215700
```

---

# 30. Memory-writer contribution analysis

Measure per source:

```text
RMS(delta_j)
RMS(source_j)
RMS(delta_j) / RMS(source_j)

cosine(source_j, adapted_memory_j)
```

at validation time.

Do this for:

```text
v16
v17
v20
v24
```

We want to know whether useful writer learning requires:

```text
small corrective changes
```

or large transformations.

---

# 31. Writer ablation

If trained writers improve real-self validation by at least:

```text
0.01 loss
```

perform writer-specific ablations.

For each:

```text
disable writer_v16 residual only
disable writer_v17 residual only
disable writer_v20 residual only
disable writer_v24 residual only
```

while retaining the raw source residual path:

```text
memory_j = source_j
```

for the disabled writer.

Report:

```text
baseline trained-writer loss
ablated loss
delta
positive batches / 20
```

This differs from deleting the whole memory source.

It specifically measures the learned writer transformation.

---

# 32. Existing source ablation

If writer learning is positive, optionally also repeat the existing renormalized source-removal ablation:

```text
remove v16
remove v17
remove v20
remove v24
```

Keep these two ablation types clearly separated:

```text
writer-residual ablation
vs
whole-source ablation
```

---

# 33. Reset-horizon evaluation

If:

```text
writer_gain > 0
```

run the recurrent-memory reset sweep:

```text
1
2
4
8
16
32
64
128
never
```

Reset only recurrent high→low memory.

Do NOT reset Blocks 2–12 KV caches.

Compare against the frozen 2B1 reset trajectory:

```text
1:     5.9737106323
2:     5.8261914968
4:     5.7499492407
8:     5.7218983650
16:    5.7104103088
32:    5.7059802055
64:    5.7035196781
128:   5.7021847963
never: 5.7013566494
```

The key question is:

> Does writer learning especially improve performance when memory survives through multiple recurrent transitions?

---

# 34. Representation drift

Compare full-context teacher raw high states against the recurrent student's RAW high states, as before.

Also separately compare:

```text
teacher raw source
vs
adapted writer memory
```

Report by source and position bins:

```text
1–16
17–32
33–64
65–128
129–256
257–512
513–1023
```

Metrics:

```text
cosine similarity
RMS difference
norm ratio
```

Important:

Do not assume the optimal writer memory must become more teacher-like.

This is diagnostic only.

A successful writer may intentionally produce a representation different from the teacher.

---

# 35. No HellaSwag yet

Do NOT run full recurrent HellaSwag in 2B2.

First determine whether writer learning produces a meaningful FineWeb validation improvement.

If 2B2 is clearly positive:

estimate the cost of HellaSwag and recommend whether it belongs after:

```text
5M
or
a later 25M writer experiment
```

but do not launch it without separate approval.

---

# 36. Checkpoint format

The 2B2 checkpoint must preserve:

```text
frozen base model state
frozen 2B1 reader state
writer adapters
writer AdamW state

local 2B2 update count
processed 2B2 tokens

four replay-loader states

Python RNG
NumPy RNG
Torch CPU RNG
Torch CUDA RNG

next-global-batch hash

2A3 checkpoint lineage
2B0 tag
2B1 tag/commit/checkpoint
2B2 branch/commit

writer architecture
rank
initialization seed
one-step temporal-credit policy
```

Write atomically.

Reopen and strict verify.

Generate SHA256.

---

# 37. Hard safety invariants

At all times require:

```text
trainable params:
exactly 49,152

GPT-2 / Full-AttnRes gradients:
none

2B1 reader gradients:
none

teacher training calls:
0

stored historical K/V grad_fn:
none

writer input source:
detached

writer output for next token:
retains grad graph until one use

temporal gradient horizon:
exactly one token

loss finite
writer gradients finite
optimizer moments finite
memory finite
KV finite
```

Any failure is a HARD STOP.

---

# 38. Explicitly forbidden changes

Do NOT:

```text
unfreeze v16-producing Block 8
unfreeze v17-producing Block 9
unfreeze v20-producing Block 10
unfreeze v24-producing Block 12

unfreeze reader
add new memory sources
add new feedback destinations

increase writer rank
add biases
add learned writer RMSNorm
add writer gates
add attention inside writer

add multi-token BPTT
add TBPTT horizon > 1
add auxiliary loss
add teacher distillation
add teacher-state matching
add RL
change context length
mask more KV layers
```

This experiment tests one thing only:

> Can a memory-only writer learn from one-step future language-model loss?

---

# 39. Final classification

Classify exactly one:

```text
MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY

MEMORY-WRITER LEARNING IMPROVES GENERIC COMPENSATION ONLY

MEMORY-WRITER LEARNING IS NEUTRAL

MEMORY-WRITER LEARNING DEGRADES

MEMORY-WRITER LEARNING IS UNSTABLE
```

Use the following frozen rule.

## Improves recurrent memory

Require:

```text
writer_gain >= 0.01
```

AND:

```text
writer_specific_gap
>= 2B1 specific gap - 0.005
```

AND no integrity failure.

## Generic compensation only

Use if:

```text
writer_gain >= 0.01
```

but the real-vs-shuffled sequence-specific gap materially collapses.

## Neutral

Use if:

```text
abs(writer_gain) < 0.01
```

with stable/integrity-passing behavior.

## Degrades

Use if:

```text
writer_gain <= -0.01
```

without numerical instability.

## Unstable

Use for:

```text
NaN/Inf
state divergence
causality failure
gradient-horizon failure
cache failure
```

---

# 40. Decision questions

At the end answer separately:

### Question A

> Should the same memory-writer experiment continue from 5M toward approximately 25M tokens?

### Question B

> Is there now enough evidence to jointly train the memory writers and the existing 1,537-parameter reader?

### Question C

> Is there enough evidence to increase the temporal credit horizon beyond one token?

These are separate decisions.

Do not automatically launch any continuation.

---

# 41. Required final report

Provide:

## Git

```text
2B1 frozen tag:
2B1 commit:
2B2 branch:
implementation commit:
results commit:
```

## Initialization

```text
source 2B1 checkpoint:
SHA256:

writer rank:
writer params:
initialization seed:

zero-writer identity regression:
PASS/FAIL
```

## Gradient semantics

```text
writer input detached:
PASS/FAIL

loss(t+1) → memory(t):
PASS/FAIL

loss(t+2) → memory(t):
must be NONE

historical-KV temporal gradient:
NONE

reader gradient:
NONE

base gradient:
NONE
```

## Training

```text
updates:
tokens:
optimizer:
LR:
runtime:
peak VRAM:
```

## Restart

```text
update-5 checkpoint SHA:
fresh-process resume:
next-batch verification:
```

## Canonical validation

```text
full context:
masked:
2B1 frozen self:
trained writer real:
trained writer shuffled:
writer bypass:
gate-zero:
teacher sources + writer:
```

## Metrics

```text
writer gain:

2B1 specific gap:
writer specific gap:
specific-gap gain:

writer recovery fraction:
```

## Per-source writer behavior

```text
source | delta RMS | source RMS | ratio | cosine(source, adapted)
v16
v17
v20
v24
```

## Routing

```text
v16:
v17:
v20:
v24:
entropy:
feedback RMS:
```

## Reset horizon

If run:

```text
interval | 2B1 | 2B2
1
2
4
8
16
32
64
128
never
```

## Writer ablations

If run:

```text
writer | ablated loss | delta
v16
v17
v20
v24
```

## Final classification

Print exactly one frozen classification.

Then answer Decision A, B, and C.

End exactly with:

# EXPERIMENT 2B2 5M COMPLETE

Do not launch any additional optimizer step.
