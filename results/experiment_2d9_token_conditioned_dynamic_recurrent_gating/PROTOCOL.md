# EXPERIMENT 2D9 — TOKEN-CONDITIONED DYNAMIC RECURRENT GATING

Execution protocol v1 — prepared 2026-09-05 against the local code at `8b25beaf`.

Implement and execute exactly this experiment. Train two matched siblings, Static and Dynamic, from the accepted O1 checkpoint, then evaluate three final conditions. Optimize for information per GPU-hour and a few-hour turnaround. Complete the primary experiment, preserve its artifacts, stop the GPU resources, and report. Do not launch a follow-up.

## 1. Workspace, implementation and scope

Current implementation worktree:

```text
/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b
```

Current branch is `experiment-2d8-trained-overlap-width-n-o1-o2`; inspected HEAD is `8b25beaf`. Inspect current status before editing, preserve unrelated work, and create:

```text
branch: codex/experiment-2d9-token-conditioned-recurrent-gating
tag on completed scientific results:
experiment-2d9-token-conditioned-recurrent-gating-final
```

Use this worktree or an isolated worktree from this verified code. The older `/Users/rahul/Documents/GPT-2 Enhancement/build-nanogpt` checkout is not the latest implementation. Do not overwrite historical branches, results or tags.

Reuse the established kernels and continuation/evaluation machinery:

```text
scripts/experiment_2d3a_core.py  — shared recurrent attention blocks
scripts/experiment_2d6_core.py  — native B6, three source rings
scripts/experiment_2d7_core.py  — accepted O1, represented by arm "O"
scripts/experiment_2d7.py       — checkpoint/optimizer/continuation machinery
scripts/experiment_2d8.py       — fresh 4096-sequence evaluation pattern
scripts/experiment_2d8_analysis.py — statistics/provenance patterns
```

These paths are relative to the worktree above. Implement a small 2D9 kernel/driver/config and targeted tests; do not rewrite the framework. Existing drivers bind experiment constants and sometimes modify imported globals: reuse deliberately, without inheriting their old parent SHA, counters, arm names or data ledger accidentally.

No joint softmax, hierarchical mixture, MLP, per-head gate, source-depth routing, changed windows, new source links, W2 intervention, auxiliary loss, extra seeds, or automatic continuation. Historical brainstorming files provide context; this protocol fixes the scope.

## 2. Scientific comparison and unchanged architecture

The question is whether the recurrent contribution benefits from a token-conditioned scalar instead of a static scalar. Both arms retain:

| Block | Local window | Recurrent source | Recurrent lags |
|---|---:|---|---|
| B1 | W2 | raw B12 post-MLP residual | 1–1023 |
| B2 | W1024 | none | — |
| B3 | W32 | raw B10 post-MLP residual | 31–1023 |
| B4 | W1024 | none | — |
| B5 | W64 | raw B8 post-MLP residual | 63–1023 |
| B6–B12 | W1024 | none | — |

Keep source identity `j=t-lag`, the full lag-1023 horizon, raw source storage, destination LN/Q/K/V projections, and separate local/recurrent softmaxes. Do not introduce recurrent lag 0. B7→B6 computation and its raw source ring remain absent. Keep dormant `g_rec_b6` and its old optimizer state only for compatibility; it must receive no gradient or update.

Per head, the inherited computation is:

```text
A_local = softmax(Q K_local^T / sqrt(64)) V_local
A_rec   = softmax(Q K_rec^T   / sqrt(64)) V_rec
A       = A_local + gate * A_rec
```

Concatenate heads, then apply the existing destination `c_proj` exactly once, including its bias once. Normal residual and MLP processing follows. Do not combine attention probabilities before multiplying by V. The local coefficient remains 1; this is an additive signed recurrent contribution.

## 3. Immutable common source

Both S and D start independently from the exact **2D7 O1** checkpoint, not from one another or from O2:

```text
Local:
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/O/scientific_cumulative_001200619520.pt

Historical remote location, resolve on the actual mounted volume:
/workspace/exp2d7_boundary_alignment/run/checkpoints/O/scientific_cumulative_001200619520.pt

SHA-256:
c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6

global update:       2290
cumulative targets: 1,200,619,520
parameters:         124,475,908
```

Expected source terminal identities, calculated with the existing project's canonical identity functions:

```text
loader cursor SHA:
682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3
next global batch SHA:
a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe
next stream SHA:
7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567
```

Verify source SHA, architecture, counters and state. Restore the exact source model, optimizer, scheduler metadata, Python/NumPy/Torch CPU/CUDA RNG and loader. Path relocation is permissible only with unchanged shard content/order and preserved logical identities. Do not approximate the cursor or reseed the scientific run.

## 4. Static and Dynamic gates

**S — Static:** exact accepted O1 behavior, with `gate_d=tanh(g0_d)`. No new parameters; total 124,475,908.

**D — Dynamic:** add one 768-element vector per recurrent destination:

```text
w_B1, w_B3, w_B5
```

All three start at exact zero. Reuse the existing raw gate parameters and their moments:

```text
B1 g0 = g_rec
B3 g0 = g_rec_b3
B5 g0 = g_rec_b5
```

Do not initialize `g0` from `tanh(g0)`; that would transform the gate twice. There are exactly 2304 new parameters, giving Dynamic total **124,478,212**. No learned normalization scale or additional bias.

For each destination, capture `h` as the current token's residual input **before `block.ln_1` and before attention**, then calculate:

```python
# Conceptual code; w and old master parameters retain inherited FP32 storage.
# Disable autocast inside this small gate calculation.
r = h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-5)
u = g0.float() + (r * w.float()).sum(-1, keepdim=True)
g = torch.tanh(u)
```

Freeze **RMS epsilon = 1e-5**, matching the project's existing RMS convention. No `sqrt(768)` scaling. Compute RMS, dot product and tanh in FP32; cast the final coefficient to `A_local.dtype` immediately before multiplying `A_rec`, matching the inherited scalar-gate cast. Do not promote the full attention output to FP32 as an incidental change.

For `[B,12,T,64]` attention outputs, reshape `[B,T,1]` gates to `[B,1,T,1]`; incremental execution has T=1. Each token uses the same gate across all heads and head dimensions. Implement identical semantics in parallel multi-pass training and incremental inference. No new persistent gate cache.

Use the actual current pass's `h`. Keep gradients through `h`, RMS normalization, `w` and the inherited `g0` attached. A shortcut that detects `w==0` and skips its computation would break first-step learning: do not use one. Pass gate inputs explicitly rather than storing mutable residual references that activation-checkpoint recomputation could overwrite.

When recurrent memory is unavailable, retain the existing zero recurrent output and safe empty-mask behavior. Computing an intrinsic gate does not create memory or a recurrent contribution at those positions.

**Interpretation:** at B1, this pre-attention input is token embedding plus position embedding, so its gate is token/position-conditioned. B3/B5 inputs also contain preceding contextual processing. Keep this distinction in the report; do not change the requested gate input to make B1 context-aware.

## 5. Optimizer and matched training

Preserve every old parameter's value, optimizer moments, step counter and group settings. Keep tied embedding/output weights tied. Old Adam step counters are not necessarily equal to one another or to the experiment's global update.

Expected inherited settings, verify against the checkpoint:

| Setting | Expected |
|---|---:|
| Base LR | 3e-5 |
| Existing recurrent scalar LR | 3e-4 |
| Base matrix weight decay | 0.1 |
| Base vector/scalar weight decay | 0 |
| Adam betas | (0.9, 0.95) |
| Adam epsilon | 1e-8 |
| Global gradient clipping | 1.0 |
| Microbatch | B32 × T1024 |
| Accumulation | 16 microbatches/update |

Add only the three `w` vectors to **base no-decay semantics**, LR 3e-5, with fresh optimizer state. Restore old state before appending new parameters, or use an explicit name/identity mapping; do not load a positional optimizer state into resized groups blindly. A separate new group with identical base-nodecay settings is acceptable if scheduler/reload behavior is exact. The new vectors participate in the same global gradient clipping as the other active parameters.

Preserve the constant continuation LR/scheduler metadata; do not restart the original baseline warmup. If recorded source settings differ from this table, resolve the discrepancy from authoritative provenance before training instead of silently overriding it.

Freeze a **new** 191-batch continuation from O1's terminal cursor. This is not a replay of 2D7's updates 2100–2290. Record per-update logical batch identities and pass counts once, then validate both arms against the same ledger.

```text
updates per arm:       191
targets/update:        524,288
new targets per arm:   100,139,008
first global update:   2291
last global update:    2481
final target counter:  1,300,758,528
```

Use CE only, all existing active parameters trainable, attached recurrent writer gradients, inherited BF16 autocast and activation checkpointing, and unchanged accumulation. Preserve the actual inherited pass schedule: three passes when global update is divisible by 32, otherwise two.

```text
Three-pass global updates: 2304, 2336, 2368, 2400, 2432, 2464
Counts: 185 two-pass updates + 6 three-pass updates
Two-pass loss weights:   (0.25, 0.75)
Three-pass weights:      (0.20, 0.40, 0.40)
```

Targets count once per batch, not once per pass. Do not require all Adam counters to equal 2481: verify each old active counter advances 191 from its own source value, new `w` counters reach 191, and dormant B6 state is unchanged.

Prefer the inherited B32/accumulation16 in both arms. If a real OOM requires a smaller microbatch, choose one common divisor and accumulation before scientific training, document the change, preserve the exact 524,288-target ordering, and use it for both arms. Do not independently retune arms.

## 6. Targeted preflight, then train

Do preparation and tests locally/CPU where possible before paid GPU time. Run only targeted checks:

1. Exact source/tensor inventory, parameter counts, tied weights and old optimizer-state equality between S and D at initialization. Only D's three zero vectors may be new.
2. Static execution still matches the accepted O1 implementation. With `w=0`, Dynamic matches Static for intrinsic gates, gated recurrent outputs, attention outputs, logits and loss, separately in parallel and incremental modes. Do not require parallel training itself to equal incremental recurrence.
3. Use short deterministic FP32 tests with `rtol=2e-5, atol=2e-6` and the same tight comparison on identical BF16 paths after converting outputs to FP32. Require identical final cast gate values at zero initialization and absolute CE difference ≤1e-6. Compare on the same device/backend. A failure must be diagnosed as an implementation/numerical discrepancy, not hidden by borrowing a loose parallel-versus-incremental tolerance.
4. Test future-suffix invariance and batch-row isolation **with nonzero disposable w as well as w=0**; zero vectors alone could hide a leaking gate input. Use at least two rows and positions reaching past lag63. Confirm no `detach` or wrong-source input in the dynamic gate.
5. Check exact O1 masks, source positions, bounded caches and B6 W1024/no B7 ring. Reuse the existing cache audit on the first full-length final evaluation batch for physical-byte confirmation.
6. One short disposable backward/update/reload smoke per arm is sufficient. On D, use a valid recurrent pass/length and require finite, nonzero gradients for all three w vectors on its first backward, plus finite active old gradients. Verify final-state schema can reload the three new vector moments. No special loss or artificial result-training step to open the gate.

Do not use broad attention diagnostics merely to collect gate statistics: the inherited `return_diagnostics` path can materialize large attention matrices. Add a narrow gate-statistics collector instead.

Discard all smoke state, then independently reload the exact source and scientific RNG/loader for each arm. Commit the tested implementation/config/protocol before result training; run the same committed version for both arms. No code changes mid-run.

Maintain concise progress and per-update loss/finite-gradient/counter records. Small or negative gates, near-zero w, or unimpressive loss are valid outcomes, not reasons to change the experiment. No milestone validation, mandatory midpoint evaluation or elaborate restart drill. Final checkpoint is required; a single recovery checkpoint around local96 is optional for operational reliability.

If interrupted, resume from verified complete state without duplicating logical updates. One completed arm need not be rewound because the other crashed. Require final matched ledgers, not simultaneous wall-clock progress. Stop on unresolved source/replay corruption, leakage, wrong cache semantics or NaN/Inf; report partial status without claiming completion.

## 7. Final checkpoints

Keep S and D in isolated directories. Suggested runtime root:

```text
/workspace/exp2d9_dynamic_gating/run/{S,D}/
```

Final filename in each arm:

```text
scientific_cumulative_001300758528.pt
```

Include model, exact optimizer/groups, scheduler state or established scheduler metadata, all RNG states, loader, local/global/target counters, parent SHA, architecture/gate formula/epsilon/precision, arm, parameter names, code commit, ledger hash and next-batch/stream identities. Strict-reopen both final checkpoints using the new schema. Final loader cursor, next global batch SHA and next stream SHA must match S versus D; derive their new values, do not reuse the parent's terminal hashes.

## 8. One fresh matched panel, three conditions

Use **4096 sequences × 1024 targets = 4,194,304 targets per condition**, with the established 64 canonical B64 validation batches and true incremental evaluator. Reset all model/cache state per sequence as in 2D8. BF16 model execution, FP32 token CE and FP64 accumulation must be identical across conditions.

Use the existing FineWeb-Edu validation shard and verify its identity. Expected SHA:

```text
8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f
```

Panel selection: `numpy.random.default_rng(20260905)`, seeded permutation of canonical complete validation batches, taking the first 64 eligible batches. Start from the established historical exclusion manifests/reserved prefix, and add all recoverable subsequent panels, especially the **2D8 4096-sequence panel**, 2D7 and 2D6F. Exclude overlapping historical target spans, preserve ordered identities and record exclusion evidence. Do not select on model losses. If the requested disjoint panel cannot be formed, report that prerequisite failure instead of silently calling reused data fresh.

Freeze the panel manifest before training results or final scores are inspected. Use a separate RNG instance for panel selection so it cannot alter scientific training RNG. No post-result reselection or enlargement.

Evaluate exactly:

| Condition | Final checkpoint | Gate behavior |
|---|---|---|
| STATIC_REAL | S | inherited static formula with S-trained g0 |
| DYNAMIC_REAL | D | full learned dynamic formula |
| DYNAMIC_STATICIZED | D | functionally omit w term, retain D-trained g0 and base weights |

Staticized is a model-wide mode for the entire recurrent trajectory: reset caches and generate all its stored source states under staticized behavior. Do not reuse Dynamic-Real caches. Do not reset g0 to the parent or refit it. Do not mutate the D checkpoint; verify its tensors/hash are unchanged by controls. Add a small round-trip mode test without a fourth large evaluation.

Save every ordered sequence identity, target count, CE/NLL and associated checkpoint/panel/condition hash. No G/O2/parent/old-panel reevaluation, OFF/SHUFFLED suite, HellaSwag, per-link decomposition, or extra panel. These three conditions are the complete primary final evaluation.

## 9. Statistics and decision rules, fixed before results

Let:

```text
A = CE(Static) − CE(Dynamic)           # matched architecture benefit
P = CE(DynamicStaticized) − CE(Dynamic)# learned w-term benefit on D weights
R = CE(Static) − CE(DynamicStaticized) # descriptive residual; A = R + P
delta_CE = 0.0001
```

On CPU, use `numpy.random.default_rng(20260906)`, 50,000 paired sequence-level bootstrap resamples, shared indices across A/P/R, 95% percentile CIs with NumPy's linear percentile convention. Chunk computation to avoid unnecessary memory. Pair before resampling; do not bootstrap tokens or arms independently. Report means, bounds, perplexities, `exp(contrast)` ratios and sequence win counts. No pooled old-panel analysis: it has no matched S/D data.

For both A and P separately report these flags, allowing statistical and equivalence flags to coexist:

- positive utility: lower CI > 0;
- benefit beyond margin: lower CI > +delta;
- negative utility: upper CI < 0;
- harm beyond margin: upper CI < −delta;
- practical equivalence: lower CI > −delta and upper CI < +delta;
- noninferiority of the second named condition: lower CI > −delta.

Use strict inequalities; touching a boundary does not clear it. Equality/overlap with zero is unresolved for sign. Do not relabel “not significant” as equivalence.

Primary classification, first applicable row:

| Rule | Classification |
|---|---|
| Integrity failure | INVALID / INCOMPLETE — no scientific winner |
| A or P upper CI < −delta | MATERIAL HARM ESTABLISHED — state whether architecture, active w term, or both |
| Both A and P lower CI > 0 | TOKEN-CONDITIONED GATING ESTABLISHES UTILITY |
| Only A lower CI > 0 | MATCHED ARCHITECTURE BENEFIT; ACTIVE W-TERM UTILITY NOT ESTABLISHED |
| Only P lower CI > 0 | ACTIVE W-TERM UTILITY; MATCHED ARCHITECTURE BENEFIT NOT ESTABLISHED |
| Both means positive, neither lower CI > 0 | DIRECTIONALLY POSITIVE, NOT ESTABLISHED |
| A practically equivalent and P lower CI ≤ 0 | ARCHITECTURES PRACTICALLY EQUIVALENT; NO ESTABLISHED ADDED UTILITY |
| Otherwise | MIXED / UNRESOLVED — report both contrasts explicitly |

The classification is separate from adopting a more complex architecture:

- Recommend D as the preferred tested architecture only if **A lower CI > +0.0001 AND P lower CI > 0**, with zero persistent-state growth and passing integrity. Report measured runtime tradeoff if available.
- If both primary lower CIs exceed zero but A does not clear +0.0001, report established statistical utility but retain static O1 provisionally; recommend a matched 250M continuation only as a future decision.
- If only directional, mixed, equivalent without established benefit, or harmful, retain static O1. Explain what remains unresolved. Do not automatically add training/evaluation to force a decision.

A/P share Dynamic and are not independent replications. CIs quantify evaluation-sequence uncertainty for these sealed training trajectories, not variation across training seeds.

Staticized removes the complete learned w term, including any average gate shift as well as token variation. It establishes that term's inference utility if P is positive, not that token variation beats every optimally refitted constant gate. Preserve this limit; do not add another large control in this experiment.

## 10. Gate statistics and memory

Collect statistics during the existing DYNAMIC_REAL evaluation, detached from autograd and without extra model conditions. For each B1/B3/B5 report:

- parent and final raw g0, final `tanh(g0)`, and `||w||_2`;
- FP32 preactivation and effective-gate mean/std/min/max;
- gate p1/p5/p25/median/p75/p95/p99, negative fraction, and mean absolute deviation from final `tanh(g0)`;
- standard deviation across all tokens; standard deviation of per-sequence gate means; mean within-sequence standard deviation. State these definitions.

Report statistics for both mathematical FP32 g and the cast coefficient actually used in BF16 attention, so unrepresented sub-BF16 changes are visible. Use running summaries plus compact float32 scalar arrays or a fixed deterministic quantile sample, never token-by-token JSON. State if quantiles are sampled. No attention matrices, residual vectors, text examples or per-head collection.

Buffer detached scalar gate values on-device for at most one evaluation batch, then transfer once per batch. Avoid per-token CPU transfers or `.item()` synchronization. If quantiles are sampled, use a separately seeded fixed sample of at most 131,072 gate positions per destination; all means, variances, extrema and fractions should use the full panel. Statistic collection must not alter model or training RNG state.

If cheap, include zero-based position bins `0`, `1–31`, `32–63`, `64–127`, `128–255`, `256–511`, `512–767`, `768–1023`. Mark positions without eligible recurrent memory; an intrinsic gate there has no recurrent output to scale.

Do not infer usefulness from nonzero w, gate size or variance alone.

Expected persistent K/V plus raw recurrent state for both arms: **33,289,728 BF16 bytes per B=1 sequence** under inherited accounting. Delta 0. Additional parameter payload: 2304 values = 9216 bytes in FP32 or 4608 bytes in BF16; optimizer state is training-only and separate. Diagnostic arrays are analysis artifacts, not model inference state.

A short optional timing test may compare S/D with gate diagnostics disabled, identical B/T and warmup, five alternating repeats, reporting median/range. Bound it to about five minutes; otherwise omit and report evaluation wall times as descriptive workload timings. Do not optimize kernels during this scientific run.

## 11. GPU workflow and storage

Prefer two independent A100-80GB processes, one per arm, no DDP or gradient sharing. One A100 with independent sequential reloads is valid; do not rent four GPUs. Resolve actual available hardware, mounted data, storage and authenticated shutdown capability rather than reusing historical pod IDs blindly. Do not start an unattended long run if its exact shutdown path is unavailable.

Target a few hours end-to-end, not a guaranteed deadline. Historical O1-family 100M training was about 28–30 minutes per arm and each 4096-sequence evaluation about 19 minutes; implementation, smoke, storage and transfer add time. Prepare code/ledgers/panels on CPU where possible, reuse existing data caches, and avoid repeated large checkpoint copies/hashes.

Each final checkpoint is about 1.5GB plus the small new parameter state. Check space before training; preserve unique scientific checkpoints and persistent volumes. If space is inadequate, use an existing safe scratch location or report the constraint; do not delete historical science to make room without authorization.

After both final checkpoints pass strict reopen, all three conditions and cheap statistics are exported, and local/persistent backups have independently matching SHAs, **stop the exact experiment GPU pod(s)**. Stop, do not delete; retain persistent storage. On a shared two-GPU pod, do not stop after the first arm finishes. If an unrecoverable failure terminates the experiment, preserve available recovery artifacts and stop its idle GPU resources rather than leaving them running.

Perform 50k bootstrap, report generation and final Git packaging CPU-side/local after shutdown. Verify actual stop status and record evidence; do not claim a submitted stop request alone proves completion. If automatic shutdown fails, clearly report it and the exact remaining resource.

## 12. Compact artifacts and completion

Under the implementation worktree, use:

```text
results/experiment_2d9_token_conditioned_dynamic_recurrent_gating/
```

Keep a compact artifact set; these are contents to preserve, not a demand for hundreds of individual files:

```text
PROTOCOL.md + frozen config
SOURCE_AND_PARAMETER_AUDIT.json
PREFLIGHT_AUDIT.json
CONTINUATION_MANIFEST.json + MATCHED_BATCH_LEDGER.jsonl
TRAINING_S.jsonl + TRAINING_D.jsonl
CHECKPOINT_MANIFESTS.json
EVALUATION_PANEL_MANIFEST.json + DISJOINTNESS_AUDIT.json
STATIC_REAL.json + DYNAMIC_REAL.json + DYNAMIC_STATICIZED.json
GATE_STATISTICS.json + optional compact scalar data
PAIRED_BOOTSTRAP.json
MEMORY_AND_RUNTIME.json
FINAL_AUDIT.json + STOP_VERIFICATION.json
EXPERIMENT_2D9_FINAL_REPORT.md
```

Final checkpoint archives:

```text
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating/{S,D}/scientific_cumulative_001300758528.pt
```

Do not commit large checkpoints/data arrays to Git. Commit the implementation, protocol, compact results and report; tag the completed scientific result and push the new branch/tag to the project's configured remote when available. Never move an existing sealed tag. If sync is unavailable, preserve local commits and report the sync limitation separately from scientific completion.

Final audit covers the source; only 2304 new D parameters; exact zero-effect behavior; causal dynamic path; optimizer/cadence/191-batch matching; dormant B6; strict reopen; terminal stream equality; one frozen fresh panel with exactly three 4096-loss arrays; paired analysis; state delta; verified backups; and GPU stop. Consolidate evidence instead of rerunning successful checks.

Begin the user-facing result with:

```text
EXPERIMENT 2D9 — TOKEN-CONDITIONED DYNAMIC RECURRENT GATING COMPLETE

Primary classification:
Preferred architecture recommendation:

Static CE:
Dynamic CE:
Dynamic Staticized CE:

Static − Dynamic + 95% CI:
Staticized − Dynamic + 95% CI:
Static − Staticized + 95% CI (descriptive):
delta_CE = 0.0001

B1/B3/B5: g0, ||w||, gate mean/std/range
Persistent-state delta:
Parameter increase:
Measured runtime, if available:
Static/Dynamic checkpoint SHAs:
Terminal stream equality:
Audit:
GPU stopped status:
```

Then briefly answer whether D beats matched S, whether the trained w term helps D at inference, what gate variation was learned, and which architecture is preferred. Give one next recommendation without executing it. Link the full report and state **exactly 191 updates / 100,139,008 new targets per arm; no follow-up launched**. If incomplete, say INCOMPLETE and identify the remaining work instead of printing COMPLETE.

Proceed with routine implementation and the defined experiment without adding optional diagnostic campaigns or requesting repeated confirmation for already-scoped work. Stop at the two-arm 100M result.
