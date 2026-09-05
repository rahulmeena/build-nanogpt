# EXPERIMENT 2D10 — RETRIEVAL-AWARE TANH VS HIERARCHICAL SOFTMAX, 100M

Please implement and execute this experiment on one pod with two A100-80GB GPUs, one independent training arm per GPU. Train exactly two new models, reuse the two sealed 2D9 100M controls, evaluate all four on one fresh panel, preserve the artifacts, stop the pod, and finish statistics/reporting/Git locally. This is a 100M screening experiment; stop after reporting it. Further training is a separate user decision.

## 1. Scope and reference implementation

Worktree: `/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b`.

Start the implementation branch from the verified completed 2D9 lineage, currently result commit `adec918f436d264840e5b60d60a589463bb1bca9`. Use a new `codex/experiment-2d10-retrieval-aware-gating-100m` branch and preserve existing branches/tags/results. This code revision is not the model initialization checkpoint.

Read the sealed 2D9 100M protocol, final report, source/optimizer audit, continuation manifest and matched batch ledger under `results/experiment_2d9_token_conditioned_dynamic_recurrent_gating/`. Reuse `scripts/experiment_2d9_core.py`, the underlying O1 kernels, evaluation machinery and proven continuation handling. The original 2D9 prohibition on MLP/hierarchical gates is superseded for these two new arms only. Its original source-state restoration, causal cache rules and numerical conventions otherwise apply unless explicitly changed below.

| Label | Architecture | Work in this experiment |
|---|---|---|
| S | Token-independent learned scalar, `A_L + tanh(g0) A_R` | Reuse sealed 2D9 100M Static; evaluate only |
| D | Accepted token-conditioned scalar, `A_L + tanh(g0+w^T RMS(h)) A_R` | Reuse sealed 2D9 100M Dynamic; evaluate only |
| T | Retrieval-aware additive tanh, defined below | Train 100M from original 2D7 O1 |
| H | Retrieval-aware two-branch softmax, defined below | Train 100M from original 2D7 O1 |

"Static" means token-independent, not that its scalar was frozen during its original training. Do not train S/D again. Do not use 250M checkpoints as the 100M controls, or initialize T/H from trained D. All four architectures must share the original pretrained O1 parent and exactly the same subsequent 100M data exposure. This is adaptation from a pretrained parent, not training from scratch.

## 2. Immutable sources and exact replay

T and H independently load:

```text
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/O/scientific_cumulative_001200619520.pt
SHA256: c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6
Global update: 2290
Inherited cumulative targets: 1,200,619,520
Registered parameters: 124,475,908
```

Read-only S/D control files are under:

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating/{S,D}/scientific_cumulative_001300758528.pt`

```text
S SHA256: 676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b
D SHA256: c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44
Both control endpoints: global update 2481; inherited targets 1,300,758,528
Sealed 100M result commit: 482ad55637c2a0adb5c7c268b37c7be243ac15c8
```

Resolve persistent copies from the saved manifests and verify content hashes. Preserve source/control files and optimizer states. A missing/mismatched source or unrecoverable replay is a prerequisite failure, not permission to substitute another age/checkpoint.

Replay the original 2D9 `MATCHED_BATCH_LEDGER.jsonl`, SHA256:

`3955889e1c0849fa2ee0072cf1ca109170e955d3fc6914d970f6c58bf1b01bbd`

Restore the parent's exact model, old optimizer groups/moments/individual counters, scheduler metadata, scientific Python/NumPy/Torch CPU/CUDA RNG and loader state. Initialize added parameters using an isolated generator, then restore scientific RNG before training. Match every original logical batch/stream identity and global pass count. Do not generate the next batches after the 250M endpoint. Path relocation is allowed only with identical shard contents/order and unchanged logical data identity.

```text
Parent cursor SHA: 682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3
First next-batch SHA: a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe
First next-stream SHA: 7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567

Per NEW arm: 191 updates × 524,288 = 100,139,008 targets
Global updates: 2291–2481 inclusive
Final inherited targets: 1,300,758,528
Final filename: scientific_cumulative_001300758528.pt
```

Terminal identities must match the original 100M controls and ledger:

```text
Cursor SHA: d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d
Next-batch SHA: 400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649
Next-stream SHA: 0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642
```

## 3. Unchanged O1 attention and new router inputs

Keep GPT-2 124M, width768, 12 heads of width64, T1024, vocab50304. Keep all normal backbone parameters trainable and embeddings/output weights tied.

| Destination | Native W | Raw post-MLP recurrent source | Eligible lags |
|---|---:|---|---|
| B1 | 2 | B12 | 1–1023 |
| B3 | 32 | B10 | 31–1023 |
| B5 | 64 | B8 | 63–1023 |

All other blocks retain W1024. No B7→B6, no new source ring, no lag0, no window growth in this experiment. Keep source identity `j=t-lag`, ring capacity1023 and attached writer gradients. Compute separate local and recurrent attention softmaxes and their separate value-weighted outputs.

At each destination and current token, obtain:

- `h`: residual entering attention, before existing ln_1; used only by T's inherited-style linear gate path.
- `q`: the actual existing projected attention query, before the SDPA 1/sqrt(64) scaling.
- `A_L`, `A_R`: completed local and recurrent attention outputs, after multiplication by V and before gating or c_proj.

For router input only, concatenate the 12 heads of each q/A_L/A_R in the same order used by c_proj, giving three width768 vectors. Define `N(x)` as FP32 affine-free LayerNorm over those 768 features with epsilon1e-5, population variance, and no learned scale/bias. Normalize each vector separately, then concatenate:

`phi = [N(q); N(A_L); N(A_R)]`, shape `[B,T,2304]`.

Normalization is token-local, never across sequences, positions or branches. Keep the original unnormalized branch outputs for the final combination. Use current-pass tensors in parallel training and current-token tensors in incremental inference. Do not detach router inputs or store mutable gate intermediates for activation-checkpoint recomputation.

Both new arms use independent routers at B1/B3/B5, each with hidden width32 and SiLU. No per-head coefficients, dropout, extra depth, temperature, entropy loss, gain parameter, joint token softmax or null third branch. Compute normalization, MLP and gate nonlinearity in FP32 with FP32 parameter masters. Cast final coefficients to the existing attention dtype immediately before combination; retain the inherited BF16 branch/projection path. Apply the existing shared c_proj once, including its bias once.

## 4. T: retrieval-aware tanh

For each destination:

```text
r(h) = h / sqrt(mean(h^2) + 1e-5)       # affine-free FP32 RMS
u = g0 + w^T r(h)
hidden = SiLU(W1 phi + b1)              # W1: [32,2304], b1: [32]
delta = W2 hidden                     # W2: [1,32], NO output bias
g = tanh(u + delta)
A = A_L + g A_R
```

Keep the original raw g0 values, their moments and their training behavior. Initialize each new width768 w vector to zero. Set W2 to zero, b1 to zero, and initialize W1 from Normal(0,0.02), using a dedicated CPU torch.Generator seeded `20260908 + block_index` (zero-based indices 0,2,4). Copy the same initial W1/b1 tensors to the corresponding H router; record their hashes. Do not consume scientific training RNG.

T initially reproduces the original O1 function. Keep its zero-valued paths differentiable; do not skip computation because weights are zero. A bias-free output head avoids adding a second explicit constant offset alongside the inherited trainable g0.

New T parameters: `3 × (768 + 2304×32 + 32 + 32) = 223,680`.
Registered total: **124,699,588**. Added FP32 parameter payload: **894,720 bytes**.

## 5. H: retrieval-aware hierarchical softmax

For each destination:

```text
hidden = SiLU(W1 phi + b1)              # same initial hidden layer as T
[z_L,z_R] = W2 hidden + b2             # W2: [2,32], b2: [2]
[lambda_L,lambda_R] = softmax([z_L,z_R])
A = lambda_L A_L + lambda_R A_R
```

This is ordinary nonnegative, sum-to-one branch competition. There is no inherited w path, extra tanh factor, residual interpolation or compensating output gain.

Initialize W2 to zero. Let `c = tanh(parent_raw_g0)` evaluated from the exact source in FP32. Verify c is finite and strictly positive for each destination. Initialize b2 to `[0, log(c)]`. Do not use rounded report values or silently clamp an invalid source.

At an eligible position, initial coefficients are `1/(1+c)` and `c/(1+c)`: this preserves the parent's local-to-recurrent coefficient ratio. It changes the combined attention output from `A_L+c A_R` to `(A_L+c A_R)/(1+c)`. There is deliberately no claim of zero-effect initialization. Measure and report the initial change; do not tune initialization against validation scores. Compare trained architectures rather than declaring the initial intervention loss to be the scientific result.

H replaces the old three recurrent gate scalars computationally. Retain their exact source tensors and optimizer states for compatibility, but exclude them from H's forward graph: no gradients, updates, weight decay or moment/counter changes. Its new output biases are trainable and replace their functional role. Dormant B6 remains unchanged in both arms. This explicit H exception overrides generic instructions that all old gates must advance their counters.

New H parameters: `3 × (2304×32 + 32 + 2×32 + 2) = 221,478`.
Registered total, including compatibility scalars: **124,697,386**. Added FP32 parameter payload: **885,912 bytes**.

For every query lacking eligible recurrent memory, force H's coefficients to `(1,0)` and return A_L exactly before c_proj. Determine availability from actual valid memory/masks, not from whether the numerical A_R happens to equal zero. This includes absent recurrent banks on the first parallel pass and early positions in later passes/incremental inference. Preserve zero recurrent outputs and finite behavior. A valid bank whose weighted values cancel still participates in routing.

## 6. Optimizer, cadence and finite budget

Use inherited BF16 autocast with FP32 model/optimizer masters, CE, gradient clipping and activation checkpointing. Restore old state by explicit parameter identity/name rather than positional assumptions after adding parameters.

| Parameter group | LR | Weight decay |
|---|---:|---:|
| Inherited backbone matrices | 3e-5 | 0.1 |
| Inherited backbone vectors/scalars | 3e-5 | 0 |
| T inherited active g0 scalars | 3e-4 | 0 |
| T new w | 3e-5 | 0 |
| New router W1/W2 matrices, T/H | 3e-5 | 0.1 |
| New router hidden bias b1, T/H | 3e-5 | 0 |
| H new output bias b2 | 3e-4 | 0 |

H compatibility g0 parameters retain old groups/state but have grad=None and no updates. All new parameters start with fresh optimizer state. Adam betas(.9,.95), epsilon1e-8, global clip1.0; no warmup restart, new LR schedule or arm-specific tuning. Verify inherited settings against provenance before training.

Keep B32×T1024, accumulation16, and the original 191-update ledger. A smaller batch would no longer directly replay the established control's configuration: diagnose a real resource failure rather than silently changing one or both new arms. Two-pass weights(.25,.75); three-pass weights(.20,.40,.40). Three passes only at global updates 2304,2336,2368,2400,2432,2464: 185 two-pass plus 6 three-pass updates. Count each target once, not once per pass.

Old active parameter counters advance 191 from their individual source counters. Every new parameter counter reaches 191 under the specified normal training graph. H's three retired g0 states and dormant B6 in both arms remain identical to source. Different architectures need not produce equal gradients, weights or optimizer states.

Commit tested code, frozen config, panel manifest and protocol before training. No code/hyperparameter changes mid-run. One recovery checkpoint around update 96 is allowed; saving must preserve scientific RNG. Resume interrupted work from a verified complete state without replaying completed logical updates. A completed arm need not be rewound because the other failed. Stop on unresolved leakage, source/replay corruption or NaN/Inf; preserve partial artifacts and report incomplete.

## 7. Focused preflight

Reuse successful existing O1/2D9 tests and add tests for actual new behavior. Prepare CPU-side before renting where possible. Required checks:

1. Source/control SHA and schema, shared backbone tensors, exact old optimizer restore, tied weights, trainable/compatibility inventories and parameter counts. Verify the reused controls' full 191-update provenance against the replay ledger.
2. T's zero-added-path equivalence to source O1 in both parallel and incremental modes, including gates, combined outputs, logits and CE. Reuse the established tight tolerances (FP32 rtol2e-5/atol2e-6; identical cast coefficients and absolute CE difference<=1e-6 on matched BF16 paths). Do not equate parallel and incremental recurrence with each other.
3. H's closed-form initial coefficient/combined-output checks, FP32 simplex behavior, correct final cast, and exact local-only behavior when memory is unavailable. Record initial T/H-versus-parent logit/CE changes on one small fixed disposable training batch; this is an implementation diagnostic, not an extra full validation condition.
4. Future-suffix invariance, row isolation and correct input placement using nonzero disposable router output weights, sequence length past lag63, and both execution modes. Verify gradient paths through q, A_L, A_R and earlier recurrent writers; no detached shortcuts or future-token inputs.
5. Short disposable backward/update/reload checks using valid recurrent passes. W2 and H b2 should get finite useful gradients. Zero W2 intentionally yields zero hidden-layer gradients on the first backward; do NOT flag that as a failure or initialize both layers to zero. Require finite nonzero hidden-layer gradients after an output-head update on a suitable batch. Discard smoke state and reload exact scientific source/RNG.
6. Parameter-name-based complete-state reload, deterministic activation-checkpoint recomputation, all retired/dormant gate states unchanged, and unchanged physical cache accounting. Reuse source audits rather than repeating unrelated historical campaigns.

## 8. One fresh panel and exactly four final evaluations

Freeze 4096 sequences ×1024 targets =4,194,304 targets per condition, using 64 canonical B64 batches and the established FineWeb-Edu validation shard. Expected shard SHA:

`8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`

Use isolated `numpy.random.default_rng(20260909)` to permute canonical complete batches and select the first 64 eligible batches. Inherit the historical reserved prefix/exclusion manifests; exclude both 2D9's 100M and 250M panels and all recoverable later used/reserved target spans. Freeze before new training/scoring, record ordered identities and disjointness evidence, and fail the prerequisite if there is insufficient eligible data. Do not select on losses, reuse old scores, or enlarge/reselect after results.

Evaluate S_REAL, D_REAL, T_REAL, H_REAL once each on that panel using true incremental inference, BF16 execution, FP32 token CE and FP64 accumulation. Reset caches for each sequence batch and condition. S/D must use their sealed historical behavior with no new-router hooks active; strict-load their checkpoints and verify unchanged tensors/files. Save ordered per-sequence CE, NLL and target counts, condition/checkpoint/panel hashes and runtime. Preserve four arrays of 4096 finite paired losses.

No full parent evaluation, old-panel reevaluation, Staticized/new-router-disabled controls, per-link ablations, HellaSwag, window changes, seed repetitions or midpoint validation in this screening stage. Training logs can show the common weighted multipass objective versus targets, clearly separated from final incremental validation.

## 9. Paired analysis and a screening decision

Primary family, positive favors the second condition:

```text
A_T = CE(D) - CE(T)
A_H = CE(D) - CE(H)
C   = CE(T) - CE(H)
delta_CE = 0.0001
```

On local CPU, use 50,000 paired sequence bootstrap resamples, isolated RNG seed 20260910, with identical resampling indices across all contrasts and NumPy linear percentiles. Chunk computation. Do not resample individual tokens or unpair arms.

Report ordinary 95% CIs for readability. For the three primary comparisons, make decisions from Bonferroni-adjusted **98.333333% marginal bootstrap intervals**, percentiles[0.833333333333,99.166666666667], giving nominal 95% family coverage under the bootstrap approximation. Label adjusted intervals explicitly; do not pick the best of several candidates using only unadjusted95% bounds.

For A_T/A_H/C, report strict-boundary flags based on adjusted intervals: positive L>0; beyond-margin L>delta; negative U<0; material harm U<−delta; practical equivalence L>−delta AND U<delta; second-condition noninferiority L>−delta. Touching a boundary does not pass. Absence of significance is not equivalence.

Report S−D, S−T and S−H as secondary/descriptive contrasts with ordinary 95% intervals, means, exp(contrast) ratios and wins/ties. These secondary comparisons do not override the primary family decision or establish a unique winner. Report CE/perplexity for all four.

Decision: integrity failure means INVALID/INCOMPLETE. Otherwise identify which new candidates establish any benefit over D and which clear the practical margin. Report the direct T/H result separately; differing significance against D does not establish a difference between T/H. If their direct interval is unresolved, say so rather than forcing a winner. Give one recommendation for possible further work without executing it.

This is a 100M screen. The accepted 250M Dynamic checkpoint remains the current mature baseline; do not compare its old-panel score against these 100M models or automatically promote a younger checkpoint over it. Do not launch a 250M continuation. A weak 100M result does not prove a mechanism can never help after longer training.

These comparisons test complete trained architectures. They do not isolate the usefulness of the retrieved-output inputs versus extra query-dependent MLP capacity, and T/H differ in parameterization, signed versus convex combination, and initial function. There is no parameter-matched query-only MLP or inference-ablation control in this bounded run. Sequence CIs are not training-seed replication; the contrasts share checkpoints and are correlated.

## 10. Compact diagnostics and resource accounting

Collect detached scalar diagnostics during T_REAL/H_REAL only, with one transfer per batch, no attention matrices/residual dumps or per-token CPU synchronization. For T report raw g0, w norms, the u/delta/g distributions; for H report learned b2, logit difference, lambda_L/lambda_R and branch entropy distributions. Report FP32 and actual BF16 coefficients, mean/std/range, negative fraction where applicable, and selected quantiles. Separate eligible-memory positions from forced local-only positions. Optional position bins may reuse the established collector. Quantiles may use a fixed sample of at most 131,072 eligible positions per destination, isolated seed 20260911; state sampling and use full-panel running means/variances/extrema/fractions.

Report router weight norms and parameter counts. A changing gate or nonzero MLP is not proof that its retrieved context improves prediction. Additional query/output normalization may discard magnitude information; no norm features are added in this run.

Expected BF16 persistent KV plus raw recurrent state: **33,289,728 bytes per B=1 sequence in every condition**, delta0. Router activations are transient and model parameters are separate from historical cache state. Verify physical cache storage on the first full-length evaluation batch. An unexpected persistent-state increase is an implementation discrepancy for this fixed-window experiment.

The user's broader preference explicitly allows higher memory use if a future matched from-scratch comparison establishes a substantial performance advantage over Karpathy GPT-2. This is not a permanent zero-memory-growth requirement; it does not authorize widening windows in the present experiment.

## 11. One two-GPU pod, prompt shutdown, local completion

Use one pod with two A100-80GB GPUs, independent workers explicitly pinned to different devices. No DDP or gradient sharing. Reuse available persistent data and volume `yhzyb27fb5` if compatible; verify current mount, capacity, sources and shutdown capability. Do not blindly act on historical pod IDs or stop unrelated resources. Do not rent four GPUs or start S/D training.

Train T and H concurrently. As each GPU becomes available, schedule its final evaluation and the read-only S/D evaluations; each full condition runs exactly once. Keep at most one substantial model job per GPU. A finished GPU may evaluate a control while the other arm still trains. Do not leave GPUs running for CPU-only bootstrap/report work.

The target is roughly 2–3 hours of pod wall-clock time after local preparation, not a guarantee or a rule to cut training/evaluation short. With two GPUs, aggregate GPU-hours are twice the pod duration. Existing 100M training is about 28–30 minutes and a 4096-sequence evaluation about 18–20 minutes; router kernels, tests and transfers add unmeasured overhead. Four evaluations occupy roughly two waves on two GPUs. Diagnose poor throughput early with a bounded smoke measurement; do not start a kernel-optimization campaign or retune science mid-run.

Report current T/H training timings and clearly label any S/D training timings as historical. Current evaluation timings are descriptive workload measurements, including router diagnostics for T/H; they do not isolate production inference overhead. Report wall-clock and aggregate GPU time separately.

Keep checkpoints on persistent storage while running, strict-reopen final T/H checkpoints, export all losses/diagnostics, and independently verify local and persistent checkpoint SHA256 values. Then **stop the shared pod as soon as all its assigned GPU work and necessary exports are complete**, and verify stopped status. A transient utilization dip or one finished arm is not completion of a shared pod. Preserve the persistent volume. On unrecoverable failure, preserve usable artifacts and stop the now-idle experiment pod.

Run statistics, final audit assembly, report generation and Git packaging locally after GPU shutdown. If shutdown or transfer fails, report the exact unresolved resource/artifact accurately. No idle pod should be kept running simply for report writing.

## 12. Artifacts and final response

Use separate new locations:

```text
Results: results/experiment_2d10_retrieval_aware_gating_100m/
Persistent runtime: /workspace/exp2d10_retrieval_gating_100m/
Local new checkpoints:
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d10_retrieval_aware_gating_100m/{T,H}/scientific_cumulative_001300758528.pt
```

Each final checkpoint includes model, complete optimizer/groups/name mapping, scheduler metadata, RNG, loader, counters, original parentSHA, arm, router formula/normalization/init metadata, compatibility-state rules, code commit, replay-ledger hash and terminal identities. Keep the historical S/D controls in their original locations and reference them in the new manifest.

Preserve a compact protocol/config, source/control audit, preflight evidence, replay manifest, training logs, checkpoint manifests, fresh panel/disjointness audit, four loss arrays, gate summaries, paired statistics, memory/runtime record, backup/shutdown evidence and final report. Keep large checkpoints out of Git. Commit and push the implementation and compact results on the new branch with a new immutable final tag `experiment-2d10-retrieval-aware-gating-100m-final`; never move old tags. If 2D10 is already occupied when execution begins, use a distinct descriptive identifier and document it before training without overwriting another experiment.

The final audit must cover source/control identity, exact 191-update replay, optimizer/counter exceptions, initialization semantics, causality, empty-memory handling, strict reopen, terminal equality with the historical controls, four same-panel evaluations, adjusted primary analysis, unchanged persistent state, verified backups and stopped pod. Reuse evidence rather than rerunning successful checks merely to increase an audit count.

Begin the final response:

```text
EXPERIMENT 2D10 — RETRIEVAL-AWARE GATING 100M COMPLETE

100M screening classification:
Candidate(s) establishing benefit over D:
Candidate(s) clearing delta_CE=0.0001:
Direct tanh versus softmax conclusion:
Current mature baseline: sealed 2D9 250M Dynamic

S / D / T / H CE and perplexity:
D−T / D−H / T−H: means, raw 95% and adjusted 98.333333% CIs
S-based descriptive comparisons:
Gate summaries; parameter/state costs; measured workload timings:
Source/control/final checkpoint SHAs; terminal replay equality:
Audit, backup status, verified pod shutdown and retained storage:
Git references and full report link:
```

Explicitly state **two new training arms, exactly191 updates /100,139,008 targets each; two reused 100M controls; four final evaluations; no further training launched**. Give one next recommendation and stop. If incomplete, say INCOMPLETE and identify remaining work. Proceed with routine implementation and this defined experiment without repeated confirmation for already-scoped work.
