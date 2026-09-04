# EXPERIMENT 2D9 — MATCHED CONTINUATION TO 250M TOTAL

Please execute the matched 250M continuation recommended in your completed 2D9 report. Continue both trained arms from their own sealed 100M checkpoints, through final evaluation, analysis, backups, GPU shutdown and Git push. This authorizes this continuation only. Stop after reporting its result.

The question is whether further matched training makes Dynamic's architecture benefit large enough to meet our existing adoption rule. Static O1 remains the preferred architecture provisionally until that rule is met.

## 1. Inherit the completed experiment

Use the existing implementation worktree:

`/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b`

The sealed 100M result is commit `482ad55637c2a0adb5c7c268b37c7be243ac15c8`, tag `experiment-2d9-token-conditioned-recurrent-gating-final`. Read its final report, checkpoint manifests and frozen `results/experiment_2d9_token_conditioned_dynamic_recurrent_gating/PROTOCOL.md`.

Inherit that protocol's architecture, numerical behavior, optimizer settings, data semantics, evaluation method, statistics, classification and adoption rules. This continuation changes the source checkpoints, training endpoint, fresh panel and artifact locations. The original fresh-w initialization and 100M stopping instructions do not apply to this resumed stage.

## 2. Resume the correct checkpoints exactly

Resume S from S and D from D. Both source filenames are `scientific_cumulative_001300758528.pt` under:

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating/{S,D}/`

Expected SHA-256 values:

```text
S: 676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b
D: c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44
```

Verify the sources against the sealed manifests. Restore each arm's own model, optimizer groups, moments, individual Adam counters, scheduler metadata, all scientific RNG states and loader cursor. Preserve D's learned g0, w vectors and their moments. Do not restart from the original O1 parent, reset w, copy weights between arms, recreate optimizer state, restart warmup or reseed training.

S and D are now different trained models. Require matched subsequent data, not equality of their weights or optimizer tensors. Reuse the existing implementation evidence; add only targeted resume/reload checks and tests for changed continuation code. Discard any disposable smoke updates before resuming scientific state.

## 3. Exact budget and matched stream

**250M means total 2D9 training per arm, not another 250M.**

```text
Already completed per arm:         191 updates / 100,139,008 targets
Additional budget per arm:         286 updates / 149,946,368 targets
Final total for 2D9 per arm:        477 updates / 250,085,376 targets
Targets per update:                524,288
Source global update:              2481
Continuation global updates:       2482–2767 inclusive
Source inherited target counter:   1,300,758,528
Final inherited target counter:    1,450,704,896
Final checkpoint filename:         scientific_cumulative_001450704896.pt
```

Continue from the shared terminal loader cursor; do not replay the previous 191 updates. Freeze a matched ledger of the next 286 batches and use it for both arms. The source next-global-batch SHA is:

`400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649`

Verify the other source stream identities against the sealed manifests. Record continuation-local and experiment-total counters unambiguously.

Keep B32, T1024, accumulation16 and the inherited global pass cadence: three passes when global update is divisible by 32, otherwise two. The continuation therefore has 277 two-pass and 9 three-pass updates. Three-pass updates are 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720 and 2752. Preserve loss weights and count targets once per update, not per pass.

Keep all optimizer settings unchanged, including base/w LR 3e-5, scalar-gate LR 3e-4 and w weight decay 0. Every active parameter's Adam counter must advance by 286 from its own source value; D's w counters reach 477. Dormant B6 parameters and optimizer state remain unchanged.

Commit the continuation protocol/config and tested driver before scientific training. Keep the attention/gating kernel unchanged. Use complete-state recovery if interrupted, without duplicating updates or rewinding a successfully completed arm.

## 4. One fresh final panel, exactly three evaluations

Before continuation training, freeze a new panel of 4096 sequences × 1024 targets using the established canonical B64 validation batches and verified validation shard. Use `numpy.random.default_rng(20260907)` in a separate RNG instance; select the first 64 eligible batches from its seeded permutation.

Inherit all historical exclusions and add the completed 2D9 100M panel. Exclude overlapping target spans and save disjointness evidence. If a genuinely fresh panel cannot be formed, report the prerequisite failure. Never select or enlarge the panel based on scores.

Evaluate only:

1. STATIC_REAL: S's final 250M checkpoint.
2. DYNAMIC_REAL: D's final 250M checkpoint with its learned dynamic gates.
3. DYNAMIC_STATICIZED: the same D checkpoint with the entire w term functionally omitted, retaining D's final g0 and all other weights.

Use the established true incremental evaluator, BF16 execution, FP32 token CE and FP64 accumulation. Reset all recurrent/cache state between sequences and conditions. Staticized must generate its complete trajectory under staticized behavior; it cannot reuse Dynamic-Real caches. Do not mutate or refit the checkpoint for this control.

Save all ordered sequence losses and identities. No midpoint validation, old-panel reevaluation, fourth condition, extra seed, per-link ablation or benchmark suite.

## 5. Preserve the decision rules

```text
A = CE(Static) − CE(Dynamic)
P = CE(DynamicStaticized) − CE(Dynamic)
R = CE(Static) − CE(DynamicStaticized)   [descriptive; A = R + P]
delta_CE = 0.0001
```

Use the same 50,000 paired sequence bootstrap resamples, RNG seed 20260906, shared resampling indices across A/P/R, and 95% linear-percentile intervals. Report means, confidence intervals, perplexities, ratios, win counts and all original strict-boundary flags. Apply the original ordered classification table unchanged.

Recommend Dynamic only if A's lower confidence bound exceeds +0.0001 AND P's lower confidence bound exceeds 0, with passing integrity and zero persistent-state growth. Otherwise retain static O1 provisionally. Statistical utility and architecture adoption remain separate decisions.

Compare the 100M and 250M contrasts descriptively. Their panels differ: do not compare absolute CE as a learning curve, pool the losses, or call the contrast difference a paired test of effect growth. These intervals measure sequence uncertainty for these training trajectories, not replication across training seeds. Staticization removes both average shift and variation from w; it does not test the best refitted constant gate.

Collect the same compact FP32/BF16 gate statistics during DYNAMIC_REAL, with no extra evaluation. Report changes from the 100M g0 and w norms. Keep persistent-state accounting explicit: expected 33,289,728 bytes/sequence in every condition, delta 0; D still adds only 2304 parameters. Treat evaluation timings with gate collection as descriptive.

## 6. Finish and stop cleanly

Reuse the established GPU workflow and retained volume `yhzyb27fb5` where available, checking current resource status and shutdown access. Prefer at most two independent A100-80GB workers, one per arm; sequential execution on one is valid. Reuse completed audits and cached data. Preserve all 100M checkpoints, results and tags.

Store this stage separately, for example under `results/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/`, with corresponding separate checkpoint archives. Use a new `codex/` branch and a new 250M final tag. Do not move the sealed 100M tag.

Strict-reopen both final checkpoints, verify exact counters and matched final loader/next-batch/stream identities, and independently verify local and persistent backup hashes. Export all three evaluations and gate statistics, then stop the exact experiment GPU resources and verify they are stopped. Retain the persistent volume. Perform bootstrap analysis, reporting and Git packaging on CPU after shutdown.

Complete a focused final audit, commit and push the continuation implementation/protocol, compact results and report. Keep large checkpoints out of Git. Report any unresolved failure accurately rather than claiming completion.

Begin the final response with **EXPERIMENT 2D9 — 250M MATCHED CONTINUATION COMPLETE**, followed by classification, preferred architecture, three CEs, A/P/R with intervals, gate statistics, memory/runtime, source/final checkpoint SHAs, stream equality, audit, backup/GPU status, Git references and report link.

Explicitly state **286 additional updates / 149,946,368 additional targets per arm; 477 updates / 250,085,376 total 2D9 targets per arm**. Give one next recommendation without executing it. No automatic 500M extension or further experiment. Proceed through this defined continuation without repeated confirmation for work already in scope.
