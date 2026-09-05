# EXPERIMENT 2D10 — H-ONLY MATCHED CONTINUATION TO 250M TOTAL

Please execute this continuation through training, evaluation, verified backups, GPU shutdown, local analysis and Git push. Train only H. Reuse the sealed 2D9 250M Dynamic checkpoint as the matched control. Stop after this result; no further training or experiment is authorized by this handoff.

## 1. Purpose and inherited protocol

H won the 2D10 100M screen against D and T beyond the prespecified practical margin. This continuation tests H against the mature Dynamic model after the same total 250M adaptation exposure from original 2D7 O1.

Use the worktree `/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b` and the sealed 2D10 implementation/result at commit `ef9d33ca5ad89a1b0b18db159310fc3e3dbe3a9c`, tag `experiment-2d10-retrieval-aware-gating-100m-final`. Start a new `codex/experiment-2d10-h-250m` branch. Preserve the sealed 100M result and tags.

Read the 2D10 100M protocol/report/manifests and the review erratum at:

`/Users/rahul/Documents/GPT-2 Enhancement/project_context/EXPERIMENT_2D10_REVIEW.md`

Inherit the implemented H architecture, all numerical settings and optimizer semantics. This stage changes the resume checkpoint, endpoint, replay ledger, panel and analysis to a single H-versus-D comparison. Do not rerun the old fresh-router initialization or reuse the old analysis entrypoint with hardcoded 100M paths. Adapt driver and provenance checks for a genuine H continuation without changing the attention/router kernel.

## 2. Exact source and read-only control

**H training source: its own sealed 100M checkpoint.**

```text
Local:
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d10_retrieval_aware_gating_100m/H/scientific_cumulative_001300758528.pt

Persistent:
/workspace/exp2d10_retrieval_gating_100m/run/checkpoints/H/scientific_cumulative_001300758528.pt

SHA256:
d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c

Global update: 2481
Inherited cumulative targets: 1,300,758,528
H adaptation so far: 191 updates / 100,139,008 targets
```

**D control: the already-trained 250M Dynamic checkpoint. Evaluate only.**

```text
Local:
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/D/scientific_cumulative_001450704896.pt

Persistent:
/workspace/exp2d9_dynamic_gating_250m/run/checkpoints/D/scientific_cumulative_001450704896.pt

SHA256:
9714b2e3f53a8c15dfecfed3e9b56c358176c1f9f609bcce7e28c35b8a358a9b

Global update: 2767
Inherited cumulative targets: 1,450,704,896
D adaptation total: 477 updates / 250,085,376 targets
Sealed control result commit: adec918f436d264840e5b60d60a589463bb1bca9
```

Verify checkpoint identities against the original manifests. Resolve relocated persistent paths only with unchanged content hashes. Do not restart H from O1, load D weights into H, reinitialize any router tensor, or retrain D, T or S.

Restore H's exact model, optimizer groups and moments, individual Adam counters, scheduler metadata, all scientific RNG states and loader cursor. Its W1, b1, W2 and b2 are already trained parameters: preserve all of them and their optimizer state. No parameters or fresh optimizer states are added during this stage.

## 3. Exact budget and historical replay

**250M means total H adaptation, not another 250M.**

```text
Already completed:               191 updates / 100,139,008 targets
Add now:                        286 updates / 149,946,368 targets
Final H total:                  477 updates / 250,085,376 targets
Targets per update:             524,288
Continuation global updates:    2482–2767 inclusive
Final inherited target counter: 1,450,704,896
Final filename:                 scientific_cumulative_001450704896.pt
```

Replay the exact 286-batch continuation already used by D in:

`results/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/MATCHED_BATCH_LEDGER.jsonl`

Expected ledger file SHA256:

`0875d5533a4a8ae753f2e0aec661d81f314609c16d4de053a6ffd48df8e751ec`

Reference its `CONTINUATION_MANIFEST.json`. Treat the old ledger as a historical data artifact and record H's actual parent separately; do not mislabel H as a former S/D checkpoint or rewrite historical provenance. Match every logical batch, stream, cursor, pass count and target count. Do not replay H's first 191 batches or continue past the specified endpoint.

Expected source handoff:

```text
Cursor SHA:     d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d
Next batch SHA: 400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649
Next stream SHA: 0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642
```

Expected final identities, matching D250M:

```text
Cursor SHA:     f549cf0a45863e65391147d4439c1ae0af2ba1c8bc1d943e1a33f532c7b3d1d0
Next batch SHA: c1d217a6af6379263b035c47a158bdeada49f9611cb2fa721ae6c939aa35fe27
Next stream SHA: c9025e28b5ef35a00c35fde4e89d38da8bfdae50235211d88516f14abebb4196
```

Keep B32×T1024 and accumulation16. Preserve global cadence: three passes when global update is divisible by 32, otherwise two. This stage has 277 two-pass and 9 three-pass updates at 2496,2528,2560,2592,2624,2656,2688,2720,2752. Loss weights stay (.25,.75) and (.20,.40,.40). Count training targets once per update, not once per pass. Clearly distinguish continuation-local updates 1–286 from H-total updates 192–477.

## 4. Preserve H's trained behavior and optimizer exceptions

Keep O1 geometry, all source identities, masks and cache capacities unchanged: B1 W2/B12 lags1–1023; B3 W32/B10 lags31–1023; B5 W64/B8 lags63–1023; all other blocks W1024; no B7→B6 or lag0.

H continues to use separate local/recurrent attention outputs, router input `[LN(q);LN(A_L);LN(A_R)]`, head concatenation in c_proj order, affine-free FP32 LayerNorm with epsilon 1e-5, width 32 SiLU hidden layer, two softmax outputs and a final BF16 coefficient cast. Combine the original branch outputs and apply shared c_proj/bias once. Keep exactly local-only behavior when recurrent memory is unavailable, determined by valid masks. Keep attached gradients and current-invocation inputs in both parallel and incremental execution.

Preserve all optimizer settings, including backbone/router matrix LR 3e-5 and weight decay 0.1, hidden bias LR 3e-5/decay 0, output bias b2 LR 3e-4/decay 0, Adam (.9,.95), epsilon 1e-8 and global clip 1.0. Preserve inherited constant scheduler metadata, BF16 autocast and FP32 masters; no warmup restart or tuning.

H's retired `g_rec`, `g_rec_b3`, `g_rec_b5` and dormant `g_rec_b6` must remain computationally unused, with no gradients, parameter changes, weight decay or optimizer-state/counter changes. Verify their complete state against H100M. Every active parameter counter advances 286 from its own source value. Router counters reach 477. Do not require all old counters to equal the global update.

Registered H parameters remain 124,697,386. D has 124,478,212: H adds 219,174 registered parameters  / 876,696 FP32 bytes versus D. No further parameters are added by this continuation. Expected persistent state remains 33,289,728 BF16 bytes per B=1 sequence in both conditions. Unexpected state growth is a discrepancy for this unchanged geometry, not a new architectural choice. The user's willingness to trade memory for future GPT-2 performance does not authorize an unplanned change here.

## 5. Bounded preflight and the report-table correction

Prepare locally/CPU before GPU rental where practical. Reuse the sealed scientific tests and run targeted checks for changed resume/analysis code, source/control loading, complete-state restoration and a short disposable update/reload smoke. Verify H100M reload equivalence to itself, not zero-effect equivalence to O1. Discard smoke state and independently reload exact H100M scientific state before training.

Fix the known flag-table renderer in the new code revision: `v.values()` must not determine column order after JSON is saved with sorted keys. Render the explicit keys:

```python
flag_columns = (
    "positive", "beyond_margin", "negative", "material_harm",
    "practical_equivalence", "second_condition_noninferiority",
)
```

Add a small JSON-roundtrip/report check so displayed labels agree with numerical flags. Record the old 100M table erratum, whose D−H/T−H margin flags should be True and harm flags False. Preserve the old numerical artifacts, report and immutable tag; do not rerun or overwrite the sealed 100M analysis simply to produce the new report.

Commit the continuation implementation/config/protocol and frozen panel before scientific training. No attention/router changes mid-run. Keep per-update finite-gradient, counter, replay and loss records. One complete recovery checkpoint near continuation update 144 is optional. Checkpoint writing must preserve scientific RNG. On interruption, resume from a verified complete state without duplicating updates. On unresolved source/replay corruption, leakage or NaN/Inf, preserve artifacts and report incomplete.

## 6. Exactly two final evaluations on one fresh panel

Before continuation training/scoring, freeze one panel of 4096 sequences ×1024 targets =4,194,304 targets per condition, using 64 canonical B64 batches from the established validation shard. Verify shard SHA:

`8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`

Use isolated `numpy.random.default_rng(20260912)`: permute complete canonical batches and select the first 64 eligible batches. Apply the historical reserved prefix and all recoverable exclusions, explicitly including both 2D9 panels and the 2D10 100M panel `722cbffc8ab96c42137c174672849ccf50a0f0527055a28265391e95aaea8b2b`. Save ordered identities, manifest and target-span disjointness evidence. No score-based selection, reuse, enlargement or reselection. If a fresh panel cannot be formed, report a prerequisite failure.

Evaluate exactly:

1. D_REAL: sealed 2D9 D250M, unchanged, read-only.
2. H_REAL: new final H250M checkpoint.

Use the established true incremental evaluator, BF16 execution, FP32 token CE and FP64 accumulation. Reset all model/cache state between sequence batches and conditions; strict-load each model in its correct architecture. Record two arrays of 4096 finite ordered sequence losses, target counts, checkpoint/condition/panel hashes and physical cache audits. Verify D's tensors/file remain unchanged.

Collect the same compact H gate statistics during its one evaluation: b2, router norms, logit differences, lambda_L/lambda_R, entropy, eligible versus unavailable positions, FP32 and actual BF16 coefficients, full-panel moments/extrema and fixed-sample quantiles. Use an isolated quantile seed 20260914 and at most 131,072 eligible positions per destination. Transfer once per batch, with no attention matrices or per-token CPU synchronization. Report H100M-to-H250M parameter changes and describe different-panel gate-distribution changes accordingly.

No T/S evaluation, Staticized/constant-router control, extra panel, midpoint validation, HellaSwag, window changes, additional seeds or from-scratch comparison in this stage.

## 7. One primary contrast and a frozen adoption rule

```text
A = CE(D250M) − CE(H250M)
delta_CE = 0.0001
Positive favors H.
```

On local CPU, use 50,000 paired sequence bootstrap resamples with isolated RNG seed 20260913 and NumPy linear-percentile 95% CI. Pair sequences before resampling; never bootstrap arms or tokens independently. Chunk memory use and record library version/method.

This stage prespecifies one primary comparison on a new panel, so use ordinary 95% bounds. The prior screen's 98.333333% intervals addressed three comparisons and must not be silently carried into a differently labeled 95% table. Do not pool 100M and 250M losses or treat the different-panel contrast change as a paired test of growth.

Report mean, 95% bounds, perplexities, exp(A), wins/ties and strict flags: positive L>0; beyond-margin L>+0.0001; negative U<0; material harm U<−0.0001; practical equivalence L>−0.0001 AND U<+0.0001; H noninferiority L>−0.0001. Touching a boundary does not pass. Statistical utility, practical equivalence and architecture adoption are separate statements.

Recommend **H as the preferred tested 250M architecture and next baseline only if A's lower 95% bound exceeds +0.0001 and all integrity checks pass**, including unchanged prescribed persistent state. Report measured parameter/runtime costs. Otherwise retain the sealed 250M Dynamic baseline and state whether H is statistically better, equivalent, harmful or unresolved under the flags. Do not extend training/evaluation to force a decision.

This evaluates complete trained architectures. It does not isolate the contribution of retrieved-output inputs, token variation, mean scaling or initialization. No same-checkpoint router-ablation utility claim is available. Evaluation-sequence CIs are not training-seed replication or evidence of superiority to an independently trained Karpathy GPT-2 baseline. If H is adopted, the user's later from-scratch baseline comparison is a reasonable next recommendation, not an automatic launch.

## 8. One GPU, prompt shutdown and durable artifacts

Use one A100-80GB GPU for H training and the two evaluations. Reuse compatible cached data and retained volume `yhzyb27fb5` where available; verify live resource/mount/shutdown capability. Avoid resuming a two-GPU allocation merely to leave a GPU idle. Historical stopped-pod IDs are not current resource instructions. Do not retrain controls.

Prepare code, data manifests and panel locally before paid GPU time where practical. Historical H throughput suggests roughly 45 minutes for the extra 286 updates, followed by roughly 40–45 minutes for two evaluations; setup, checks, router behavior and exports add time. These are planning estimates, not a deadline or reason to alter the scientific budget.

Suggested separate locations:

```text
Results: results/experiment_2d10_h_250m/
Persistent runtime: /workspace/exp2d10_h_250m/
Local final H checkpoint:
/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d10_h_250m/H/scientific_cumulative_001450704896.pt
```

The final checkpoint must include the complete model/optimizer/group-name mapping, scheduler, RNG, loader, counters, source H100M SHA, original O1 ancestry, architecture and retired-state rules, code commit, historical replay-ledger hash and terminal stream identities. Strict-reopen it and verify terminal equality with D250M. Preserve all historical checkpoints and the D control in their existing locations.

Export the checkpoint, both evaluations and required diagnostics; independently verify local and persistent checkpoint/artifact hashes. **Stop the exact experiment pod as soon as assigned GPU work and necessary verified exports are complete**, then verify actual stopped status and retain persistent storage. An unrecoverable failure also requires preserving available artifacts and stopping the experiment's idle GPU. Do not stop during a temporary utilization dip while a required job is still running.

Perform bootstrap/statistics, final audit assembly, reporting and Git packaging locally after shutdown. Keep the artifact set compact: protocol/config, source/control and resume audits, replay/training records, panel/disjointness, checkpoint manifests, two loss arrays, gate statistics, paired analysis, memory/runtime, backup/shutdown evidence and final report. Correctly label D training times as historical and H evaluation times as including diagnostics; these do not isolate production inference overhead.

Commit and push the continuation code and compact results on the new branch, with a new immutable `experiment-2d10-h-250m-final` tag. Do not move old tags or put large checkpoints in Git. If sync/shutdown/export fails, report that limitation accurately rather than claiming complete success.

## 9. Final response and stopping point

Begin with **EXPERIMENT 2D10 — H-ONLY 250M CONTINUATION COMPLETE** and report:

- Classification and preferred architecture; whether H clears the practical margin over D.
- D/H CE and perplexity; D−H mean and 95% paired CI; all named flags rendered in explicit order.
- Descriptive comparison with the 100M D−H result, with the different-panel limitation.
- H gate statistics, parameter/state costs, training/evaluation/pod timings.
- Source H100M, final H250M and reused D250M checkpoint SHAs and paths.
- Replay/terminal equality, final audit, verified backups, stopped-pod status and retained volume.
- Git references and full report link; one next recommendation without executing it.

Explicitly state **H only: 286 additional updates  / 149,946,368 additional targets; 477 updates  / 250,085,376 total adaptation targets. One reused 250M Dynamic control; exactly two final evaluations; no further experiment launched.** If incomplete, use INCOMPLETE and name the remaining work. Proceed through the defined continuation without repeated confirmation for already-scoped actions.
