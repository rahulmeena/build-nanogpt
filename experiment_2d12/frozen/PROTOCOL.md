# Experiment 2D12 — H10B recurrence ablations, zero training

Prepared 2026-09-06 for handoff to the task that conducted 2D11. This document specifies a new experiment; preparing it has not launched a GPU or performed any of the scientific evaluations below.

Please implement and execute this protocol through verified export, GPU shutdown, local analysis and final reporting. Complete local preparation before starting paid compute. The scope is exactly five conditions of one frozen H10B checkpoint. Do not launch the proposed approximately 500M from-scratch local-only control as part of this experiment.

## 1. Question and fixed conditions

Measure how much the trained H10B model's incremental prediction quality depends on its recurrent branches when each is structurally removed, individually and together. All five conditions load exactly the same trained weights:

| Label | B1 recurrence | B3 recurrence | B5 recurrence |
|---|---|---|---|
| H_ON | Original | Original | Original |
| H_B1_OFF | OFF | Original | Original |
| H_B3_OFF | Original | OFF | Original |
| H_B5_OFF | Original | Original | OFF |
| H_ALL_OFF | OFF | OFF | OFF |

This is inference-only intervention on the final H10B checkpoint, not fresh initialization. There are zero optimizer updates and zero new training targets. No GPT-2 evaluation, HellaSwag, additional gate variants, seeds, retraining, gate calibration or continuation is included. All five conditions must finish regardless of how good or bad any preliminary scores look, subject to technical integrity and the billing ceiling.

## 2. Immutable source and provenance

Workspace: `/Users/rahul/Documents/GPT-2 Enhancement`.

Use the completed 2D11 implementation at:
`/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d11_from_scratch`.

Sealed source commit: `ca0103cd3aee5326855f4e72d1d6db93ce073550`.
Sealed tag: `experiment-2d11-from-scratch-h-vs-gpt2-10b-final`.

Use this H10B checkpoint:
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u19072.pt`.

Required SHA-256:
`bdcbe4ceede0af61b0cea177d677c31b4b26d3f002a3aabdba1f3bde6810011b`.

Expected completed updates: 19,072. Historical logical training targets: 9,999,220,736. Expected registered parameter count: 124,697,386, including the four unused compatibility scalars. Load the model strictly through the 2D11 FreshH implementation, preserving tied embedding/output weights. The old 2D10 adaptation initialization is not a substitute.

Previously recorded persistent checkpoint location:
`/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u19072.pt`.
Resolve its current mounted location and verify its content hash; do not assume an old pod path remains accessible unchanged. Existing volume ID: `yhzyb27fb5`; preserve it and its contents without resizing or deleting anything.

Create an isolated worktree from the sealed source, on branch `codex/experiment-2d12-h10b-recurrence-ablation`. Use new experiment/output directories. Preserve the 2D11 worktree, original code, checkpoints, reports and tag. Freeze the new protocol, code commit, checkpoint identity and intervention manifest before final-panel scoring.

## 3. Exact definition of recurrence OFF

At each selected destination, replace the branch mixture before the shared output projection:

```text
Original: A = lambda_L * A_L + lambda_R * A_R
OFF:      A = A_L
          effective (lambda_L, lambda_R) = (1, 0)
```

Apply the original shared `c_proj`, including its bias, exactly once. Preserve both residual additions, LayerNorms, MLPs, embeddings, positions and every parameter tensor.

OFF bypasses both the disabled recurrent read and the retrieval-aware router at that destination. Do not retain an A_R-dependent local coefficient. Do not merely set A_R to zero while retaining the learned router, subtract the weighted recurrent term, zero the old g_rec scalar, or change softmax logits to a finite approximation of infinity. The intended output is exactly the local branch output before projection.

**Implementation trap:** the sealed H special-block methods accept legacy `gate_override` arguments but do not use them in H's `combine` computation. H also ignores the old scalar g_rec parameters. Implement an explicit disabled-destination set and verify its behavior in the actual incremental evaluator. Apply the same definition in the parallel path used for local reference checks, without changing the unmodified ON computation.

Keep all native windows and all enabled recurrent eligibility unchanged:

| Destination | Native W, including current token | Enabled recurrent source | Enabled recurrent lags |
|---|---:|---|---|
| B1 | 2 | B12 post-MLP residual | 1–1023 |
| B3 | 32 | B10 post-MLP residual | 31–1023 |
| B5 | 64 | B8 post-MLP residual | 63–1023 |

All other blocks retain W1024. OFF does not widen any window. Preserve existing writer-ring and local-cache allocation/bookkeeping for this study, including rings unused by an OFF condition; skip the disabled read/router computation. This avoids introducing a separate cache-storage redesign. Do not claim these retained buffers represent the minimum memory required by an optimized local-only deployment.

Each condition must process its entire sequence under its own intervention, with independent caches initialized empty. Its altered hidden states become its own future writer states. Never inject ON writer values, gates or caches into an OFF trajectory. Enabled branches retain the original formulas and trained parameters but may produce different states and gate values in response to the intervention.

## 4. Local preparation and required verification

Before paid startup, finish the implementation, CPU checks, frozen panel, analysis code, resumable evaluation runner, artifact-export mechanism and stop controller. Inspect applicable repository instructions. Record actual test results and distinguish local checks from pending CUDA checks.

Required focused checks:

1. Empty intervention set reproduces the sealed H implementation, using identical weights and inputs. Verify logits and per-sequence CE, including the incremental path.
2. At a disabled special block, the attention mixture equals A_L exactly. Full block output agrees with an independently written local-only reference. Verify that projection/bias is applied once and residual/MLP behavior is unchanged.
3. Changing the disabled recurrent bank's values or availability mask cannot change that block's output, for otherwise fixed input/cache. Exercise empty and nonempty banks; prove the disabled router is not invoked. Apply this at B1, B3 and B5. These are block-level independence checks, not claims that enabling other recurrent paths cannot change a whole-model trajectory.
4. Preserve masks, writer identities, native cache capacities and causal boundaries at lags 1, 31, 63 and 1023. Verify suffix causality, batch-row isolation, fresh-state resets, and cache ownership for every condition. Exercise actual full-length ring bookkeeping with a small-width fixture where useful.
5. H_ALL_OFF agrees with a reference backbone using the same weights and W2/W32/W64 local attention, with other blocks W1024 and no routing. Verify incremental versus causal-parallel agreement for this fully local reference. Do not require finite two/three-pass H training output to equal recurrent incremental inference; these are different execution modes.
6. Hash model tensors before and after all evaluations. No parameter changes, optimizer construction/steps, backward calls or scientific checkpoint rewriting are permitted in this experiment.
7. Test five-condition ID coverage, paired alignment, bootstrap arithmetic/signs, interval flags, atomic output writes, resume deduplication, and stop behavior on fixtures. Preflight must use synthetic inputs or previously used validation examples, never the new final panel.

Set numerical tolerances before scientific scoring. For direct ON-versus-sealed comparisons under identical operations, require exact equality. For independently written FP32 reference paths, start with atol 2e-6 / rtol 2e-5 and report actual maxima. For CUDA BF16 implementation-versus-reference comparisons, diagnose expected arithmetic-order differences on disposable inputs and freeze justified tolerances before scoring; do not relax tolerances after inspecting final-panel effects.

## 5. One genuinely fresh paired panel

Use the established validation shard `edufineweb_val_000000.npy`, SHA-256:
`8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`.

Known local copy:
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy`.

Freeze exactly 64 complete canonical B64 batches: 4,096 sequences × 1,024 next-token targets = **4,194,304 scored targets per condition**. Use the 2D11 target indexing: for sequence input start s, inputs are `[s,s+1024)`, targets `[s+1,s+1025)`. Verify every token hash.

Construct the exclusion union from 2D11's sealed `panel_disjointness.json` and its source manifests, plus its `monitor_panel.json`, `fresh_panel.json`, and any newer used/reserved panel manifests discoverable in the project. Preserve previously reserved target spans as exclusions. Audit target-span overlap rather than trusting labels or numeric sequence IDs alone. Do not rerun the old preparation entrypoint if it would overwrite frozen 2D11 files or omit 2D11's own panels.

A prior local review found 918 eligible canonical B64 batches after the then-known exclusions; this is evidence of feasibility, not a new reservation. Recheck the current union. Use an isolated NumPy `default_rng(20260921)` permutation of all complete canonical batch indices, accepting the first 64 whose target spans do not intersect exclusions. Preserve accepted order; do not select based on text, losses or preliminary effects. Record NumPy version, seed, exclusions with source hashes, IDs, offsets, sequence token hashes, ordered batch indices and overall panel identity. Verify unique sequences and absence of overlapping selected target spans.

Use this same panel for all five conditions. The previous H10B CE of approximately 3.01495 belongs to another panel and cannot serve as this experiment's ON reference. If the required panel cannot be constructed, report the exact missing-data/exclusion issue during local preparation rather than silently reusing data or reducing sample size.

## 6. Scientific evaluation and restart behavior

Reuse 2D11's true incremental next-token CE semantics: model.eval(), inference_mode, BF16 CUDA forward, explicit FP32 token cross entropy and FP64 accumulation. Score all 1,024 positions of every sequence, including early positions with no eligible recurrence. No warmup targets are discarded. There is no two/three-pass weighted objective during this evaluation.

Default physical batch size is the previously validated B128 on A100 80GB. Confirm memory fit, row isolation and speed during a bounded disposable preflight; if necessary choose B64 or B32 based only on these checks. Freeze one common batch size and numerical configuration for all five conditions before final-panel scoring. Physical batch size is separate from canonical B64 panel identity. Process conditions in the order H_ON, H_B1_OFF, H_B3_OFF, H_B5_OFF, H_ALL_OFF, using identical panel order and fresh caches.

Run without full gate-statistics collection or attention dumps. Record the effective intervention configuration and focused preflight checks instead. Audit the final full-length cache state: with the retained bookkeeping, expected BF16 persistent state is 33,289,728 bytes per sequence for every condition. Timings are descriptive and must specify skipped recurrent read/router work and retained cache buffers.

Atomically persist each completed batch with condition, sequence IDs, input/panel hashes, NLL sums, target counts, CE, source-checkpoint hash and code/config identity. Export completed batches while later batches run. On an interruption, resume only missing complete batches from empty sequence state; never merge stale configuration/panel results or duplicate IDs. Corrupt/incomplete batches may be rerun, with the retry and extra compute recorded. Do not rerun a completed evaluation to search for preferable numbers.

Successful scientific coverage is exactly five conditions × 4,096 sequences = **20,480 sequence-condition results / 20,971,520 target predictions**, excluding disposable preflight and explicitly recorded retries. Save all per-sequence values; an aggregate loss alone is insufficient.

## 7. GPU use, billing ceiling and shutdown

Use **one A100 80GB GPU by default**, or a compatible one-GPU option with validated BF16 behavior and a current quote within the following ceiling. Do not resume a four-GPU pod just to use one device. If no suitable one-GPU resource can be assigned, finish local preparation and present the concrete resource issue; changing the GPU count is outside this default execution scope.

The proposed handoff budget is **at most 3 cumulative billed GPU-hours and USD 10 compute cost, whichever is reached first**, across startup, preflight, scientific work, retries, transfers and stop latency. Existing retained storage charges are separate; do not create/resize storage or increase the cap automatically. Check and record the actual quoted rate before startup. The prior four-A100 preflight estimated approximately 270 seconds for H's distributed fresh-panel evaluation, suggesting roughly 90 minutes for five full-H-equivalent evaluations on one similar GPU before overhead. This is a historical planning estimate, not a runtime guarantee; OFF read skipping may shorten it.

Complete local work first. Stage the code/data or prepare an automatic minimal staging command. Avoid copying the entire repository, training dataset or optimizer backups onto a new paid instance. Reuse verified persistent checkpoint/data copies when available. If extracting a model-only staging artifact locally, record its own file hash and prove tensor identity with the required full checkpoint.

Start with bounded CUDA preflight, targeted at no more than 10 minutes, then begin the scientific queue immediately. Do not perform reports, bootstraps, plotting, Git operations or wait for user discussion while a pod is billing. If preflight or setup needs extended repair, preserve diagnostics and stop the pod before continuing locally.

Before paid startup, prepare the stop command/controller, verify provider credentials and stop capability through the existing interface, and arm an independent local deadline/progress watchdog. On resource allocation, register the returned pod ID with that watchdog immediately. Capture provider identity and initial stopped/running state without exposing credentials. Freeze the deadline from the actual price and remaining cumulative budget. Leave at least 10 minutes of budget for output verification and stopping; enforce the hard ceiling independently of the scoring process. Detect stalled progress with stage-appropriate bounded timeouts, rather than treating a live process as evidence of progress.

Raw GPU outputs must have two independently hash-verified durable copies: local archive and existing persistent storage. Because the original checkpoint is unchanged and already backed up, do not re-export the 1.65 GB checkpoint at the end. Stop the GPU immediately after the final required raw-output export/verification, or on unrecoverable failure/budget exhaustion, while preserving the volume. Verify provider status as stopped/EXITED and record when billing ended; a finished Python process is not proof of shutdown. Analyze partial results only as incomplete if the full five-condition coverage was not achieved.

## 8. Frozen local analysis

Perform analysis after verified GPU shutdown. Align all conditions by exact sequence identity. For condition c, compute overall CE as total NLL / total target count, and perplexity as exp(overall CE), not mean per-sequence perplexity.

The four primary paired contrasts are:

```text
d_B1[i]  = CE_H_B1_OFF[i]  - CE_H_ON[i]
d_B3[i]  = CE_H_B3_OFF[i]  - CE_H_ON[i]
d_B5[i]  = CE_H_B5_OFF[i]  - CE_H_ON[i]
d_ALL[i] = CE_H_ALL_OFF[i] - CE_H_ON[i]
```

Positive means removing that recurrence hurts the trained model. Negative means the intervention improves CE.

Use 50,000 paired sequence bootstrap resamples, isolated NumPy `default_rng(20260922)`, FP64 means and linear percentiles. Draw one resampled vector of 4,096 sequence indices per replicate and apply it to all four paired contrasts. Use bounded CPU memory chunks. Freeze the numerical environment locally before scoring.

Report ordinary 95% intervals (2.5th/97.5th percentiles) and **Bonferroni-adjusted 98.75% marginal intervals** (0.625th/99.375th percentiles) for the four-contrast family. Use the adjusted intervals for primary conclusions; do not choose whichever interval is favorable.

Retain delta_CE = 0.0001 nats/target as the project's small-effect reference. For adjusted bounds [L,U], report separate flags:

- removal_hurts: L > 0
- removal_hurts_beyond_reference: L > +0.0001
- removal_helps: U < 0
- removal_helps_beyond_reference: U < -0.0001
- practical_equivalence_at_reference: L >= -0.0001 and U <= +0.0001

Otherwise the respective claim is unresolved. These flags need not be mutually exclusive: a very small established effect can also lie wholly inside the equivalence band. Failure to establish harm is not evidence of equivalence.

Also report `100 * expm1(mean(d_c))` as the relative perplexity increase under OFF, transforming CI endpoints monotonically, and the fraction of sequences with positive/negative/zero paired differences. Make one four-contrast plot with adjusted intervals and reference lines at 0 and ±0.0001. Do not normalize these effects by the historical G−H gap from another panel or label the result a percentage of H's advantage explained.

Because canonical selection samples contiguous B64 groups, add a clearly labeled sensitivity analysis resampling the 64 canonical groups, keeping their 64 sequences together, with 50,000 paired group resamples and isolated seed 20260923. Use the same interval levels; it does not replace the prespecified sequence analysis. Flag conclusions that change with the sampling unit. Neither analysis measures variation across independent training runs or guarantees independence across documents.

## 9. Interpretation and deliverables

The main conclusion is the model's dependence on recurrent branches **under the specified local-only intervention**. OFF both removes recurrent content and restores the local mixture coefficient to one; it isolates complete branch removal, not recurrent content while holding learned mixture weights fixed.

Individual effects need not add up to ALL_OFF. Removing one branch changes later representations and can change the enabled gates. Report descriptive nonadditivity without adding unplanned confirmatory claims. Small mean gate coefficients are not percentages of predictive contribution or required fractions.

A harmful frozen intervention does not establish how poorly a model trained without recurrence would perform. A harmless intervention does not show recurrence provided no training benefit. The attached CE2/CE3 temporal gradients were verified in the preceding audit, but this inference-only study does not measure their training contribution. Recommend any later matched from-scratch control as a separately scoped experiment; do not launch it automatically.

Deliver a self-contained final report with all five CE/perplexity scores, all four paired contrasts and both interval levels, flags, sensitivity results, caveats, and the concrete next recommendation. Include source/checkpoint/panel/intervention identities, implementation/reference tests, unchanged-weight verification, sequence/target coverage, raw per-sequence results, bootstrap settings, numerical tolerances, actual hardware/batch size, per-condition timing, retries, actual billed wall/GPU hours and compute cost. Separate measured scientific work from staging/preflight/transfers/idle when logs support it, and explicitly mark unmeasured overhead.

Save at minimum: PROTOCOL.md, SOURCE_IDENTITIES.json, INTERVENTIONS.json, PANEL.json, PANEL_DISJOINTNESS.json, PREFLIGHT_AUDIT.json, per-condition raw results, PAIRED_ANALYSIS.json, contrast CSV/plot, ARTIFACT_HASHES.json, RUNTIME_ACCOUNTING.json, STOP_VERIFICATION.json, FINAL_AUDIT.json and FINAL_REPORT.md. The final audit must verify all five conditions, zero scientific training updates, exact target coverage, backup hashes and stopped provider status.

Commit the new experiment code, frozen protocol and appropriately sized results to its new branch, push it and tag `experiment-2d12-h10b-recurrence-ablation-final` only after successful final audit. Verify remote commit/tag identities. Keep large checkpoints out of Git and leave the original 2D11 tag unchanged. If incomplete, report the blocking condition and saved partial progress without creating a misleading final-success tag.
