# Experiment 2D13 — fresh local-only training versus H at 524M targets

Prepared 2026-09-06 for the task that conducted 2D11/2D12. Preparing this specification has not launched scientific training or a GPU. Please implement and execute the following fixed scope through checkpoint preservation, evaluation, verified shutdown, local analysis and Git archival.

## 1. Scope and comparison

Train **one new local-only model, L**, from the exact untrained backbone tensors used to initialize 2D11 H. Remove all three recurrent reads and their branch routers throughout L training and inference. Compare L with the **already saved H checkpoint at 1,000 updates**. Both models were trained from scratch; reuse means reusing the existing H comparator, not initializing L from trained H weights.

| Condition | Initialization | Training for this experiment | Endpoint |
|---|---|---|---|
| L524M | Original untrained H backbone tensors | 1,000 new updates, local-only | 524,288,000 logical targets |
| H524M | Same original untrained backbone, in completed 2D11 | None; load saved u01000 | 524,288,000 logical targets |

Prefer one GPU. The user also explicitly permits four GPUs if retaining the existing distributed implementation materially simplifies the verified comparison. Choose and document the execution mode during local preparation, before paid startup; implement and test the selected route rather than building unnecessary alternative runners. GPU count does not change the scientific targets or effective batch. No new H/GPT-2 arm, additional seeds, H10B comparison, H524M OFF evaluation, window changes, HellaSwag or extension beyond update 1,000 is included. This is the approximately 500M screen already discussed; its exact endpoint matches the available H checkpoint. Complete all 1,000 updates unless a technical integrity failure or the billing ceiling prevents completion. Monitoring is descriptive; do not introduce an outcome-dependent early-stop rule or tune the recipe after looking at scores.

The question is whether the complete H recurrent training recipe improves early learning over the same local-attention backbone trained without recurrence. This does not isolate recurrent content independently of router capacity, initialization of branch mixtures, or the H multipass objective, and does not settle the 10B comparison.

## 2. Immutable sources and initial weights

Workspace: `/Users/rahul/Documents/GPT-2 Enhancement`.

Reuse code from the sealed 2D12 worktree:
`/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d12_h10b_ablation`.
Commit/tag: `622f34b29ec54816dcf293aef9228939c72a6dd8` / `experiment-2d12-h10b-recurrence-ablation-final`.

The underlying 2D11 source remains:
`/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d11_from_scratch`, sealed at `ca0103cd3aee5326855f4e72d1d6db93ce073550`.

Create a new isolated worktree from the 2D12 sealed commit and branch `codex/experiment-2d13-local-only-from-scratch-500m`. Use a new experiment package and output directory. Preserve both earlier worktrees, their reports, checkpoint files and immutable tags.

**L must start from this untrained artifact:**
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/initial.pt`.
SHA-256: `387b8b791ced54c2f1c0da65c47d00f35e8c3d1514897bad4c70db36db6c5c94`.
Expected size: 498,868,405 bytes; completed_updates=0; loader starts at shard 0, position 0, wraps 0, logical_batches 0.

Expected initial backbone tensor identity, using 2D11's `tensor_identity(base.named_parameters())`:
`7680b0c094254f9b7bd11a9208d753867c9ce68f2efab01cb185934736988851`.

Strictly load these saved model tensors, preserve tied embeddings/output weights, and independently prove backbone identity before training. Do not regenerate the scientific initialization from seed 1337: the earlier run observed platform-dependent seeded generation. Reuse the recorded initial Python/NumPy/CPU RNG state and CUDA seed where applicable, and freeze a new reproducible initial state for the chosen GPU count after constructing L. Disposable preflight must not consume that state.

**The evaluation-only H comparator is:**
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u01000.pt`.
SHA-256: `559af751b9cceda5a4b08ef785d64f015cc4ce1978da7130a687f228f80a3244`.
Expected size: 1,651,147,173 bytes; completed_updates=1000; logical_targets=524288000.
Expected H model tensor identity: `7d0f5d3da3dcc41a5c9e83c4d3a94fa4aa3caa7be226d1d92521d3c2f0f7548a`.

Known persistent H location:
`/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u01000.pt`.
Resolve current mounted paths by provenance/hash rather than assuming an old pod layout. Existing volume ID `yhzyb27fb5` is recorded at 190 GB; preserve it without resizing or deleting historical files. Both local checkpoint file hashes above were independently rechecked while preparing this protocol; the executor must verify actual inputs at use.

## 3. L architecture and trainability

Retain GPT-2's 12 blocks, width 768, 12 heads, head dimension 64, context 1024, padded vocabulary 50304, standard residuals, tied embeddings, normalization/projections/MLPs and all backbone weights. Use the 2D12 structural ALL_OFF definition from step zero:

```text
B1: local attention W2;  no recurrent read or router
B3: local attention W32; no recurrent read or router
B5: local attention W64; no recurrent read or router
Other blocks: original local causal W1024

At B1/B3/B5: A = A_local; effective (lambda_L,lambda_R)=(1,0)
Apply the original shared output projection/bias exactly once.
```

Preserve the 2D12 verified OFF implementation and unchanged ON behavior. Do not use the ignored legacy gate_override/g_rec mechanisms. Do not restore full W1024 at B1/B3/B5. L has no cross-pass or cross-minibatch recurrent information.

For checkpoint compatibility, retain the three router modules and four legacy scalars in the registered schema, but freeze them, exclude them from the optimizer, bypass their computation and verify their tensors remain identical to initialization. Explicitly enable gradients on every backbone parameter. Counts should be:

- L registered parameters: 124,697,386; L trainable backbone parameters: 124,475,904.
- L frozen parameters: 221,482, comprising 221,478 router parameters and four compatibility scalars.
- Historical H trainable parameters: 124,697,382; the trainable difference is 221,478.

Do not describe matching registered counts as equal active model capacity. Keep the existing incremental cache bookkeeping for both endpoint evaluators; unnecessary L writer buffers can remain allocated, as in 2D12. Cache/storage optimization is outside this experiment.

**Loader and helper traps:** 2D12's `load_model()` requires update 19072 and freezes every parameter. It cannot be used unchanged for L training or H524M loading. Build L outside inference_mode so its tensors support autograd; the old 2D12 evaluation entrypoint deliberately forbids backward and optimizer construction. Reuse its verified kernels, not that inference-only entrypoint for training. The inherited multi-pass function accepts only two or three passes: implement L's collapsed forward using the single `forward_pass` path, not an unsupported num_passes=1 call. 2D11's trainer/physical-batch helper, optimizer restoration and checkpoint validator assume four ranks and H's active parameter groups; the validator also assumes every frozen tensor is a zero scalar. Implement explicit L initialization, H1000 evaluation loading and a new L checkpoint schema. Never load H's optimizer groups/moments into L or accidentally train with a frozen backbone.

## 4. Single-pass L objective and required equivalence checks

Use **one actual CE pass per L microbatch**, after validation below. In this dropout-free local-only model, repeated passes see identical tokens and have no usable cross-pass state. Therefore in real arithmetic:

```text
CE1 = CE2 = CE3 = CE_local
0.25*CE1 + 0.75*CE2 = CE_local
0.20*CE1 + 0.40*CE2 + 0.40*CE3 = CE_local
```

This removes redundant computation while preserving L's mathematical weighted objective. It does not make L compute-matched to H. Floating-point reduction/scaling order may differ; do not promise bitwise identical trajectories.

Before scientific training, compare the actual L single-pass implementation with explicit ALL_OFF two/three-pass reference objectives. Check losses, all active parameter gradients, inactive gradients, global clipping and AdamW updates/state. Cover activation checkpointing on/off and both initial and disposable noninitial weights. CPU FP32 checks should use atol 2e-6 / rtol 2e-5 and report actual errors. A local authoring probe on the existing tiny ALL_OFF implementation passed both objectives with checkpointing on/off; largest absolute parameter-gradient error was 5.22e-8. This is documented in `project_context/EXPERIMENT_2D13_LOCAL_PREPARATION_CHECK.json` and does not replace production/CUDA checks.

For same-input CUDA BF16 single-versus-repeated objective checks, prespecify loss atol 1e-5 / rtol 1e-5, and global active-gradient relative L2 error <=0.01 with cosine similarity >=0.9999. Report per-tensor discrepancies, finite/zero patterns, clipping norms and optimizer state/update differences as well; investigate unexplained localized or sign-scale discrepancies even if the global metric passes. Near-zero-gradient tensors need absolute rather than unstable relative diagnostics. These bounds accommodate BF16 weighting order, not missing gradient paths or normalization errors. Validate FP32 mathematical equivalence separately. If required checks fail, stop paid debugging, diagnose locally and report the blocker; do not silently switch objectives, increase tolerances or perform the scientific run with redundant passes.

Use non-reentrant activation checkpointing, as in 2D11, for scientific L training. No dropout, stochastic augmentation, gradient truncation changes, compilation or additional losses are introduced.

## 5. Exact stream, accumulation and optimizer schedule

Each update uses exactly 512 sequences ×1024 targets =524,288 targets. The preferred one-GPU configuration is:

```text
microbatch size = 32 sequences
accumulation = 16 microbatches
loss passed to backward = microbatch mean CE / 16
optimizer steps = 1 after all 16 microbatches
```

Zero gradients once before the update, accumulate all 16, clip once, set the update's LR and step once. On one GPU, do not retain the old loss/4 divisor, rank averaging, four-microbatch loop or DDP requirement.

If four GPUs are selected, retain the verified original geometry: B32 per GPU, four accumulation microbatches per rank, loss/4 before backward and DDP rank averaging exactly once. Wrap L's actual single-pass forward/loss entrypoint in DDP; no_sync must include forward and backward on the first three microsteps, with the fourth synchronized. Only backbone parameters require gradients. Keep all four ranks participating in every scientific training update. Any all_reduce of loss values is for logging, not an additional gradient division.

No LR scaling for GPU count. In either mode, prove that all 512 rows are used exactly once and the normalized gradient matches the original global-mean convention. The local algebra/partition tests can emulate rank grouping without renting four GPUs; actual four-rank CUDA/DDP checks are required if the scientific run uses four. Reduction-order rounding remains documented, and one-versus-four GPU execution is not claimed bitwise identical. Four GPUs do not fix a failed single-pass objective equivalence check; that check applies in both modes.

Use the sealed 2D11 logical loader: eight original B64 batches per update, original shard order and boundary rule, shifted global token views and original sequence boundaries. Follow the first 1,000 entries of `experiment_2d11/frozen/stream_plan.jsonl`. Before training, freeze expected per-update token hashes/cursors from the first 1,000 rows of `experiment_2d11/results/run/metrics-attempt01.jsonl`. Hash every actual L training batch and require identity with the corresponding H batch before applying its update.

Expected terminal cursor after update 1000: shard 5, position24,576,000, wraps 0, logical_batches 8000. Expected final update's combined input/target hash:
`b29fc811de065ce888ea4556365e3f72ac221bf1675769707a6371635ae8ef45`.
Verify the full stream and terminal checkpoint cursor, not only this final hash. Use verified persistent training shards and original metadata; avoid copying the entire training dataset when only the matched prefix is required.

Create a **fresh** fused CUDA AdamW optimizer over L's trainable backbone only: betas (.9,.95), epsilon 1e-8; matrices weight_decay 0.1; vectors/biases weight_decay 0. No state/momentum at update 0. Global gradient clipping is 1.0. Use BF16 autocast, FP32 master parameters, deterministic settings and `torch.set_float32_matmul_precision('high')`, matching the recorded numerical recipe. Record PyTorch/CUDA/device versions; prefer the prior PyTorch 2.8.0+cu128 environment. No torch.compile.

Preserve the original 10B LR schedule prefix exactly: use `experiment_2d11.common.learning_rate(u-1)` for updates u=1..1000. That is 715-update linear warmup to 6e-4 followed by the original cosine with max_steps 19073 toward 6e-5. First LR is 6e-4/715. Do not fit a cosine schedule that decays to its minimum at 500M. Stop after update 1000 with a valid resumable optimizer/LR state, so a separately authorized future continuation can preserve the original schedule.

## 6. Initialization, preflight and resumable checkpoints

Complete code, input manifests, tests, analysis, monitor templates, export queue and stop controller locally. Create and validate L's step-zero checkpoint, containing exact model/frozen tensor identities, fresh optimizer groups, RNGs, initial loader cursor, accounting and next LR index. Inspect applicable repository instructions. Freeze the protocol and code identity before paid use.

Local tests must additionally cover structural OFF independence, all intended backbone gradients, unchanged frozen routers, tied weights, exact stream partition and normalization, LR boundaries at updates 1/715/716/1000, checkpoint resume, expected optimizer step counters and evaluation ID coverage. Reuse relevant 2D12 reference/causality/reset checks. Do not reintroduce extensive unrelated architecture tests.

The bounded preflight uses disposable weights/data and must exercise full-size B32 ×T1024 forward/backward, one complete 512-sequence update under the chosen geometry, both objective equivalence checks, finite gradients and memory fit. For four GPUs, include per-rank gradient/parameter agreement and rank-average normalization; validate full DDP forward semantics. Verify save/reload reproduces the next update and optimizer state within frozen numerical tolerances, including a schedule boundary. Exercise the actual initial tensor loading and both L/H incremental evaluators. Benchmark steady-state update time and representative evaluation work for a conservative total projection. Do not score the new final panel in preflight.

Then restore L's exact scientific step-zero checkpoint, all RNGs and stream position. Scientific training begins at update 1; disposable optimizer steps count only as preflight compute.

Write complete atomic checkpoints every 250 updates, retaining at least the latest two valid recovery checkpoints and permanently preserving u0, u500 and u1000. Reopen and validate each saved file before declaring it usable. Include all model tensors, frozen-tensor hashes, optimizer names/groups/moments/steps, world size and each rank's Python/NumPy/CPU/CUDA RNG state, loader cursor, completed updates, next LR index, one-pass accounting, evaluation ledger and cumulative resource accounting. Use a new L-specific validator; frozen router W1 is a nonzero matrix, not a scalar to test with .item(). In four-GPU mode, all ranks must agree on the scientific state; have one writer/export worker serialize the shared model/optimizer snapshot rather than copying it four times.

Export u0/u500/u1000 and completed evaluation artifacts in the background to local storage while GPU work continues. Maintain a bounded save/transfer queue. Keep two independently hash-verified durable copies of required checkpoints and raw evaluations, on the existing persistent volume and local archive. Resume only from validated checkpoints; record discarded/replayed work and retries separately. Never modify original H/initial artifacts or silently prune historical project files.

## 7. Monitoring and final paired evaluation

Monitor L at completed updates **0,250,500,750,1000**, using the sealed 2D11 monitoring panel: first 20 canonical B64 batches, 1,280 sequences ×1024 targets. Preserve/restore scientific RNG state and training mode around monitoring; evaluation must not advance the training loader. Resume must complete any pending evaluation at its saved update and deduplicate already complete, identity-matched artifacts. Reuse recorded H scores at these points from `experiment_2d11/results/run/evaluations/attempt01/ce_{u}.json` and/or the corresponding analysis CSVs, verifying their identities and scoring mode. There is no new H monitoring job. Plot same-panel CE versus logical targets. H's weighted multipass training loss and L's single-pass loss are different objectives; do not overlay them as interchangeable training losses. Any training-loss plot must separate them or explicitly identify H's CE1/CE2 terms.

After L reaches update 1000, perform exactly two final evaluations: **L524M and H524M**, on one newly frozen 4,096-sequence panel. Use true incremental inference for both, BF16 forward, FP32 token CE and FP64 NLL accumulation; reset caches for every sequence/batch and score all 1,024 targets. Do not use parallel L evaluation against incremental H: 2D12 documented BF16 cross-shape numerical differences large enough to matter here. Preserve the verified independent incremental references and record actual CUDA errors/tolerances; no tolerances are chosen after seeing final effects.

Use a common physical evaluation batch size per GPU, default B128 on A100 80GB; select B64/B32 only if required by prescoring memory/row-isolation checks. Keep the selected setting fixed for both final conditions. If four GPUs are used, distribute every monitoring/final evaluation across all four independent workers with disjoint sequence ownership; do not run each full panel on every GPU or assign only one GPU a complete condition. Freeze a balanced deterministic assignment compatible with the selected batch size, permitting a smaller final batch. Verify merged global counts/IDs and row isolation. No full gate-statistics or attention dumps. Verify each source checkpoint/tensor identity before and after evaluation. Use unchanged ON recurrence for H524M; do not run its ablation mode. Atomic batch outputs must include checkpoint/code/config/panel identity, sequence IDs/token hashes, NLL, counts and CE, with strict resume deduplication.

The new final panel uses the established validation shard SHA:
`8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`.
Known local copy:
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/exp2d2c_transport_20260827/edufineweb_val_000000.npy`.

Select 64 complete canonical B64 batches with isolated NumPy `default_rng(20260924)`, accepting the first nonoverlapping eligible indices in its permutation of complete batches. Exclude the full known previously used/reserved target-span union, including both 2D11 panels, all 2D12 selected spans and any newer project manifests. Preserve accepted order, sequence boundaries and target indexing. For input start s: input[s:s+1024], target[s+1:s+1025]. Freeze dataset/panel identities, source exclusion hashes, token hashes, offsets, selection seed and NumPy version locally before any scientific scoring. Reaudit availability; never silently reuse a prior final panel or pick examples using scores.

Final coverage: 8,192 sequence-condition rows /8,388,608 target predictions, 4,194,304 targets per model. L monitoring adds five ×1,280 sequences =6,553,600 scored targets. Thus the prescribed new monitoring+final work totals 14,942,208 target predictions, excluding preflight and explicitly recorded retries. No HellaSwag or third final condition is required.

## 8. Prespecified local analysis and interpretation

After verified GPU shutdown, align L/H final rows by exact sequence identity. Overall CE is total NLL / total targets; perplexity is exp(overall CE).

One primary contrast:
`d_i = CE_L524M[i] - CE_H524M[i]`.
Positive favors H; negative favors the fresh local-only model. Use 50,000 paired sequence bootstrap resamples, isolated NumPy `default_rng(20260925)`, FP64 means and linear 2.5th/97.5th percentiles. There is one primary contrast, so no multiple-comparison adjustment is required. Add a clearly labeled sensitivity bootstrap over the 64 canonical B64 groups, keeping their sequences together, with 50,000 resamples and seed 20260926. Keep the primary sampling unit fixed and flag conclusions that change with it. Neither interval represents independent training-seed variation.

For primary 95% bounds [lo,hi], report separate flags using delta_CE=0.0001:

- H benefit: lo>0; H benefit beyond reference: lo>+0.0001.
- L benefit: hi<0; L benefit beyond reference: hi<−0.0001.
- Practical equivalence at reference: lo>=−0.0001 and hi<=+0.0001.
- L noninferiority at reference: hi<+0.0001.

Failure to establish benefit is not equivalence. Flags may overlap for tiny precisely estimated effects. The 0.0001 reference is the project's small-effect convention, not a sufficient deployment-value criterion. Report relative PPL difference `100*expm1(mean(d))`, transform interval endpoints monotonically, and include sequence win/loss/tie fractions.

Report model learning and compute separately. Endpoint accounting is:

| Counter | Historical H524M | New L524M |
|---|---:|---:|
| Optimizer updates | 1,000 | 1,000 |
| Logical targets | 524,288,000 | 524,288,000 |
| CE pass-equivalent updates | 2,031 | 1,000 |
| CE pass-target evaluations | 1,064,828,928 | 524,288,000 |
| Mean actual CE passes/logical target | 2.031 | 1 |

H used 969 two-pass and 31 three-pass updates. Do not substitute the full 10B average 2.03125 at this prefix or assign L fictitious repeated-pass compute. Exclude activation-checkpoint recomputation from CE exposure counters while including it in actual measured compute. Report preflight, discarded work, monitoring, checkpoint/export and idle/overhead separately where measurable.

Recover historical H training-only timing from its first 1,000 metric rows: the preparation review found 2,556.652936697 seconds on four GPUs, or 2.840725485 GPU-hours. Reverify this source before reporting it. Compare with L's actual training time/GPU-hours and show CE versus logical targets; label hardware, device count and scheduling differences. This is not a FLOP-matched comparison or an isolated same-configuration speed benchmark. The new experiment's bill includes only newly incurred work, not the sunk H training cost.

If L matches or improves H at this early endpoint, recommend whether longer matched training is worth testing; do not assert that H's10B benefit has disappeared. If H wins, the result supports the full H training recipe over this local-only control at 524M; it does not uniquely prove that attached temporal gradients caused the gain. Retain the distinction from 2D12's frozen H10B OFF result. No automatic next run or architecture adoption is authorized.

## 9. GPU choice, operations and proposed budget

Use one A100 80GB by default, with a currently verified quote. Four A100 80GB GPUs in one pod are also authorized if keeping the established four-rank training path materially reduces implementation or verification difficulty. Record the reason for that choice locally. Four GPUs are an execution choice, not a requirement for a fair comparison. Do not resume a four-GPU pod to leave three devices unused. Complete local readiness first and freeze the selected GPU count, geometry, evaluator partition and checkpoint schema.

The proposed execution ceiling is **6 cumulative billed GPU-hours or USD 10 compute, whichever comes first**, including all startup attempts, preflight, training, evaluations, saves/transfers and shutdown. This provides up to 90 billed pod-minutes in four-GPU mode, before any tighter dollar limit or previous attempts; it is not a six-hour allowance on that pod. It is a ceiling, not a planned runtime. Count prior attempts and any stopped/replaced pod toward the same cumulative budget. Storage charges are separate; no storage expansion or automatic budget increase. Record the actual rate, hardware and available disk capacity before scientific work.

Prepare provider stop capability and arm an independent local progress/deadline watchdog before paid startup; register the assigned pod immediately on allocation. Stage only required code, manifests and missing input files, preferably reusing verified persistent artifacts. Prepare extraction using the known working persistent-filesystem options, including --no-same-owner, rather than rediscovering the 2D12 staging failure. Use resumable transfers and bounded startup/preflight, targeted at useful CUDA work within 10 minutes and at most 10 further minutes of preflight. If extended repair is needed, stop paid compute and continue locally.

Before scientific training, project the full workload from measured update/evaluation/transfer timings, including a conservative export tail and at least 30 wall minutes contingency in one-GPU mode or 10 wall minutes in four-GPU mode. It must fit the remaining cumulative billing ceiling. If it does not fit, stop after preserving preflight evidence and report measured alternatives. Do not reduce targets/evaluations or alter scientific settings to fit an optimistic estimate. Monitor remaining budget, preserve recovery state and stop before the hard deadline; enforce it independently of the training process. No indefinite unattended idle pod. If a mode change is needed after startup, stop the first pod before switching, preserve its evidence/cost and rerun the necessary frozen-config preflight; never keep two pods billing in parallel for this single-arm study.

Start the final L checkpoint export while endpoint evaluations run. After required final checkpoint/raw-output hashes are independently verified locally and on persistent storage, stop the pod promptly and verify provider stopped/EXITED status. Do bootstrap, charts, final audits/reporting, Git work and user discussion locally after shutdown. Do not re-export unchanged H checkpoints unnecessarily or restart a GPU just to write reports. Incomplete budget-limited runs remain incomplete and must not receive a final-success tag.

## 10. Deliverables and success audit

Preserve the frozen protocol, code/input/initial/checkpoint identities, intervention/trainability manifest, optimizer groups, exact stream/hash plan, LR/accounting configuration, CPU/CUDA checks, objective and accumulation equivalence evidence, monitor/final panels, per-update logs, complete resumable L checkpoints and raw per-sequence evaluations.

Final report must contain L/H endpoint CE/PPL, paired 95% interval and flags, group sensitivity, monitoring curves, actual logical/pass exposure, trainable/registered counts, update/terminal-stream audit, new checkpoint hashes, source H identity, compute/timing breakdown, all retries, backup verification, and provider stop evidence. Explicitly state that L was initialized only from untrained weights, H was reused only as comparator, and exactly one new scientific training arm completed 1,000 updates.

Minimum final audit: exact original backbone initialization; every prescribed training-batch hash/cursor matched; all intended parameters trained and inactive tensors stayed unchanged; correct LR/optimizer progression; complete restoreable u1000 state; both endpoint evaluation identities and exact target coverage; all raw/required checkpoint backup hashes; budget compliance; stopped provider; original sealed artifacts unchanged. Record any limits without upgrading them to verified facts.

Commit new code, protocol and appropriately sized results, push the new branch and tag `experiment-2d13-local-only-from-scratch-500m-final` after successful audit. Verify remote identities; keep large checkpoints out of Git. Do not launch another experiment or continuation after reporting.
