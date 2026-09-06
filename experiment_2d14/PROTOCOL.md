# Experiment 2D14 — CE1-free H from scratch, matched at 524M targets

Prepared 2026-09-06 for a new execution task and a separate pod. This document authorizes no work in the ongoing 2D13 task or its pod. The preparation task has not launched training. When the user hands this protocol to the execution task for execution, complete the fixed scope below through verified exports, shutdown, local analysis and Git archival.

## 1. Question and fixed scope

Train **one new recurrent model, R**, from the exact untrained H initialization used in 2D11. Keep H's architecture and training recipe, but remove the direct first-pass CE objective and renormalize the remaining pass weights. Compare it with the saved standard **H524M** checkpoint and the independently training local-only **L524M** model from 2D13. All endpoints are exactly **1,000 optimizer updates / 524,288,000 logical training targets**. “524M” is an abbreviation for this endpoint, not a new token budget.

| Arm | Architecture and objective | Training in this task |
|---|---|---|
| H524M | Standard recurrent H; (.25,.75) or (.20,.40,.40) | None; reuse sealed 2D11 u01000 |
| L524M | Same initial backbone, local-only; one CE pass | None; obtain completed 2D13 u01000 |
| R524M | Standard recurrent H; **(0,1) or (0,.5,.5)** | One fresh 1,000-update run |

Reusing H/L means reusing trained **comparators**. R must not start from either trained checkpoint. It starts from the original saved update-zero H tensors, including the original fresh routers. Do not modify, restart or extend 2D13.

Measure two distinct outcomes: prediction quality with recurrence ON, and dependence on recurrence under matched ALL_OFF evaluations. Removing CE1 does not force recurrence: CE2/CE3 still supervise the full local/recurrent predictor, and the gates can favor local attention. A larger OFF penalty alone is not success if ON quality becomes worse than the accepted tolerance.

Complete all 1,000 updates unless a technical integrity failure or the cumulative billing ceiling prevents completion. Monitoring is descriptive. No outcome-based early stop, hyperparameter adjustment, window change, new gate bias, extra seed, HellaSwag, individual-block ablation, shuffled-memory intervention or 10B extension is included. This is an early screen; it cannot establish the user's eventual quality requirement against GPT-2 at 10B.

## 2. Isolated code and immutable inputs

Workspace: `/Users/rahul/Documents/GPT-2 Enhancement`.

Create a new worktree at `parallel_2d2_master_dev/2d14_ce1_free_scratch` and branch `codex/experiment-2d14-ce1-free-h-524m`, starting from the sealed 2D12 commit **622f34b29ec54816dcf293aef9228939c72a6dd8**, tag `experiment-2d12-h10b-recurrence-ablation-final`. Use a new `experiment_2d14` package. Reuse audited 2D11 H training and 2D12 ON/OFF kernels. Preserve sealed code, reports, checkpoint files and tags.

Read these source trees as references:

- `parallel_2d2_master_dev/2d11_from_scratch`, sealed commit **ca0103cd3aee5326855f4e72d1d6db93ce073550**: fresh H, optimizer, logical stream, cadence and LR.
- `parallel_2d2_master_dev/2d12_h10b_ablation`: structural ALL_OFF and true incremental evaluation.
- `parallel_2d2_master_dev/2d13_local_scratch`: ongoing comparator's frozen manifests, evaluator grouping and, when complete, final L provenance. Its inspected preparation HEAD was **16542a846c4eb70906c12aa3de220537cc9d18dd**, not a final result tag.

Copy any required code into this task's versioned package or pin it to immutable commits. Do not import mutable code from the running 2D13 worktree. Do not execute old training/controller modules just to obtain definitions.

**Scientific initialization — mandatory:**

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/initial.pt`

```text
SHA-256: 387b8b791ced54c2f1c0da65c47d00f35e8c3d1514897bad4c70db36db6c5c94
File size: 498,868,405 bytes
Completed updates: 0; optimizer state: empty
Backbone tensor identity:
7680b0c094254f9b7bd11a9208d753867c9ce68f2efab01cb185934736988851
Full initial H tensor identity:
c1ad032bfc2fccd4fdabae0f0c4064f856b75ae8037e2373a1cc02f8216d6386
```

Use the sealed `tensor_identity` convention. Strictly load the saved tensors and preserve tied embeddings/output weights. Do not regenerate scientific weights from seed 1337: prior platform-dependent initialization differences make a seed alone insufficient. Restore the saved initial Python/NumPy/CPU RNG state and recorded CUDA seed as appropriate, then freeze a complete R update-zero state for the chosen world size. Do not inherit trained optimizer moments or loader progress.

**Standard H comparator — evaluation only:**

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u01000.pt`

```text
SHA-256: 559af751b9cceda5a4b08ef785d64f015cc4ce1978da7130a687f228f80a3244
File size: 1,651,147,173 bytes
Completed updates: 1000; logical targets: 524288000
Model tensor identity:
7d0f5d3da3dcc41a5c9e83c4d3a94fa4aa3caa7be226d1d92521d3c2f0f7548a
Known persistent path:
/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u01000.pt
```

Resolve mounted paths by provenance and hash. Verify actual inputs at use, even if a previous task verified them.

**L comparator — bind when complete:** obtain 2D13's published final checkpoint and audit/identity manifest, not a checkpoint still being written or an arbitrary file named u01000. Verify its source-initial/backbone identity, completed updates, logical targets, full stream audit, terminal cursor, local-only architecture, frozen routers and checkpoint SHA. Record the final source commit/tag when available. Do not invent its future SHA or wait for its final prose report if an immutable checkpoint and the required machine-readable completion evidence are already available. If its evidence is incomplete, the joint comparison remains pending.

## 3. R architecture and precise loss change

Retain H's 12 blocks, width 768, 12 heads, head dimension 64, context 1024, padded vocabulary 50304, shared embeddings, residuals, projections, LayerNorms and MLPs. At recurrent destinations retain the original hierarchical softmax router over local/recurrent retrieval, conditioned on normalized query and both retrieved outputs.

| Destination | Local window | Recurrent writer | Eligible historical lags |
|---|---:|---|---|
| B1 | W2 | B12 post-MLP residual | 1–1023 |
| B3 | W32 | B10 post-MLP residual | 31–1023 |
| B5 | W64 | B8 post-MLP residual | 63–1023 |

Other blocks retain W1024; no B6 recurrence is restored. Retain the original one-token overlap. No available memory means forced local weights (1,0). Otherwise the original fresh router output gives **(0.5,0.5)** before training; do not import mature H's learned ratios.

R has **124,697,386 registered / 124,697,382 trainable parameters**. All backbone and router parameters train. The four unused compatibility scalars remain frozen at their original zeros and outside the optimizer. L's active count is 124,475,904; equal registered schemas do not imply equal active capacity.

For one-based scientific update `u`:

```python
num_passes = 3 if u % 32 == 0 else 2
# Pass 1 executes the full backbone and produces attached h8/h10/h12.
# Subsequent passes consume the preceding pass's attached writer states.
loss = CE2 if num_passes == 2 else 0.5 * CE2 + 0.5 * CE3
```

CE2 must reach Pass1 through recurrent retrieval, destination K/V projections and any router dependence on recurrent retrieval. CE3 must retain the corresponding graph through Pass2 into earlier writers. Do not detach memory, run the Pass1 backbone under `no_grad`/`inference_mode`, freeze it, use detached cached writers or truncate the graph. Shared parameters also receive gradients from the later passes' own computations; loss weights are not percentages of total gradient assigned to physical passes.

For this first controlled test, **continue computing CE1 and log it detached as a diagnostic**, but exclude its tensor from the aggregate training loss. Use the attached CE2/CE3 tensors to form the new loss. Do not use `FreshH.forward()`'s detached logging vector to reconstruct it, and do not accidentally return the inherited standard-H weighted loss. Skipping Pass1 logits/CE is a later optimization, outside this run. The Pass1 backbone remains indispensable.

Wrap the actual R forward entrypoint, containing all passes and the new loss, in DDP if using four GPUs. Calling a wrapped module's `.module.forward_multi_pass()` directly for scientific training would bypass the required DDP forward semantics. Do not inherit 2D13's backbone-only trainability or local-only forward. The 2D12 `load_model()` helper freezes parameters and assumes update 19072; reuse its kernels, not that loader for R training or u1000 inference.

## 4. Matched stream, update geometry and numerical recipe

Every update consumes **512 sequences ×1024 targets =524,288 logical targets**. Freeze the selected execution mode before startup:

| Mode | B per GPU | Accumulation per rank | Loss divisor before backward |
|---|---:|---:|---:|
| Four A100 80GB GPUs, preferred for retaining original H DDP | 32 | 4 | 4, then DDP rank averaging once |
| One A100 80GB, if chosen and verified locally | 32 | 16 | 16; no rank averaging |

Choose one mode and implement its checks, rather than building two runners unnecessarily. The user permits four GPUs; they are not required for matching scientific exposure. Four GPUs preserve H's original training geometry and reduce elapsed time; they are not guaranteed to minimize cost. 2D13's frozen mode is one GPU/B32/16 accumulation. Record the different numerical reduction orders; do not promise bitwise-equivalent trajectories across world sizes.

Use eight original canonical B64 batches per update, the sealed shard order/boundary rules and shifted global token views. Do not redefine sequence boundaries, shuffle examples or use an independent distributed sampler. Validate coverage of all 512 sequences exactly once. Freeze the first 1,000 2D11 stream-plan entries and expected batch hashes/cursors from `experiment_2d11/results/run/metrics-attempt01.jsonl`; require each R batch hash to match **before** applying that update. The reference H-prefix copy accompanying this protocol helps preserve 2D13's matching evidence but does not replace validation of actual R batches.

```text
Terminal cursor: shard=5, position=24576000, wraps=0, logical_batches=8000
Final combined input/target batch SHA:
b29fc811de065ce888ea4556365e3f72ac221bf1675769707a6371635ae8ef45
```

Use a fresh fused CUDA AdamW optimizer over all R trainable parameters: betas (.9,.95), epsilon 1e-8; weight decay .1 for matrices and 0 for vectors/biases. Zero gradients once per update, accumulate all microbatches, clip the global norm once at 1.0, and take one optimizer step. No extra division from logging collectives. In DDP, `no_sync` must enclose forward and backward for the first three microsteps; synchronize the fourth.

Use BF16 autocast, FP32 master weights, the original deterministic settings and high FP32 matmul precision, non-reentrant activation checkpointing, no dropout and no `torch.compile`. Prefer the recorded PyTorch 2.8.0+cu128 environment; record the actual software/hardware versions.

Use the original **10B LR schedule prefix**, not a newly compressed 524M schedule. At update `u`, call the sealed `learning_rate(u-1)`: 715-update linear warmup to 6e-4, cosine based on max_steps=19073, minimum 6e-5. Do not scale LR for GPU count. All 1,000 LR values and optimizer step counters must match the H prefix. Restore scientific RNG/loader state after preflight and around monitoring.

## 5. Required implementation checks before scientific training

Prepare the package, frozen protocol/manifests, transport bundle, CPU checks, resource controller and analysis code locally before starting a paid pod. Mark scientific training ready only after these checks and the bounded CUDA preflight pass:

1. **Initialization and trainability:** strict original full-H tensor identity, tied weights, empty optimizer, intended parameter groups, correct fresh 50/50 routers, four frozen zero scalars. Legitimately zero initial router gradients from zero output weights are not missing parameters.
2. **Objective correctness:** actual R loss and all active gradients match an independently assembled `CE2` / `.5*CE2+.5*CE3` reference. Verify CE1 has no direct path into the aggregate loss. Check two/three passes and checkpointing on/off, including disposable noninitial weights. Use CPU FP32 atol 2e-6 / rtol 2e-5 and report actual errors; same-graph scalar loss construction should agree exactly. Freeze CUDA tolerances before scientific scoring; do not widen them after a failure.
3. **Attached temporal gradients:** isolate CE2 gradients into Pass1 h12/h10/h8 and CE3 gradients into Pass2 and Pass1 writers at eligible positions. Require finite nonzero aggregate gradients at each relevant writer and show the detached-source negative control removes those paths. The existing `project_context/EXPERIMENT_2D11_TEMPORAL_GRADIENT_AUDIT.md` is a reference, not proof of the newly implemented R entrypoint. Do not substitute total shared-parameter gradients for a writer-path test.
4. **Global update semantics:** prove partition/normalization, clipping and AdamW progression. For four GPUs, exercise actual full-size B32×T1024 two- and three-pass DDP updates, inspect cross-rank parameter/gradient agreement, and include real `no_sync` behavior. One-GPU mode needs all 16 microbatches and correct normalization, not the inherited four-microbatch helper.
5. **Recovery:** atomic save/reload reproduces the next disposable update, optimizer state, stream and pass cadence, including u31→u32. Capture and restore each rank's RNG state. Discard all preflight optimizer/data progress before scientific update 1.
6. **Inference:** unchanged ON behavior versus the sealed H implementation and structural ALL_OFF versus the independently verified local incremental reference. Validate H/R/L loading and batch row isolation. Use the same physical B128 evaluation shape; do not invoke broad parallel-versus-incremental BF16 tolerances to excuse an intervention mismatch. Preflight uses disposable/monitor examples, never the final panel.

Exercise the full production shape and benchmark steady-state training, incremental scoring and checkpoint export. Do not expand preflight into a second scientific training arm. Stop paid debugging if the bounded startup/preflight cannot establish readiness, preserve evidence and repair locally.

## 6. Checkpoints and training/pass accounting

Save complete atomic R checkpoints every 250 updates. Permanently preserve u0/u500/u1000; keep at least the latest two validated recovery states during training. Only prune this run's superseded recovery files after validation and according to its frozen retention policy. Reopen each saved checkpoint before declaring it valid. Preserve model, optimizer names/groups/moments/steps, all rank RNG states, frozen scalar hashes, world size, loader cursor, completed update, next LR index, pass accounting, evaluation ledger and cumulative billing ledger.

One writer serializes a shared snapshot in DDP. Use bounded queues, pod-local temporary staging and asynchronous verified exports while GPU work continues. Required checkpoints and raw results need independent persistent-volume and local-Mac copies. The shared original initial checkpoint may be referenced by hash instead of duplicated, but retain R's own complete initial execution/RNG manifest. Preserve replay/discarded work separately after recovery.

At update 1000: **969 two-pass +31 three-pass updates**. Report the following explicitly:

| Counter | Standard H | Local-only L | CE1-free R |
|---|---:|---:|---:|
| Optimizer updates | 1,000 | 1,000 | 1,000 |
| Logical training targets | 524,288,000 | 524,288,000 | 524,288,000 |
| Backbone pass-equivalent updates | 2,031 | 1,000 | 2,031 |
| Backbone pass-target exposure | 1,064,828,928 | 524,288,000 | 1,064,828,928 |
| Actual CE evaluations, including diagnostics | 1,064,828,928 | 524,288,000 | 1,064,828,928 |
| CE target evaluations with nonzero objective weight | 1,064,828,928 | 524,288,000 | **540,540,928** |
| Mean backbone passes/logical target | 2.031 | 1 | 2.031 |

R's nonzero-weight count is `(969 + 2*31)*524288`; its diagnostic-only CE1 count is 524,288,000. These are pass counts, not gradient percentages or equal FLOPs. Activation-checkpoint recomputation is excluded from pass-target exposure but included in measured compute. CE1 removal does not halve R's backbone work. Log CE1/CE2/CE3 and the actual weighted objective separately; do not plot differently weighted objectives as interchangeable losses.

## 7. Shared monitoring and final panel

The authoring task preserves exact copies of 2D13's already frozen manifests in:

`/Users/rahul/Documents/GPT-2 Enhancement/project_context/experiment_2d14_reference_inputs/`

Read `REFERENCE_MANIFEST.json` and verify the copies. Freeze these identities in 2D14 before viewing new final scores. **Do not select another final panel.** This common panel supports the requested three-model paired comparison. It is a panel reserved by 2D13 and shared prospectively with 2D14, not a fresh independent confirmation after seeing 2D13 outcomes. Leave 2D13's original two-condition analysis unchanged.

```text
Validation dataset SHA:
8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f

Final PANEL.json file SHA:
30d8bd365c0f82db7993efe6628370a9817c275a46c36a55c7c7f8629ea7de7c
Final panel identity:
1d301019ceccca0c59801bcd1aa38a28b4e0bd90b9e54ed0c28dc27a60454602
4096 sequences; 64 canonical B64 groups; 4,194,304 targets per condition

MONITOR.json file SHA:
018d763b29c3787a26bfcb33ba4b0509b3015f603d1fe73f58347c9c440b5204
Monitor panel identity:
db6a61e95fa56a0fc3fb15c31a322d5248a7ad36f330ef25eeca96a799891381
1280 sequences; first 20 canonical B64 groups; 1,310,720 targets
```

Monitor R at **u0/u250/u500/u750/u1000** on this same monitor panel, without changing the training loader or RNG state. Reuse identity-verified H/L monitoring outputs for descriptive same-panel curves when available. Original H monitoring lives in `experiment_2d11/results/run/evaluations/attempt01/ce_{u}.json` and corresponding analysis CSVs. No new H/L monitoring campaign is included; document evaluation-shape/software differences in historical curves rather than treating them as paired final evidence.

**This task owns exactly five completed endpoint evaluations**, independently scoring all conditions using its frozen evaluator:

| Label | Checkpoint | Incremental inference |
|---|---|---|
| H_ON | Original H u01000 | All recurrent reads ON |
| L_LOCAL | Completed 2D13 L u01000 | Structural local-only model |
| R_ON | New R u01000 | All recurrent reads ON |
| H_ALL_OFF | Same H u01000 | B1/B3/B5 recurrent reads OFF |
| R_ALL_OFF | Same R u01000 | B1/B3/B5 recurrent reads OFF |

Re-evaluate H/L in this task so the five final conditions share numerical execution. The original 2D13 H/L evaluations remain its separate report. Do not mix previously reported absolute CE values from other panels into the joint table.

Use **true incremental inference for all five**, BF16 forward, FP32 token CE, FP64 NLL sums, exactly 1024 targets per sequence, and fresh independent caches per sequence/condition. ALL_OFF means bypass recurrent reads and routers, use local weight 1 and recurrent weight 0, and apply the shared output projection once. Do not merely zero AR while retaining a subunit local weight. Preserve local windows and the existing cache bookkeeping; L is a separately trained model, not the H_ALL_OFF checkpoint. No memory must leak across conditions, batches or sequences.

Keep **B128 per worker**, exactly the same grouping of manifest rows as 2D13. For four GPUs, distribute whole B128 batches by deterministic global batch index modulo four, with every batch scored once. Keep batch composition/ordering unchanged when a condition is scored later or on a different GPU count. If B128 cannot be validated, stop before final scoring and report the incompatibility rather than silently changing the shared design. Merge by global IDs, not file arrival order.

Each atomic batch artifact records checkpoint SHA/model tensor identity, code/configuration/panel identities, mode, batch grouping, condition, sequence IDs/token hashes/offsets, canonical group, NLL, count and CE. Strictly deduplicate resumptions by full binding, not just file existence. Verify checkpoint weights before/after inference. Retain counts and timing, but no large gate/attention dumps are required; use ON/OFF CE to measure dependence.

Final coverage: **20,480 sequence-condition rows /20,971,520 target predictions**. R monitoring adds 6,553,600 target predictions; prescribed new monitoring plus final scoring totals **27,525,120**, excluding disposable checks and recorded retries.

## 8. Joint paired analysis after GPU shutdown

Align all five endpoint rows by exact sequence identity. Compute total NLL/total targets and PPL=exp(CE). Prespecify these **six primary contrasts**, using per-sequence values:

| Symbol | Definition | Positive value means |
|---|---|---|
| A | H_ON − R_ON | R predicts better than standard H |
| B | L_LOCAL − R_ON | R predicts better than trained local-only L |
| C | L_LOCAL − H_ON | Standard H predicts better than L |
| G_H | H_ALL_OFF − H_ON | Standard H benefits from recurrence at inference |
| G_R | R_ALL_OFF − R_ON | R benefits from recurrence at inference |
| I | G_R − G_H | R has the larger ALL_OFF penalty |

Use **50,000 paired sequence bootstrap resamples**, isolated NumPy `default_rng(20260928)`, FP64 means and linear percentiles. Use each sampled sequence-index vector for all five conditions and all six contrasts; especially preserve within-sequence pairing for I. Report raw 95% CIs and Bonferroni-adjusted **99.1666666667% CIs** for the six-contrast family (percentile bounds 0.4166666667 and 99.5833333333). Use adjusted intervals for primary classifications. This is a new joint family, not a retroactive change to 2D13's original single contrast.

Add a labeled sensitivity analysis with 50,000 paired resamples of the **64 complete canonical groups**, seed 20260929, using the same six-contrast adjustment. Report changed conclusions. These intervals quantify evaluation-sequence/group uncertainty for these fixed training trajectories, not training-seed replication.

Use **delta_CE=0.0001** as the inherited small-effect reference, not a sufficient deployment-value threshold. For each contrast's adjusted [lo,hi], report positive (`lo>0`), beyond-reference (`lo>delta`), negative (`hi<0`), material-negative (`hi<-delta`) and equivalence (`lo>=-delta and hi<=delta`) separately. Failure to establish benefit is not equivalence. For A, R noninferiority to H requires `lo>-delta`.

The prespecified joint goal flag requires all three:

1. R quality is noninferior to H: `lo(A)>-delta`.
2. R has a useful ON/OFF effect: `lo(G_R)>delta`.
3. R has a larger dependence effect than H: `lo(I)>delta`.

Also report whether R improves ON quality beyond the reference against H and/or L. If dependence grows while ON quality is materially worse, state that the user's joint objective was not met. If R is noninferior but not superior, do not call it a better predictor. Greater OFF sensitivity can include coadaptation or mixture rescaling; it is not uniquely proof of better recurrent content. No architecture adoption or continuation is automatic.

Report relative PPL changes with explicit numerator/denominator, e.g. H-to-R reduction `100*(1-exp(-mean(A)))`. Include sequence win/loss/tie fractions and same-monitor-panel CE curves against logical targets and measured training GPU-hours. Keep the H/L/R objectives and compute accounting distinct. Historical H first-1000 training-only timing was **2,556.652936697 seconds on four GPUs /2.840725485 GPU-hours**; reverify before reporting. Different hardware/world sizes and concurrent shared-volume I/O limit speed conclusions. Never compare R524M's ablation gap directly with H10B's different-age/different-panel 2D12 gap as evidence of training-recipe effect.

## 9. Separate pod and shared-volume discipline

The user reports increasing network volume **yhzyb27fb5 to 200 GB** on 2026-09-06. Treat 200 GB as user-reported until provider verification. Check supported concurrent mounting/location before allocation; attaching this pod must not detach or move 2D13's volume.

Use a unique persistent namespace `/workspace/exp2d14/<run_id>/` and local archive `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d14/<run_id>/`. Keep environments, package caches, temporary files and transfer staging on this pod's local disk where possible. Existing datasets, initial weights and historical H are read-only shared inputs; do not duplicate the dataset or existing H checkpoints. Store this task's dependency bundle, PID files, status, controller binding, heartbeat, locks and checkpoints only under its own namespace. No shared mutable `latest` pointer, global package upgrade, broad ownership change, pruning or deletion outside that namespace.

The 2D13 provider code inspected during preparation **hardcodes pod y96gntb89tzuvj** and its own controller paths. **Do not import or run that provider/controller unchanged. Do not stop, restart, reconfigure or signal that pod.** A new controller must take an explicitly assigned 2D14 pod ID, verify its run binding, GPU count and volume ID, and restrict every mutation/shutdown to that ID. Do not enumerate volume-attached pods and stop them collectively. The other task's instruction to keep its pod alive during preparation does not apply here.

Before paid startup, prepare a capacity/reservation estimate from available metadata; confirm actual capacity/free space on mounting before writes or scientific work. The recorded 2D13 preparation audit was:

```text
Observed existing use:           179,770,149,871 bytes
Conservative 2D13 remaining reserve: 8,500,000,000 bytes
Proposed 2D14 peak new reserve:      8,500,000,000 bytes
Shared safety buffer:               2,000,000,000 bytes
Conservative projected total:     198,770,149,871 bytes
User-reported new quota:          200,000,000,000 bytes
```

Source: `parallel_2d2_master_dev/2d13_local_scratch/experiment_2d13/results/LOCAL_DEVELOPMENT_LOG.json`. This is an older snapshot, not a live free-space measurement. The new quota appears sufficient for this conservative plan, but only narrowly. Recalculate current usage, this run's actual checkpoint/temp peak, and the other run's still-required reservation; do not assume all observed free space belongs to R. Avoid double counting work already reflected in the current snapshot, but do not release 2D13's reservation without completion evidence. Persist this run's reservation and monitor remaining space. Do not edit 2D13's own reservation files or require its process to honor a newly invented lock.

If the estimate does not fit, finish local preparation and report the measured shortfall before paid scientific work. Do not reclaim historical/2D13 files or silently expand beyond 200 GB. Defer startup or request a specific extra allocation based on the measured requirement. Any pod started solely to validate mounting/capacity must be stopped promptly if the prerequisite fails.

Read L's checkpoint only after atomic completion. Validate its completion manifest/hash and confirm the source did not change during acquisition. Ignore `.tmp`/in-progress files. Readiness concerns with 2D13 do not authorize repairing, messaging, changing or stopping that task. Report required user coordination instead.

## 10. Billing, idle time and a late L comparator

Proposed ceiling for **this new task only**: **8 cumulative billed GPU-hours or USD 15 compute, whichever comes first**, including all attempts, startup, preflight, scientific training, monitoring, final evaluation, transfers and verified shutdown. Four GPUs permit at most two billed pod-hours before a tighter dollar limit; one GPU permits at most eight. This is a ceiling, not a runtime promise. It does not consume or redefine 2D13's separate allowance. Record current quoted rates and storage costs separately; no automatic budget increase.

Prepare an independent local deadline/progress watchdog and tested provider stop path before starting the pod. Bind it immediately to the assigned new pod. Target useful CUDA work within 10 minutes, with at most 10 further minutes of preflight; if repairs exceed that window, preserve evidence, stop paid compute and continue locally. Derive workload estimates from measured update/evaluation/transfer rates before scientific training. Include all five final conditions, five monitors, exports and at least 10 minutes contingency on four GPUs or 30 minutes on one GPU. If the remaining ceiling cannot cover the projected work, stop and report measured alternatives; do not silently reduce targets or final coverage.

Run all selected GPUs during training. Distribute each evaluation across the pod's workers using the fixed batch partition, rather than leaving three devices unused. Export checkpoints/raw results in the background. Charge every retained idle device to the ledger; do not imply perfect utilization from assigned jobs alone.

**Do not keep this pod billing while waiting for L524M.** After R training, complete R_ON/R_ALL_OFF/H_ON/H_ALL_OFF and any available L_LOCAL work. If L is still unavailable when independent GPU work and required exports are finished, verify local and persistent copies, stop this task's pod, verify provider stopped/EXITED status, and report “R complete; joint comparison awaiting L.” Do local analysis/code/report preparation while stopped. Do not poll from a paid GPU or auto-start recurring monitoring without the user's request.

Once L's completed checkpoint becomes available, the remaining L_LOCAL evaluation is still within this protocol's scope. Reuse the same frozen code, B128 grouping, precision and compatible GPU type. A short one-GPU evaluation session can avoid restarting four devices; validate the fixed-batch numerical mode and include its startup/evaluation/export time in the same cumulative ceiling. Do not retrain R, rescore completed identity-matched conditions or select a new panel merely because L arrived later. If the remaining budget is insufficient, report the exact unfinished work and measured additional requirement before incurring extra cost.

After GPU outputs/checkpoints have independent verified durable copies, stop the assigned pod promptly and verify actual provider state. Bootstrap, figures, audits, reporting, Git and discussion run locally. A transient utilization dip during a live save/worker transition is not a reason to kill an active valid update. Conversely, completed work is not a reason to keep a pod waiting for CPU analysis or another task.

## 11. Deliverables and completion gate

Preserve the preregistered protocol, code/input/panel/initial identities, trainability and loss manifests, complete stream/LR plan, CPU/CUDA/autograd/DDP/recovery checks, per-update metrics and pass counters, resource/reservation ledgers, resumable R checkpoints, five-condition raw endpoint rows, R monitors, H/L source provenance, independent backup hashes and provider stop evidence. Store large checkpoints outside Git.

The final report contains all three ON/local endpoint CE/PPL values, both recurrent ON/OFF effects, their paired difference I, raw/adjusted six-contrast intervals, group sensitivity, the joint goal flag, learning curves, trainable counts, actual pass exposure and time/cost breakdown including failures, idle periods and any late evaluation session. State that exactly **one new scientific R arm** trained here from untrained weights, while H and L were reused only as comparators. Describe initial 50/50 gates and the limits of one early matched trajectory.

Final audit must establish: exact full-H initial identity; intended loss with CE1 excluded; attached writer gradients; matched 1,000-update stream/LR/cadence; correct optimizer/trainability; validated final R state; independently verified L provenance; unchanged source H/initial/2D13 artifacts; all five final condition identities and exact coverage; complete backups; shared-volume preservation; cumulative budget compliance; and all this task's pods stopped. Missing L means joint completion is pending, not a successful five-condition result.

Commit and push the new implementation/protocol/results branch. Create immutable tag **experiment-2d14-ce1-free-h-524m-final** only after the complete joint audit passes, and verify remote identities. Intermediate R completion can be committed and reported without that final-success tag. Do not launch further training after reporting.
