# Experiment 2D15 — fresh L and CE1-free R, sequential four-GPU training through 2.62144B

Prepared 2026-09-07 for the existing **Run Experiment 2D13** task. This is a combined handoff for both arms. Preparation has not started a pod or training. The user's latest instructions supersede earlier two-pod proposals: **use one pod with four GPUs; train L first, then R; match historical H's actual settings, including fused=False and four-rank training geometry.**

**The pod must remain running until both arms have finished all prescribed GPU work and their required outputs are safely exported. Finishing L is not a shutdown condition.** No intermediate user confirmation is needed to start R after L. Do not launch another task or operate the old 2D13/2D14 pods.

## 1. Scientific scope and milestones

Run two new, separately initialized training trajectories. Each starts from the exact original untrained 2D11 H artifact, never a trained 2D13/2D14 checkpoint. L uses its original backbone tensors; R uses the full original H tensors, including fresh routers. Each reaches update 5,000 under the original 10B LR schedule prefix.

| Milestone | Completed updates per arm | Logical targets per arm | Historical comparator |
|---|---:|---:|---|
| Approximately 500M | 1,000 | 524,288,000 | H u01000 |
| Approximately 1B | 2,000 | 1,048,576,000 | H u02000 |
| 2.62144B | 5,000 | 2,621,440,000 | H u05000 |

These are checkpoints along each continuous fresh run, not three independently restarted runs. Both arms start at update zero once; after each milestone they continue with their own optimizer/RNG/loader state. R's update zero remains fresh even though L has already completed on the same pod.

Reuse H checkpoints only as evaluation comparators. Do not train a new H arm. Preserve the original fused=True L/R experiments and their negative/positive findings as historical results; do not relabel, overwrite or continue their checkpoints. Name the new arms **L_nf4** and **R_nf4** in manifests and reports to distinguish them.

Complete all 5,000 updates per arm, with interim reports at the three matched checkpoints. No score-dependent early stop or recipe adjustment is introduced. The user will decide what to do next after seeing the three milestones. No extension beyond 5,000 updates, additional seed, new window, gate initialization change or extra training arm is included.

Keep the recurrence-dependence measurements from 2D14: at each milestone evaluate H_ON, L_LOCAL, R_ON, H_ALL_OFF and R_ALL_OFF. This is **15 completed condition evaluations**, not 15 training runs. No HellaSwag is included in this scoped CE/dependence comparison; it remains necessary before a later broad 10B benchmark claim.

## 2. Source isolation and checkpoint identities

Workspace: `/Users/rahul/Documents/GPT-2 Enhancement`.
Create a new worktree `parallel_2d2_master_dev/2d15_lr_nonfused_serial` and branch `codex/experiment-2d15-l-r-nonfused-serial-4gpu`, based on sealed 2D14 commit **f260dd90f717034653ef6ccf3ddccbfdb4805a20**. Create a new `experiment_2d15` package/controller. Reuse audited kernels and analysis concepts, not old launch/shutdown scripts unchanged.

Reference commits:

- 2D11 historical H: `ca0103cd3aee5326855f4e72d1d6db93ce073550`.
- 2D12 ON/OFF kernels: `622f34b29ec54816dcf293aef9228939c72a6dd8`.
- 2D13 L: `c0c8810bb4ade078da50f8fbe49cb18a3e9519e9`.
- 2D14 R: `f260dd90f717034653ef6ccf3ddccbfdb4805a20`.

Preserve those worktrees, their scientific artifacts and immutable tags. Pin/copy required utilities into the new package rather than importing mutable external worktrees. Do not invoke an old provider/controller merely to obtain a helper.

**Mandatory original initialization:**

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/initial.pt`

```text
SHA-256: 387b8b791ced54c2f1c0da65c47d00f35e8c3d1514897bad4c70db36db6c5c94
Size: 498,868,405 bytes
Completed updates: 0; optimizer state: empty
Initial backbone tensor identity:
7680b0c094254f9b7bd11a9208d753867c9ce68f2efab01cb185934736988851
Initial full H tensor identity:
c1ad032bfc2fccd4fdabae0f0c4064f856b75ae8037e2373a1cc02f8216d6386
```

Strictly load saved tensors, preserve tied embeddings/output weights and prove identity using the sealed tensor-identity function. Do not regenerate scientific weights from seed 1337; platform-dependent initialization was previously observed. Restore the saved original Python/NumPy/CPU RNG and CUDA seed for each arm, and freeze separate complete four-rank initial execution states. Disposable preflight must not consume either scientific initial state.

Historical H checkpoints live locally under `runpod-checkpoint-archive/experiment_2d11/run/checkpoints/` and persistently under `/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/`:

| File | SHA-256 |
|---|---|
| u01000.pt | `559af751b9cceda5a4b08ef785d64f015cc4ce1978da7130a687f228f80a3244` |
| u02000.pt | `a0a3f37aec8cbd48b74221ac9d5c1c55732f1fdb40219de5ea85fa2396b6659d` |
| u05000.pt | `fc76b6d96f8805d0462f9c801936acc1b18d18332216d3f09138329f5b11b7a2` |

All three actual local checkpoint byte hashes were independently rechecked during protocol preparation. Reverify inputs at use, including completed updates, optimizer groups/steps, full model identity and data cursor. Matching declared configuration files alone is insufficient.

## 3. Exact matching to H's optimizer and execution

The prior mismatch must be removed, not merely mentioned again. Historical H's original initialization/u1000 serialize `fused=False` and `foreach=None`. Before scientific work, inspect the actual original initialization and all three H checkpoint optimizer groups, plus the sealed CUDA restore/step code. Freeze a machine-readable table of all optimizer options and their actual CUDA dispatch. If a historical milestone conflicts with the expected recipe, resolve it before training rather than assuming consistency.

Use the same PyTorch **2.8.0+cu128** training environment, CUDA recipe and four A100 80GB GPUs as H's recorded configuration. Verify actual hardware/software and keep them fixed across L and R. If the requested hardware/runtime cannot be matched, report the difference before allocating/training instead of silently substituting a one-GPU run.

```text
world size = 4; all four GPUs train the same active arm
microbatch per rank = 32 sequences ×1024 targets
accumulation = 4 microbatches per rank
effective global batch = 512 sequences =524,288 targets/update
backward loss = microbatch mean objective /4
DDP averages ranks exactly once
one global clip at 1.0, then one optimizer step
AdamW: betas=(0.9,0.95), eps=1e-8
weight_decay=0.1 for matrices, 0 for vectors/biases
fused=False; foreach=None, subject to verification against saved H groups
other flags/group order/dtypes: match actual H, not library defaults guessed later
```

Do not replace `foreach=None` with True/False for speed or assume `fused=False` alone specifies every backend detail. Verify the actual dispatch against historical H's runtime and flags. Keep `fused=False` after initialization, every restore and every scientific update; record it in every checkpoint. Inherited 2D13/2D14 assertions demanding fused=True must be replaced for this new experiment. For L, preserve H's optimizer options but include only its active backbone parameters; never load H's trained optimizer moments or mismatched parameter groups into L.

Use full DDP forward semantics for the actual L/R objective. `no_sync` encloses forward and backward for microsteps 0–2; microstep 3 synchronizes. Do not bypass DDP through `.module.forward_multi_pass`, double-divide gradients, use independent samplers, or retain a one-GPU loss/16 path. Use H's exact physical partition: at microstep m and rank r, select rows `[(m*4+r)*32 : (m*4+r+1)*32]` from the reconstructed global batch.

Preserve BF16 autocast, FP32 master parameters and token CE, non-reentrant activation checkpointing, no dropout, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, high FP32 matmul precision and no compilation. Preserve H's DDP options: broadcast_buffers=False, find_unused_parameters=False, static_graph=False. Record the resolved attention/backend settings. Matching the recipe does not promise identical model trajectories after the intended architectural/loss intervention or eliminate hardware/library numerical limits.

For update u, use the sealed `learning_rate(u-1)`: warmup 715 updates, peak 6e-4, minimum 6e-5, cosine max_steps=19073. **Do not compress the LR schedule to 5,000 updates** or scale LR for GPU count. The three milestones are prefixes of the same original training schedule.

## 4. Architecture and loss definitions

Both arms preserve the 12-block, width-768, 12-head GPT-2 backbone, head dimension 64, context 1024 and padded vocabulary 50304. Local windows remain B1 W2, B3 W32, B5 W64, and W1024 elsewhere.

**L_nf4:** structural ALL_OFF throughout training/inference. B1/B3/B5 use local attention with effective weights (1,0), no recurrent read/router, and one original output projection. Freeze the unused routers and four compatibility scalars; train every backbone parameter. Registered parameters:124,697,386; active:124,475,904. Use one actual CE pass after verifying equivalence to repeated local-only weighted passes; this deliberate collapse was already part of L's definition and avoids meaningless duplicate computation. It does not make L compute-matched to H.

**R_nf4:** full H recurrence/routers. B1 reads prior B12 states at lags1–1023; B3 reads prior B10 at31–1023; B5 reads prior B8 at63–1023. Keep original fresh 50/50 router initialization on eligible positions and forced (1,0) when memory is unavailable. All backbone/router parameters train; only the four original unused zero scalars are frozen. Active count:124,697,382.

```python
# R, one-based scientific update u
num_passes = 3 if u % 32 == 0 else 2
objective = CE2 if num_passes == 2 else 0.5 * CE2 + 0.5 * CE3
```

Keep all Pass1/Pass2 writer states attached to autograd. CE2 must reach Pass1 h12/h10/h8, and CE3 must retain its path through Pass2 into earlier writers. No detached memory, no_grad Pass1, gradient truncation or cached detached source. Continue computing/logging detached CE1 as a diagnostic but exclude its tensor from the training objective. Do not reconstruct the objective from detached logging values or accidentally use the inherited H weighted loss. H's original (.25,.75)/(.20,.40,.40) objective belongs only to the reused comparator.

## 5. Full data stream and counters

Freeze the first **5,000** entries of the original 2D11 stream plan and `metrics-attempt01.jsonl`, which covers these checkpoints before H's later recovery. Match actual per-update input/target hashes, before/after cursors, target counts and LR before taking each step. Use eight original canonical B64 logical batches per update, unchanged shifted-token views, sequence boundaries, shard order and boundary rules. No shuffle, new data or logical stream restart at milestones.

The 5,000-update prefix reaches training **shard 26**. Verify/stage the required 27 shards by the original manifest; do not reuse the old 524M preparation's six-shard-only check. Prefer existing read-only verified dataset files on the shared volume to duplicate uploads.

| Update | Cursor after update (shard, position, wraps, logical_batches) | Combined token hash |
|---|---|---|
| 1000 | (5,24576000,0,8000) | `b29fc811de065ce888ea4556365e3f72ac221bf1675769707a6371635ae8ef45` |
| 2000 | (10,49152000,0,16000) | `c020e99a9aab8ed087db51e4dbb84b1ae75bb5306be6adb1e622739abe75bd1a` |
| 5000 | (26,22937600,0,40000) | `20dbd52ccb6f7196e8014ddd63dd103886e0b8405c49192d96d875c8b1b2786f` |

These terminal examples do not replace all 5,000 individual checks for each arm.

| Update | H/R two-pass /three-pass updates | H/R backbone passes | H/R actual CE pass-target evaluations | R nonzero-weight CE targets | L actual CE targets |
|---|---|---:|---:|---:|---:|
| 1000 | 969 /31 | 2031 | 1,064,828,928 | 540,540,928 | 524,288,000 |
| 2000 | 1938 /62 | 4062 | 2,129,657,856 | 1,081,081,856 | 1,048,576,000 |
| 5000 | 4844 /156 | 10156 | 5,324,668,928 | 2,703,228,928 | 2,621,440,000 |

R's diagnostic-only CE1 count equals its logical targets. H's actual CE passes all have nonzero objective weights. Exclude activation-checkpoint recomputation from these exposure counters but include it in measured compute. Separately report disposable/replayed work. Each new arm has 2.62144B logical targets; their combined new scientific exposure is5.24288B, not one 2.62144B budget split between them.

## 6. Local readiness and bounded verification

Prepare **both arms, their four-rank runners, evaluator, analysis and serial controller before scientific L starts**. Do not finish L and then begin implementing R while holding the pod. Prepare locally before paid startup where possible. If the user has already reserved the assigned pod, retain it as explicitly requested and include that time in the ledger.

CPU verification must cover exact initialization, trainability, optimizer group inheritance with fused=False, L single-versus-repeated objective/gradient equivalence, R CE1 exclusion and attached writer paths, stream partition/counters, checkpoint compatibility and shutdown-controller logic. Reuse meaningful prior tests with the new optimizer/geometry; do not treat an old fused=True test result as proof of this run.

Before L training, use disposable states for full-size B32×1024 four-rank CUDA preflight of **both** L and R, including two/three-pass R updates, finite gradients, rank agreement, correct global normalization/clipping/AdamW dispatch, memory fit and save/reload across u31→u32. For L verify single-pass equivalence against explicit ALL_OFF repeated objectives. For R isolate gradients into earlier writer tensors and use a detached-source negative control. Verify ON/OFF incremental references, causality, row isolation and cache reset at the actual scoring shape.

Use CPU FP32 atol2e-6/rtol2e-5. Preserve the prior L BF16 equivalence criteria (loss atol/rtol1e-5, active-gradient relative L2<=.01, cosine>=.9999) with detailed finite/zero patterns and clipping/optimizer comparisons. Actual R/reference construction should use attached loss tensors with identical operations. Freeze CUDA/DDP/resume tolerances before scientific work; investigate failures instead of increasing tolerances after observing them. H's four-rank preflight provides an independent normalization reference.

Add mocked controller tests proving that finishing L, finishing R training with evaluation pending, a temporary idle interval, an estimated-budget alert, and an exception while R remains incomplete **do not issue a pod stop**. Only the complete combined readiness predicate below permits automatic shutdown. Exercise resumable transition L→R so restarting the controller cannot retrain L, skip R or initialize R from L's optimizer/RNG/weights.

Restore each arm's exact step-zero state after disposable preflight. No preflight weight, optimizer step or token cursor enters scientific training. Record all checks and actual timing projections before starting L.

## 7. One pod, serial stages and checkpoints

Use a new, explicitly bound pod with **four GPUs**, and a unique root `/workspace/exp2d15/<run_id>/` containing separate `L_nf4/`, `R_nf4/`, `shared_inputs/` and `controller/` subdirectories. Local archive: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d15/<run_id>/`.

One parent controller owns the whole experiment and the assigned pod. Its durable stages are:

```text
LOCAL_READY -> BOTH_ARMS_PREFLIGHT_PASSED
-> L_TRAIN_AND_SCORE (through u5000)
-> L_DURABLY_COMPLETE
-> R_RESET_TO_ORIGINAL_INITIAL_STATE
-> R_TRAIN_AND_SCORE (through u5000)
-> JOINT_GPU_OUTPUTS_VERIFIED -> STOP_ASSIGNED_POD
-> LOCAL_FINAL_ANALYSIS_AND_GIT
```

Use all four GPUs for L, release that process group, then use all four for R. Never divide the pod into two two-GPU jobs. Start R automatically as soon as L's required GPU work and handoff checks finish; don't wait for local bootstrap, plots, a report or user confirmation. Finish both arms on this same pod unless an external failure or explicit user instruction requires recovery elsewhere.

Save complete atomic checkpoints every500 updates. Preserve u0/u1000/u2000/u5000 for each arm and keep the latest two validated rolling recovery states. Include all model tensors, optimizer options/names/groups/moments/steps, each rank's RNG, loader cursor, next LR/pass index, arm identity, evaluation ledger and cumulative resource ledger. Validate reopened files and independently verify required checkpoints/raw outputs in local and persistent copies. Background export queues must be bounded; serialize snapshots once rather than once per rank.

Freeze a retention policy that only removes this run's superseded rolling files after validated replacements and required export acknowledgments. Never prune old 2D11/2D13/2D14 artifacts. Resume from the relevant arm's validated checkpoint; resolve pending milestone scoring by full identity, record discarded/replayed work and retain global progress across process/controller restarts.

**Storage prerequisite:** the last recorded 2D14 volume snapshot used194,041,417,965 bytes of200,000,000,000. Both completed histories must remain preserved. Budget approximately18 GB additional peak space for this run's two milestone archives, rolling snapshots, temporary publication and small artifacts, plus a3 GB shared reserve. That older snapshot projects approximately215 GB required, so the current200 GB allocation is unlikely to fit. A proposed230 GB allocation provides headroom; this protocol does not silently authorize a resize. Verify current usage and the actual retention/temp peak locally or through read-only storage access, and obtain a specific storage increase from the user if needed **before paid startup where possible**. If a pod is already user-reserved, report the blocker and preserve that reservation per the shutdown instruction; do not delete historical files or start scientific work without sufficient capacity.

Check local Mac headroom as well as network quota. Put environment/package caches, disposable files and staging on pod-local disk; reuse original dataset/H checkpoints read-only. Quota is capacity metadata, not an exact identity condition in the stop path: the earlier `size==190` shutdown failure must not recur. Bind mutations strictly to the newly assigned pod ID and expected volume ID; do not inherit old hardcoded pod IDs or broad `finally: stop()` cleanup.

## 8. Monitoring and matched milestone scoring

Monitor each new arm at u0/250/500/750/1000/2000/3000/4000/5000 on the original 1,280-sequence H monitor panel. Preserve/restore RNG, training mode and loader state around scoring. Reuse verified H monitoring artifacts at those exact updates for descriptive learning curves; report historical batch-shape differences. Do not overlay L/R/H differently weighted training objectives as interchangeable losses.

For decisive comparisons, select and freeze **one new 4,096-sequence validation panel**, then use its exact same sequences at all three milestones and for all five conditions. This permits same-panel trajectory plots. Do not reuse the already examined 2D13/2D14 final panel as fresh confirmation.

Use validation shard SHA `8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`, 64 complete canonical B64 groups, isolated NumPy selection seed20260930, accepting the first eligible groups in a permutation. Exclude all previously scored/reserved target spans recovered from the established project manifests, including 2D11, 2D12 and the shared2D13/2D14 panel. Preserve sequence boundaries and shifted targets. Freeze selection/exclusion hashes, offsets, IDs/token hashes, dataset/panel identity and NumPy version before scientific scoring. If no fresh4096 panel remains, resolve that explicitly before training; do not silently reuse or change the validation corpus.

At each milestone u=1000/2000/5000:

- During **L stage**, score L_LOCAL_u, H_ON_u and H_ALL_OFF_u using their matching-age checkpoints.
- During **R stage**, score R_ON_u and R_ALL_OFF_u, completing that milestone's five-condition comparison.

All15 conditions use true incremental BF16 forward, FP32 token CE, FP64 NLL sums, B128 per worker, reset caches and1,024 targets per sequence. Distribute whole fixed B128 groups across all four workers by batch index modulo4, preserving composition/order. Never compare parallel L evaluation with incremental H/R. ALL_OFF bypasses recurrent reads/routers and forces (1,0), with the original local windows and one output projection; it does not merely zero AR while retaining the learned local multiplier.

Every atomic batch output binds the exact arm/milestone/checkpoint SHA/model identity, code/configuration/panel identity, precision/grouping, global IDs, token hashes, offsets, group, NLL/count/CE. Verify weights unchanged and ON/OFF checkpoint identity. Resume/deduplicate by full binding. Each milestone requires20,480 rows /20,971,520 predictions; the three milestones total **61,440 rows /62,914,560 predictions**. Nine monitors per new arm add23,592,960 predictions, so prescribed new monitoring+milestone scoring totals **86,507,520**, excluding disposable checks/retries.

Produce H/L interim summaries during L stage if useful, clearly labeled incomplete three-model comparisons. During R stage, publish the complete500M/1B/2.62144B comparisons as their rows become available; continue training without waiting for local analysis. Do not tune settings based on these reports.

## 9. Paired analysis and interpretation

At each milestone define six contrasts exactly as in2D14: A=H_ON−R_ON; B=L_LOCAL−R_ON; C=L_LOCAL−H_ON; G_H=H_ALL_OFF−H_ON; G_R=R_ALL_OFF−R_ON; I=G_R−G_H. Positive A/B favors R quality; positive G values favor recurrence ON; positive I means greater R OFF sensitivity.

Use50,000 paired sequence bootstrap resamples with isolated seed20261001, FP64 means and linear percentiles. Apply each sampled index vector to all five conditions at all three ages, preserving within-sequence and across-age pairing. The prespecified family is **18 contrasts** (six×three milestones). Report raw95% intervals and Bonferroni-adjusted **99.7222222222% intervals**, percentile bounds0.1388888889/99.8611111111. Add a50,000-resample paired64-group sensitivity with seed20261002 and the same18-contrast adjustment. Use adjusted intervals for classifications, including interim classifications; the family does not shrink because later results are pending.

Keep delta_CE=.0001 and separate positive, beyond-reference, negative, materially negative and practical-equivalence flags. At each age, the CE1-free joint goal requires lo(A)>−delta, lo(G_R)>delta and lo(I)>delta. Always report R versus L as well: larger recurrence dependence alone is not better quality. The final u5000 result is the main endpoint; earlier results describe the learning trajectory under the same declared family. Plot CE/PPL and contrast means versus targets and measured GPU-hours. No separate formal effect-growth test or curve-area selection is added after seeing results.

These are three checkpoints from each of two trajectories, not three independent training replications. Even with matching optimizer/geometry, the intentional L/R architecture/pass differences remain. Do not infer isolated recurrent-content usefulness from gate coefficients or OFF sensitivity alone. Do not compare absolute CE across different old/new panels, or claim broad GPT-2 superiority without a matched10B/HellaSwag assessment.

## 10. Pod retention, cost and final shutdown rule

**Explicit user override:** retain the one four-GPU pod through both arms and all required GPU work. Older per-arm budgets, deadline-stop watchdogs, preflight-stop instructions and `finally: stop()` handlers from2D13/2D14 must not terminate this combined run. They are not inherited authorization to stop the pod after L or on a temporary idle interval.

A single combined completion predicate controls normal automatic shutdown:

```text
L updates==5000 AND R updates==5000
AND all18 new-arm monitor evaluations complete
AND all15 milestone condition evaluations complete and identity/coverage checked
AND required checkpoints/raw results independently verified locally and persistently
AND no remaining assigned GPU work
```

Only then stop the assigned pod and verify EXITED/stopped. CPU bootstrap, final plots/reporting, Git and discussion do not justify holding it after GPU work and safe exports finish. Completion of L, a milestone, one child process or R's training alone is insufficient. Finishing an earlier arm must schedule the next stage rather than trigger shutdown.

If an error or genuine stall occurs while work remains, preserve state, diagnose/recover without changing the scientific recipe, and notify the user if intervention is needed. **Do not silently stop the pod because an estimated time/cost threshold, heartbeat timeout or programming exception fired.** If recovery becomes impossible or continued billing is wasteful, explain the exact blocker and request an explicit stop decision; until then retain the user's reservation. An explicit user stop instruction takes precedence. External provider failures are outside this retention guarantee and must be reported with recovery evidence.

Track the total four-GPU cost from allocation through final verified stop, including held preparation, preflight, all attempts, transition overhead, scoring and transfers. Historical H's first5,000 updates took12,792.263897419 seconds on four GPUs: **3.5534 pod-hours /14.2136 GPU-hours**. L's earlier one-GPU training suggests roughly6.07 GPU-hours for5,000 updates before a new-mode measurement. These are planning anchors, not guarantees of new wall time. A rough combined planning allowance is **6–8 pod-hours /24–32 GPU-hours** including scoring and overhead; use measured preflight throughput to revise it. At the previous $1.59/GPU-hour quote this is approximately$38–51 compute, but verify the current whole-pod rate before allocation.

This is a planning range and soft alert, **not an automatic hard shutdown ceiling**. Notify the user of a materially larger projection before startup, or promptly if it develops during the run. Do not reduce targets, skip evaluations, change precision/geometry or prematurely stop to make a budget estimate appear satisfied. Prepare R fully in advance and overlap exports/local interim analysis with useful GPU work to minimize idle time while honoring the requested reservation.

## 11. Deliverables and closeout

Preserve both initial states; complete resumable L/R checkpoints at u1000/u2000/u5000; actual optimizer-implementation/dispatch audit; full5,000-update hash/LR/cursor evidence for both arms; trainability/gradient/DDP/recovery checks; panel/monitor identities; per-update pass/time/resource logs; all15 raw condition outputs; three interim/final paired tables and trajectories; shared-volume reservation/retention records; independent export hashes; and the combined stop-controller tests and actual final stop evidence.

Final scientific count: **two new arms,5,000 updates each,2,621,440,000 logical targets each**, three reused historical H comparators,15 milestone condition evaluations. The report must distinguish successful execution from whether either scientific quality/dependence goal was met. Preserve the old fused=True results separately and explain why this run was restarted from untrained tensors.

Commit and push only the new package/protocol/results on the new branch; keep large checkpoints outside Git. After both arms, all comparisons, exports and shutdown audit are complete, tag **experiment-2d15-l-r-nonfused-serial-4gpu-final** and verify remote refs. Do not issue a combined final-success tag after only L. Do not launch another experiment or longer continuation after reporting.
