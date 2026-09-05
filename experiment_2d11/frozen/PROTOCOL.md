# EXPERIMENT 2D11 — FROM-SCRATCH O1 + H VERSUS THE SAVED GPT-2 10B CONTROL

## Execution request and scope

Please execute this experiment through preparation, training or a prespecified early stop, evaluation, independently verified backups, prompt GPU shutdown, local analysis and Git push. Prepare everything possible on the local machine before starting paid GPUs. Train one fresh O1 + hierarchical retrieval-aware gating model (H). Reuse the original completed from-scratch Karpathy GPT-2 124M control (G); do not retrain G for 10B tokens.

Use all four GPUs of one user-assigned A100 80GB pod for distributed H training and distribute evaluation work across all four GPUs. This is one training trajectory, not four independently trained models. Do not launch T/D/S arms, extra seeds, window schedules, architecture searches or an extension beyond the endpoint. The selected O1 windows remain fixed. The purpose is to compare learning curves, HellaSwag, final prediction quality and training/inference cost against ordinary GPT-2.

The user specifically wants intermediate reviews around 500M and 1B tokens so a clearly poor run can stop early. Use the matched historical points at **524,288,000** and **1,048,576,000** targets. Describe these as approximately 500M and 1B, with exact counts always recorded. Automatic continuation/stop decisions below prevent an expensive pod waiting for human review.

Operational default: **24 cumulative billed pod-hours / 96 aggregate GPU-hours maximum** for this experiment on four GPUs, including setup, preflight, evaluation, exports and any recovery attempts. This is a proposed cost ceiling, not a runtime guarantee. Record the provider's quoted total pod price before startup; do not interpret a per-GPU quote as the total price. A user-supplied stricter ceiling overrides this default. Do not automatically exceed it, rent additional resources, upgrade storage, or change GPU type/count. If no pod has been assigned, finish local preparation first, then request only the missing pod identification/configuration. Do not launch an arbitrary paid resource.

## 1. Read the evidence and create an isolated workspace

Project root: `/Users/rahul/Documents/GPT-2 Enhancement`.

Read:

- `PROJECT_CONTEXT.md`
- `project_context/FROM_SCRATCH_BASELINE_REUSE_AND_RUNTIME.md`
- `project_context/NEXT_GATING_COMPARISON_AND_BASELINE_ROADMAP.md`
- `baseline_artifacts/gpt2_124m_fineweb10b_20260810T141222Z/FINAL_BASELINE_REPORT.md`, `stage3_launch_report.json`, `metrics.jsonl`, and `baseline_learning_curve.csv`
- `parallel_2d2_master_dev/2d3a_1b/results/experiment_2d10_h_250m/REPORT.md`
- Sealed H/O1 implementation and relevant causal, cache, gradient and checkpointing tests.

Start a new isolated worktree/branch, proposed branch `codex/experiment-2d11-from-scratch-h-vs-gpt2-10b`, from sealed commit `1eda20fc8ee66b4bbfcccae598c8e1354b2d86aa` in `parallel_2d2_master_dev/2d3a_1b`. Preserve all historical branches, tags, reports and checkpoints. Copy this protocol into the new experiment's tracked directory. Resolve name collisions without overwriting existing work.

Authoritative G implementation: original reproduction commit `0f7766f9a0c24e9c2854e698862f0a22c19174c7`, based on upstream Karpathy commit `6104ab1b53920f6e2159749676073ff7d815c1fa`. Current `train_gpt2.py` has later modifications; do not assume it is the original recipe. Import/extract model definitions without accidentally executing the upstream top-level training loop.

## 2. Freeze the control and exact endpoint

Read-only G checkpoint:

```text
/Users/rahul/Documents/GPT-2 Enhancement/baseline_artifacts/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt
SHA256: 924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871
Size: 497,958,271 bytes
```

G has 124,475,904 parameters, 12 blocks, 12 heads, width 768, context 1024 and padded vocabulary 50,304. Original fresh initialization seed: 1337. Recorded hardware was one A100-SXM4-80GB; runtime 18:05:26. Historical final validation CE is 3.0750441551208496 and HellaSwag normalized accuracy is 0.303326030671181. These historical scores are not substitutes for the new paired final evaluation.

**Checkpoint timing matters.** Original evaluation and saving happen before the optimizer update in loop iteration 19072. The saved checkpoint therefore contains **19,072 completed updates / 9,999,220,736 targets**, although the historical process subsequently completed a 19,073rd update. H's primary endpoint is exactly **19,072 updates / 9,999,220,736 targets**. Stop training there. Keep the original 19,073-update learning-rate schedule, evaluated at each executed update's original index; do not shorten/rescale its cosine schedule to 19,072.

G's missing optimizer/RNG/loader state prevents exact continuation, but does not prevent using the completed control. Its existing 78-point learning curves can be reused at matching exposures. Intermediate checkpoints at updates 1000/2000 are not known to exist: do not invent paired historical sequence losses or retrospectively train replacements. Search for any historical checkpoints only during local preparation; absence does not prevent the defined comparison.

## 3. Fresh H architecture and initialization

Keep the tested H computation:

| Destination | Native window | Recurrent writer | Eligible recurrent lags |
|---|---:|---|---|
| B1 | 2 | raw post-MLP B12 residual | 1–1023 |
| B3 | 32 | raw post-MLP B10 residual | 31–1023 |
| B5 | 64 | raw post-MLP B8 residual | 63–1023 |

Other blocks use native W1024; B6 has no B7 recurrence. Preserve three raw 1023-position rings, source identities, position semantics and separate local/recurrent attention. No lag0 recurrence, future-token access, detached training writers, native-window growth, new norm features or architectural tuning.

At each special block, compute actual projected query q and completed pre-c_proj attention outputs A_L/A_R. Concatenate heads in the existing c_proj order. Separately normalize each width-768 vector using affine-free FP32 LayerNorm, epsilon 1e-5. Concatenate to width 2304, apply a 32-unit SiLU hidden layer, then two logits and softmax. Cast coefficients to attention dtype, combine original A_L/A_R, and apply the shared c_proj and bias exactly once. Force exactly (lambda_L, lambda_R)=(1,0) when the valid-memory mask shows no eligible memory. Keep the existing causal incremental semantics and attached multipass training graph.

**Fresh initialization is explicitly different from adaptation.** Instantiate the canonical ordinary GPT-2 base from fresh seed 1337 with the original initialization order, tied embedding/output weights and residual-projection scaling. Add H without changing those base tensors. Do not load learned G, O1, D or H250M weights, moments, trained gate priors or inherited adaptation counters into H.

Use isolated CPU generators seeded `20260916 + zero_based_destination_block` for router W1 ~ Normal(0, .02). Set b1=0, W2=0 and **b2=[0,0]** at all three destinations. Thus eligible-memory queries start with equal branch weights (0.5,0.5); unavailable memory stays local-only. This fixed finite initialization uses no learned gate values and is not tuned using validation. It does not preserve the pretrained O1 function, which is not the objective of fresh training. The old adaptation constructor computes log(tanh(parent g0)) and cannot be blindly called on fresh zero gates; implement a separate explicit fresh initialization path.

Prominently retain this initialization in interim and final interpretation: **“A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.”** The equal mixture applies only where recurrent memory is available. Mature H250M's approximately 25%/1%/4% mean recurrent weights at B1/B3/B5 provide context for this optimization difference, not initial values to import. Keep the existing initialization and futility rules unchanged.

Retain the four legacy gate scalars only for schema compatibility, initialized to zero, computationally unused and excluded from the optimizer with requires_grad=False. H's expected registered count is 124,697,386, of which 124,697,382 are active parameters. Verify actual counts and tied-parameter deduplication. Registered H−G is 221,482; router matrices/biases account for 221,478 and four unused scalars account for the rest. State both counts rather than conflating active and compatibility parameters.

Hash the canonical fresh base weights and H's mapped base weights to prove equality at initialization. Document reproducibility against the historical seed/source; no historical step-zero checkpoint is presumed. Freeze and preserve H's initial state. Training/RNG state must restart from this initialization after all disposable GPU smoke tests.

## 4. Match training exposure and define fresh optimization

Use the original FineWeb-Edu `sample-10BT` token shards, tokenizer, document delimiters and train/validation split. Recorded stored training count is 9,853,989,344 and validation count is 100,000,000. Preserve original shard order, skipped shard tails and loader wrap behavior; processed training targets exceed stored training tokens. Do not replace this with a new shuffled dataset or globally concatenated stream.

Reconstruct a **logical original B64, T1024, world_size=1 loader**. Each update takes its next eight B64 batches in order, then distributes the resulting 512 sequences across ranks and B32 physical microbatches. Do not simply change the original loader's world size to four: its shard-boundary rule depends on B and world size and would change examples. Prove distributed reconstruction equals the logical original loader, including target shifts, sequence boundaries, all shard transitions, wraparound and final cursor. Freeze a compact full-run identity ledger from verified shard metadata and validate actual token hashes as data become available. Do not falsely claim a metadata-only plan has already hashed unavailable remote token content.

```text
GPUs/ranks:                  4
Physical microbatch/rank:    B32 × T1024
Accumulation/rank:          4
Global targets/update:      524,288
Scientific updates:         u = 1 ... 19,072
Optimizer LR index:         i = u - 1
Three-pass updates:         u divisible by 32 (596 updates)
Two-pass updates:           all others (18,476 updates)
```

Count targets once, not once per pass. Two-pass loss weights are (.25,.75); three-pass weights (.20,.40,.40). Preserve attached temporal gradients and non-reentrant activation checkpointing. Fresh cadence starts at update 1, without the adapted checkpoint's inherited offset.

**Required pass accounting, alongside logical targets and GPU-hours:** at the matched endpoint, H executes `18,476×2 + 596×3 = 38,740` CE pass-equivalent global updates, representing `38,740×524,288 = 20,310,917,120` CE pass-target evaluations. These are not additional optimizer updates or distinct training examples.

| Matched-endpoint accounting | G | H |
|---|---:|---:|
| Completed optimizer updates | 19,072 | 19,072 |
| Logical training-target exposure | 9,999,220,736 | 9,999,220,736 |
| CE pass-equivalent global updates | 19,072 | 38,740 |
| CE pass-target evaluations | 9,999,220,736 | 20,310,917,120 |
| Mean CE passes per logical target | 1 | 2.03125 |

Use **logical target exposure**, not “unique targets”: the original stream wraps and repeats some data. Pass counts are unweighted evaluations of the training CE objectives, even though the loss weights sum to one. They do not measure exact FLOPs or imply an exact 2.03125× compute ratio: architecture costs, backward execution and activation-checkpoint recomputation also matter. A pass-equivalent global update is not a literal microbatch forward-call count. Exclude validation, HellaSwag, disposable preflight and recomputation from these scientific CE pass-target counters; retain their actual resource costs, and any failed/replayed work, in the existing execution/GPU-hour accounting.

Record completed updates, logical targets, CE pass-equivalent updates, CE pass-target evaluations and mean CE passes per logical target in checkpoint metadata, interim reports and the final report. For a retained prefix of `u` completed updates, use `logical_targets = 524,288*u`, `H_CE_pass_equivalent_updates = 2*u + floor(u/32)`, and `H_CE_pass_targets = 524,288*(2*u + floor(u/32))`; G's matched-prefix CE pass-target count equals its logical targets. At u=0, report the mean as unavailable rather than divide by zero. At the two planned early reviews, H's expected CE pass-target counts are **1,064,828,928** (u=1000) and **2,129,657,856** (u=2000). An early-stopped run reports its actual prefix, not the full endpoint total.

Use fused AdamW, betas (.9,.95), epsilon 1e-8, global gradient clip 1.0, BF16 autocast with FP32 master parameters. Preserve the original `torch.set_float32_matmul_precision('high')` setting and use `torch.compile=False` for this experiment. Apply the original schedule to **all active groups**, including router biases: 715-update linear warmup to 6e-4, followed by original cosine decay toward 6e-5 with max_steps=19073. Matrices get weight decay .1; vectors/biases get zero. Do not inherit the adaptation LR 3e-5 or its 10× output-bias multiplier. This is a fixed fresh-training recipe, not a claim that its gate initialization or hyperparameters are optimal.

DDP must wrap a forward entrypoint that executes the complete multipass loss graph. Calling `.module.forward_multi_pass` outside DDP's forward is not sufficient. Use no_sync around both forward and backward for the first three accumulation microsteps; synchronize the last. Divide local mean losses by four accumulation steps; account for DDP's rank averaging exactly once. All ranks execute the same pass count and collective order. Handle inactive compatibility parameters explicitly. Do not assume the alternating two/three-pass graph qualifies for static_graph=True.

## 5. Complete local preparation before paid startup

Build a single frozen run bundle containing code, config, manifests, loaders, tests, GPU preflight, trainer, batched evaluators, checkpoint/export workers, supervisor/watchdog, analysis and report templates. Prepare a compatible Linux CUDA environment/image or wheel bundle in advance; do not copy a macOS virtual environment to Linux. Prefer the recorded software versions, including PyTorch 2.8.0+cu128, when compatible; document any necessary environment difference. Do not upgrade libraries on a running training pod.

Resolve local G artifacts and H sources, dataset manifests, provider volume location, disk requirements and HellaSwag data/tokenization before rental. Pre-tokenize and length-bucket all 10,042 HellaSwag validation examples locally, preserving original example IDs, labels, four choices, masks and text processing. Do not use the test split. Freeze input file hashes and original scoring code. Do not keep GPUs running to download datasets, build ordinary Python code, write analysis or debate the protocol.

Reuse the existing persistent volume `yhzyb27fb5` (recorded 160 GB) if it is available to the assigned pod; verify rather than assuming old paths/status are current. Preserve its historical contents. If its filesystem is inaccessible while stopped, prepare from existing manifests and perform the residual checks promptly on startup. Do not boot four GPUs just to browse files or inventory the volume. Do not rent a separate preparation resource without authorization.

Local checks must cover canonical stream reconstruction; seed/base initialization; two/three-pass loss and LR indexing; router simplex/local-only behavior; causal suffix invariance and row isolation with a sequence extending past lag63; attached gradient paths; activation-checkpoint recomputation; checkpoint/RNG/loader restoration; scoring masks; four-way evaluation partition coverage; stop-rule boundaries; and explicit analysis flag rendering after JSON roundtrip. Use reduced-size CPU fixtures where full GPU behavior cannot be tested locally. Mark CUDA/DDP checks pending rather than pretending CPU checks establish GPU correctness.

Prepare the complete 24-point CE and seven-point HellaSwag schedules below. Build a synthetic end-to-end rehearsal of the supervisor covering success, early futility, rank failure, export delay, deadline and shutdown retries, without touching real pods. The expensive phase must not depend on Codex being awake to issue the next command.

Before boot, write and commit `LOCAL_READY.json`, the frozen protocol/config, input manifests, current source commit, local test results, pending CUDA checks, expected disk/transfer requirements, price and duration ceilings, the exact assigned pod ID, startup command and shutdown command. Secrets stay in existing credential facilities, not Git, reports or command output. If the user has not yet assigned a pod, finish all other ready work first.

## 6. Paid-phase startup, preflight and automatic progression

Once ready and within the assigned resource scope, start/resume the pod and immediately run the prepared controller. Log billing/start timestamps and actual GPU models, count, interconnect/topology, driver/runtime, available persistent space and input identities. No interactive setup session should be required. Install/verify the independent shutdown watchdog before long-running GPU work.

Targets: useful CUDA work within approximately 10 minutes of billable startup, and bounded GPU preflight within 20 additional minutes. These are operational budgets, not guarantees. If startup needs substantial downloads, debugging or compilation, stop the pod and repair locally. GPU-specific failures may receive one bounded recovery attempt of at most 10 minutes; do not keep the pod alive through open-ended debugging.

Preflight on disposable state must exercise full-sized B32 two-pass and three-pass backward/update, four-rank gradient agreement, shared initial weights, global-batch equivalence to a single-rank accumulation reference, finite active gradients, correct inactive gates, checkpoint reload and resume across a pass-cadence boundary. Declare numerical tolerances before the checks; report observed maximum/aggregate errors, and do not loosen tolerances to conceal a semantic mismatch. Zero W2 intentionally gives zero W1/b1 gradients on the first backward; verify useful hidden gradients after the output head has updated.

Benchmark steady-state four-GPU H training across both pass counts and a short ordinary GPT-2 four-GPU training reference on the same hardware. Use isolated disposable weights/data. Also time representative incremental CE and **batched** HellaSwag work with representative lengths. Benchmark tests and transient single-rank reference checks are bounded necessary work; do not create long one-GPU jobs that leave the other three GPUs waiting.

Include a bounded G/H incremental-inference benchmark with the same batch, token identities and context length, diagnostic collection disabled for both, and cache priming distinguished from timed token processing. This supports an inference-speed comparison without treating H's diagnostic-inclusive final evaluation time as a clean production-speed measurement. Preserve exact measured settings and label preflight weights as disposable.

Calculate projected remaining full-run time and GPU-hours using measured pass-weighted update times, all scheduled evaluations, checkpoint/transfer costs and a conservative allowance. It must fit within **the remaining billed-time allowance minus two hours of contingency**: for example, after 30 minutes of startup/preflight, at most 21.5 hours of projected remaining work within the 24-hour cap. Include prior failed attempts in elapsed billable time. If it cannot fit, stop after preserving preflight evidence and report the measured alternatives. Do not silently cut required evaluations or change the training method to make an optimistic estimate pass.

After preflight passes, restore the exact frozen fresh H state, empty optimizer state, all scientific RNGs and original training cursor. Start training immediately with no additional confirmation or manual review barrier. Publish the measured estimate asynchronously while GPUs work. Smoke updates and benchmarks do not count toward the scientific budget.

## 7. Fixed monitoring schedule and approximately 500M/1B decisions

Use the original G validation panel: the first 20 batches from the reset **logical original B64** validation loader, exactly 1280 sequences ×1024 targets. Freeze its ordered IDs and hash. Distribute these same examples across four evaluation workers; do not turn 20 batches into 20 batches per rank. H monitoring CE uses true incremental inference. Historical G parallel causal CE is the reference for curves; reproduce its evaluator on available G checkpoints and document numerical-mode differences. These interim historical-aggregate comparisons have no invented paired confidence intervals.

Evaluate H on this fixed CE panel after these completed update counts:

```text
0, 250, 500, 750,
1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
18000, 19000, 19072
```

These are 24 monitoring CE points. Never display the training pass loss as incremental validation. Retain per-update training/pass losses as separately labeled diagnostics.

Evaluate **full 10,042-example HellaSwag** at seven completed update counts:

```text
0, 1000, 2000, 5000, 10000, 15000, 19072
```

The two early reviews have these fixed reference values, to be checked against original JSONL before freeze:

| Review | Completed updates | Targets | G historical CE | G historical HellaSwag |
|---|---:|---:|---:|---:|
| Approximately 500M | 1000 | 524,288,000 | 4.13529109954834 | 0.2550288787094204 |
| Approximately 1B | 2000 | 1,048,576,000 | 3.6628191471099854 | 0.26070503883688506 |

Let `g500 = H_CE_at_1000 - G_CE_at_1000` and `g1 = H_CE_at_2000 - G_CE_at_2000`; positive is worse for H. Define `a1 = H_Hella_at_2000 - G_Hella_at_2000` as a fraction, not percentage points.

Prespecified budget screens:

1. **Severe deficit at approximately 500M:** if `g500 >= 0.50` nats, save an early-stop checkpoint and end the run. The CE corresponds to at least approximately 65% higher perplexity on this panel. Once this rule fires, the scheduled HellaSwag can be skipped to save cost; mark it explicitly as skipped due to the screen.
2. **Persistent poor start at approximately 1B:** stop if all of `g500 >= 0.15`, `g1 >= 0.15`, `g1 >= 0.75*g500`, and `a1 <= 0.005` hold. This means a substantial CE deficit at both checkpoints, less than or equal to 25% closure of the deficit, and no HellaSwag advantage greater than 0.5 percentage points to offset it.
3. Otherwise continue automatically to the final endpoint. A small loss gap or a noisy HellaSwag change alone does not stop training. Numerical failure or integrity corruption follows the failure path independently of these quality screens.

These are conservative **compute-budget heuristics**, not hypothesis tests or proof that H could never recover with longer/different training. Preserve and report them verbatim. Do not move the thresholds after seeing scores. An early-stopped experiment is classified `EARLY_STOPPED_FOR_FUTILITY`, not a completed 10B comparison or evidence that ordinary GPT-2 wins at matched 10B. Do not compare an early H checkpoint directly against G10B as if exposures matched.

At each early review, atomically save/export a complete checkpoint and small `INTERIM_500M.json/md` or `INTERIM_1B.json/md` containing both scores, exact targets, gaps, decision, runtime, GPU-hours and revised completion estimate. The controller evaluates these simple rules immediately; reports/plots are generated locally while training continues. No paid wait for a user reply. The user may explicitly stop the run through the controller; do not repeatedly request permission to continue after a passing screen.

## 8. Efficient GPU evaluation and a fresh paired final comparison

During evaluation, every GPU gets independent sequence/example work from a deterministic four-worker plan or balanced queue. Keep scientific examples fixed while balancing HellaSwag lengths. Record ownership and verify every expected ID appears exactly once after merge, with no duplicates or omissions. Aggregate results by canonical ID, not completion order. Evaluation needs no training-gradient synchronization; avoid DDP forward collectives that can deadlock with unequal worker lengths.

HellaSwag must score the original four candidate completions with average next-token CE over completion tokens, preserving the original prompt/mask boundary and first-completion-token prediction. Reproduce the original `get_most_likely_row` scoring dtype and reduction behavior for compatibility with the historical curve, and apply it consistently to both G and H. Do not silently change scoring precision between historical-compatibility monitoring and the final H evaluation that will be reused. Record execution-mode/rounding differences from historical G separately. Use true incremental H execution; never substitute multipass approximate validation. Batch independent examples/choices and right-pad only beyond actual ends, masking scores correctly. No cross-choice cache aliasing, prompt-cache mutation shared across alternatives, truncation, choice leakage or per-example serial GPU launches when batching is possible. Validate batching against a small simple reference locally and on GPU. Preserve original IDs and per-choice normalized scores/predictions. Use current trained H parameters at each checkpoint, with resets between independent sequences.

Physical evaluation batch size may be chosen during bounded preflight from memory-fitting candidates using measured speed and row-isolation checks. Freeze it before scientific scoring; examples, sequence lengths, masks and target counts must not change. A canonical B64 panel is an identity convention, not a requirement to leave GPU memory or evaluation throughput unused.

At final H endpoint, evaluate a **separately frozen fresh panel of 4096 sequences ×1024 targets** for both G and H. Use the established validation shard SHA `8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`, canonical B64 batch selection and isolated NumPy default_rng seed 20260917. Exclude the historical monitoring prefix and all recoverable previously used/reserved target spans, including 2D9/2D10 panels. Freeze IDs and disjointness evidence locally before scientific training; do not choose examples based on scores. If a fresh panel cannot be constructed, resolve the protocol locally before paid startup.

Use validated true incremental G/H implementations for the final CE comparison, BF16 forward, FP32 token CE and FP64 accumulation. Reset state identically between sequences. Save paired ordered per-sequence NLL/CE and counts, plus checkpoint/panel/code identities. Validate the ordinary G incremental implementation against its original causal parallel model; record actual numerical tolerances/errors. The new paired G score, not its old 3.075044 aggregate, is the primary comparator.

Final H HellaSwag is the already scheduled update-19072 evaluation; do not repeat it for analysis. Evaluate saved G once with the same frozen data, completion scoring and validated execution semantics to obtain paired per-example results. Reuse the resulting G scores in the final comparison and report any difference from historical 30.3326%. Do not rerun H's final evaluation merely to add diagnostics or plots.

Collect compact final H gate means/std/ranges/entropy and parameter norms during its final fresh CE evaluation, with separate eligible-memory and forced-local positions. Avoid attention matrices, hidden-state dumps, per-token CPU synchronization or a separate diagnostics evaluation. Confirm persistent BF16 state at B=1: H 33,289,728 bytes versus ordinary G 37,711,872 under the same W−1 historical KV accounting; distinguish this from weights, transient workspace and peak training VRAM.

If a quality screen ends training early, skip the final 10B comparison workload. Export the early checkpoint and completed matching interim metrics, then shut down. There is no requirement to manufacture a fresh paired early panel without an exposure-matched historical G checkpoint.

## 9. Prevent avoidable idle time and protect paid progress

Use a persistent supervisor independent of chat turns. Predefine the full state machine:

```text
LOCAL_PREP -> READY -> STARTUP -> CUDA_PREFLIGHT
 -> RESTORE_FRESH_STATE -> TRAIN
 -> SCHEDULED_EVALUATION -> AUTO_CONTINUE_OR_EARLY_STOP
 -> FINAL_EVALUATIONS (only if final endpoint reached)
 -> FINAL_EXPORT_VERIFICATION -> STOP_POD -> LOCAL_ANALYSIS
```

Training, evaluation, checkpoint/export and shutdown transitions are automated. The local controller polls/downloads compact progress in the background and can generate plots without interrupting GPU jobs. All four training ranks remain allocated for H training; do not reserve one GPU for analysis or monitoring.

Collect short progress heartbeats with state, completed update/evaluation batch, last successful write and stage deadline. Brief low GPU utilization during synchronization or a coherent checkpoint is expected and must not itself trigger shutdown. Trigger failure handling on exited ranks, missing job progress beyond a bounded stage deadline, a stalled controller or a completed queue with no remaining GPU work. A training job with no completed update for five minutes is suspect; respect specifically bounded save/evaluation stages. Do not allow an unexplained idle/debugging state to persist beyond five minutes before beginning checkpoint/stop handling.

Implement best-effort independent shutdown protection for this exact assigned pod using existing credentials/control tools. The watchdog must be able to stop the assigned experiment pod after job failure or billing deadline even if the Codex conversation disconnects. Test its logic without stopping unrelated resources. No deletion of the persistent volume. An unresolved stop API failure must produce retries and an explicit prominent alert; process exit is not proof that billing stopped.

Checkpoint complete scientific state every 500 completed updates and at all major reviews. Preserve initialization, 1000, 2000, 5000, 10000, 15000 and final checkpoints; retain the latest two rolling recovery checkpoints. Only prune this experiment's older redundant rolling checkpoints after successor verification. Never delete historical/user data. If interrupted, restore exact model, optimizer groups/moments/counters, LR position, every rank's RNG, logical loader cursor and completed-evaluation ledger; avoid duplicated/skipped scientific updates or evaluations.

Save at a coherent optimizer boundary. An asynchronous writer must receive an immutable CPU snapshot; it must not serialize tensors that ongoing training mutates. Atomic rename only after complete write. Preserve scientific RNG around all evaluation and checkpoint operations. Prove a disposable resume reproduces the uninterrupted next step within declared numerical tolerances. Optimizer counters advance from zero for fresh active parameters; legacy compatibility parameters never acquire optimizer state.

Copy and hash checkpoints/artifacts to the user's local archive **while later GPU training/evaluation proceeds**, with bounded transfer/CPU workers. Reuse saved persistent storage as the remote independent copy. Benchmark bandwidth early; prioritize already completed immutable checkpoints so the final transfer tail stays short. Log transfer overhead; do not consume hours rehashing every prior experiment on a paid pod.

Avoid full GPU weight/optimizer CPU copies and full-state SHA scans every update. Use cheap device-side finite checks, normal clipping/gradient synchronization, compact loss logging and CPU data-identity ledgers during training; perform full state audits at checkpoints and completion. Reduce per-microbatch host synchronization. Full checkpoint integrity remains required, but its implementation must not dominate training time.

Use a soft billing deadline with at least 30 minutes reserved for a safe checkpoint/export/shutdown. The 24-hour cap spans all attempts. If progress cannot finish within the remaining ceiling, stop at a verified resumable state and classify `BUDGET_STOPPED`, not complete. Do not keep the pod running while seeking an increased budget. The independent hard deadline must still stop billing if a local transfer is unavailable: ensure the atomic persistent checkpoint exists, record backup as incomplete, and recover from persistent storage later under a separately authorized plan. Never claim independently verified local backups when this fallback occurs.

As soon as GPU work finishes, complete only the required final artifact transfer and verification, then stop the pod. Verify provider status as stopped/EXITED rather than merely checking nvidia-smi. Perform bootstraps, charts, report writing, Git operations and user discussion locally after shutdown. Report useful training, GPU evaluation, startup, checkpoint/export, failure/recovery and idle minutes separately; include full billed wall time and aggregate GPU-hours. Minimize idle time, but do not claim that four GPUs guarantee lower total compute cost.

## 10. Local analysis, interpretation and deliverables

For completed 10B comparisons, use paired per-sequence final CE contrast `G - H` and paired per-example HellaSwag accuracy contrast `H - G`; positive favors H in both. Compute 50,000 paired bootstrap resamples for each endpoint, isolated seeds 20260918 and 20260919, chunked locally with frozen NumPy version and linear percentiles. Do not resample individual CE tokens or unpair model predictions.

Report ordinary 95% intervals and Bonferroni-adjusted **97.5% marginal intervals** (percentiles 1.25/98.75) for the two final quality endpoints, with the latter governing joint claims. Report CE difference, relative perplexity change, HellaSwag difference in percentage points, per-sequence wins/ties and paired HellaSwag correctness table. No CI should be invented for intermediate comparisons where only G's historical aggregates exist.

Classify each endpoint separately as benefit, harm or unresolved from the adjusted interval's relation to zero. Report the historical CE margin .0001 as an additional small-effect reference, explicitly not a definition of a substantial improvement worth several times the compute. Do not automatically declare overall superiority from one favorable endpoint if the other is unresolved or worse. Describe gains and their training/inference costs. These are single training trajectories, not seed replication; this experiment compares complete architecture-plus-training recipes, including H's multipass compute, rather than isolating the gate's causal contribution.

Generate scientific PNG/PDF plots and CSV/JSON data for:

- Historical G versus fresh H validation CE/perplexity versus processed targets, with matching points identified.
- HellaSwag versus processed targets using all available G points and the seven planned H points; label sparse H sampling and omit unavailable points after an early stop.
- Training objectives separately labeled, avoiding a claim that H's weighted multipass loss is the same objective as G's single-pass loss.
- Quality versus measured cumulative GPU-hours and elapsed time, clearly displaying GPU count/hardware and historical versus new measurements.
- Training throughput, evaluation/inference throughput, peak training VRAM, persistent state and parameter counts.

A four-GPU H wall-time improvement over historical one-GPU G is not an architecture speedup. Include the bounded same-hardware G/H benchmark, distinguish measured full-run time from projected G four-GPU time, and separate training-only from evaluation-inclusive costs. Report exact processed targets even where chart labels say 500M/1B/10B.

Required artifacts include frozen protocol/config/manifests and implementation commit; initial-state identity; baseline identity; distributed data plan and actual replay/terminal checks; CPU/CUDA test results; measured preflight and cost projection; per-update metrics; evaluation ID/prediction/loss files; interim reports and stop-rule inputs/decisions; complete checkpoint manifests and independently verified local/persistent hashes; recovery/deadline records; provider stopped-status evidence; resource accounting; charts/data; audit JSON and final Markdown report.

For a successful full run, suggested final tag is `experiment-2d11-from-scratch-h-vs-gpt2-10b-final`. For a clean early screen, use an explicitly early-stopped tag such as `experiment-2d11-from-scratch-h-1b-futility-final`; do not label it a completed 10B experiment. Push the new branch and appropriate immutable result tag after local analysis, verifying remote identities. Do not put model checkpoints or datasets in Git. Do not change old tags.

Finish with the exact outcome (completed, early-futility-stopped, budget-stopped, or invalid/incomplete), comparison metrics at matching exposures, actual runtime/GPU-hours and idle breakdown, checkpoint locations/hashes, verified pod state, Git identities and the strongest conclusion supported. No further training is authorized by this handoff.
