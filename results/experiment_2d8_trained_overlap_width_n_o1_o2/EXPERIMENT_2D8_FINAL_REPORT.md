# Experiment 2D8 Final Report

## Opening summary

Classification: **O1 CONFIRMED; WIDER O2 OVERLAP ADDS NO ESTABLISHED BENEFIT**

Experiment 2D8 trained exactly one new arm, O2, from the sealed 2D6 parent and compared it with the existing trained 2D7 N and O1 checkpoints. All three models were scored on one genuinely fresh, disjoint, pre-sealed panel of 4,096 matched sequences.

The fresh panel independently replicated the O1 advantage over N:

- N − O1: `+0.000154525486` CE
- Paired 95% CI: `[+0.000103576253, +0.000205237518]`
- Conclusion: O1 is superior to N beyond the preregistered `delta_CE = 0.0001` margin.

Increasing overlap at B3 and B5 did not improve O1:

- O1 − O2: `−0.000004652351` CE
- Paired 95% CI: `[−0.000051136085, +0.000041085229]`
- Conclusion: practical equivalence is established; O1 is slightly better numerically.

O2 nevertheless remained statistically better than N:

- N − O2: `+0.000149873135` CE
- Paired 95% CI: `[+0.000099390197, +0.000200113966]`
- Conclusion: O2 is statistically superior to N, but the lower confidence bound misses the practical-margin criterion by approximately `0.000000609803` CE.

The fresh numerical ranking is:

1. O1
2. O2
3. N

The preferred architecture is **O1**. It is now independently confirmed over N, while the wider O2 intervention provides no established gain and does not reduce persistent-state storage.

Final audit: **PASS (35/35 checks)**.

GPU status: **STOPPED**. Persistent network volume `yhzyb27fb5` was retained.

## Decision summary

| Question | Result | Decision |
|---|---|---|
| Did fresh data replicate O1 > N? | Yes; CI entirely above `+0.0001` | Confirm O1 |
| Does O2 improve over O1? | No; O1/O2 practical equivalence established | Do not widen |
| Does O2 improve over N? | Statistically yes; beyond-margin superiority narrowly unresolved | O2 remains viable, but not preferred |
| Is `O2 < O1 < N` supported? | No; observed order is `O1 < O2 < N` | No monotonic dose response |
| Preferred geometry | O1 | Adopt O1 |
| Further O3/O4 testing | Not currently warranted | Stop overlap-width expansion |

## Scientific question

Experiment 2D7 found that a one-token boundary overlap, O1, was statistically better than the non-overlap baseline N, but it did not establish superiority beyond `delta_CE = 0.0001` on its original 2,048-sequence panel.

Experiment 2D8 asked four narrow questions:

1. Does a fresh panel independently replicate O1 over N?
2. Does wider overlap at B3/B5 improve O2 over O1?
3. Does O2 improve over N?
4. Do the data support a monotonic overlap-width dose response?

The experiment was preregistered as a narrow N/O1/O2 comparison. It trained only O2, used one fresh panel, evaluated exactly three REAL conditions, and prohibited follow-up arms and post-result GPU diagnostics.

## Geometry definitions

The source-position convention was hard-locked:

`source position j = t - lag`

No `j−1`, `j+1`, current-token, or future-token substitution was permitted.

| Arm | B1 local | B1 recurrent | B3 local | B3 recurrent | B5 local | B5 recurrent |
|---|---|---|---|---|---|---|
| N | 0–1 | 2–1023 | 0–31 | 32–1023 | 0–63 | 64–1023 |
| O1 | 0–1 | 1–1023 | 0–31 | 31–1023 | 0–63 | 63–1023 |
| O2 | 0–1 | 1–1023 | 0–31 | 30–1023 | 0–63 | 62–1023 |

O2 is not a two-token-overlap intervention at every recurrent destination:

- B1 is identical to O1 and overlaps only at lag 1.
- B3 overlaps at lags 30 and 31.
- B5 overlaps at lags 62 and 63.
- The maximum recurrent lag remains 1023 in every recurrent block.
- No old recurrent candidates were dropped to preserve candidate count.

## Common architecture

All arms used the accepted post-2D6 architecture:

| Block | Local path | Recurrent path |
|---|---|---|
| B1 | W2 | B12→B1 through lag 1023 |
| B2 | W1024 | none |
| B3 | W32 | B10→B3 through lag 1023 |
| B4 | W1024 | none |
| B5 | W64 | B8→B5 through lag 1023 |
| B6 | W1024 | no B7→B6 recurrence |
| B7–B12 | W1024 | no additional recurrent readers |

The separate local and recurrent softmaxes were preserved. No joint/common softmax was introduced.

The existing trainable recurrent gates `g_rec`, `g_rec_b3`, and `g_rec_b5` were restored from the parent with their inherited values and optimizer state. They remained trainable. No new parameters or gates were added. Parameter count remained `124,475,908`.

## Provenance and checkpoint lineage

### Sealed 2D6 parent

- Remote path: `/workspace/exp2d6_b6_native_100m/checkpoints/scientific_cumulative_001100480512.pt`
- SHA-256: `6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7`
- Global update: `2099`
- Cumulative targets: `1,100,480,512`

O2 was independently initialized from this parent. It was not initialized from N, O1, G, 2D5C, Routed, or another earlier checkpoint.

### Existing N checkpoint

- Remote path: `/workspace/exp2d7_boundary_alignment/run/checkpoints/N/scientific_cumulative_001200619520.pt`
- SHA-256: `57e62a2094693205b520e2986047d46c28d042d4ec34d6e65b2135f474adec20`
- Global update: `2290`
- Cumulative targets: `1,200,619,520`
- Status: strict-load PASS; not retrained

### Existing O1 checkpoint

- Remote path: `/workspace/exp2d7_boundary_alignment/run/checkpoints/O/scientific_cumulative_001200619520.pt`
- SHA-256: `c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6`
- Global update: `2290`
- Cumulative targets: `1,200,619,520`
- Stored 2D7 arm name: `O`
- Status: strict-load PASS; not retrained

### Sealed 2D7 continuation

- Manifest SHA-256: `f15a5de4b5428031adfe8877f01e6487dcdfc6749e337f552feb7c6f92e9cc4d`
- Ledger SHA-256: `555ac4b4425fcd711edf2e923412ecfac1db49802653570fe56b02ae4139c1aa`
- Logical global batches: `191`
- Update range: `2100–2290`
- Targets per update: `524,288`
- New targets: `100,139,008`

O2 consumed the exact sealed 2D7 logical continuation, not a regenerated similar stream.

## Minimal preflight

The preflight was authorized only after every required check passed.

### State restoration

| Check | Result |
|---|---|
| Parent SHA and counters exact | PASS |
| Strict model-state load | PASS |
| Optimizer restored and finite | PASS |
| Scheduler restored | PASS |
| RNG restored without reseeding | PASS |
| Loader restored | PASS |
| Parameter count unchanged | PASS |
| Recurrent gates inherited and trainable | PASS |
| N checkpoint exact and strict-loadable | PASS |
| O1 checkpoint exact and strict-loadable | PASS |

### Runtime boundary audit

| Boundary | Expected behavior | Result |
|---|---|---|
| B1 lag 0 | local only | PASS |
| B1 lag 1 | local + recurrent | PASS |
| B1 lag 2 | recurrent only | PASS |
| B3 lag 29 | local only | PASS |
| B3 lag 30 | local + recurrent | PASS |
| B3 lag 31 | local + recurrent | PASS |
| B3 lag 32 | recurrent only | PASS |
| B5 lag 61 | local only | PASS |
| B5 lag 62 | local + recurrent | PASS |
| B5 lag 63 | local + recurrent | PASS |
| B5 lag 64 | recurrent only | PASS |
| B1/B3/B5 lag 1023 | recurrently eligible | PASS |
| Source identity | `j=t-lag` | PASS |
| Future/current recurrent access | absent | PASS |
| B6 | native W1024 | PASS |
| B7→B6 | absent | PASS |

### Disposable smoke

- Official optimizer updates: `0`
- Forward loss: `13.061607360840`
- Output shape: `[1, 70, 50304]`
- Finite loss and gradients: PASS
- Nonzero finite B1/B3/B5 recurrent-gate gradients: PASS
- Dormant B6 gate gradient absent: PASS
- Scientific continuation state altered: no

The first real O2 logical batch, loader cursor, stream identity, and global update all matched the sealed continuation before training began.

## O2 training execution

O2 was the only newly trained arm.

| Metric | Value |
|---|---:|
| Hardware | 1 × NVIDIA A100-SXM4-80GB |
| DDP | disabled |
| Starting global update | 2099 |
| First new update | 2100 |
| Final global update | 2290 |
| Optimizer updates | 191 |
| Targets per update | 524,288 |
| New targets | 100,139,008 |
| Final cumulative targets | 1,200,619,520 |
| Objective | CE only |
| Training wall time | 1,678.499 s / 27.975 min |
| Mean throughput | 59,993.9 targets/s |
| Scientific checkpoints written | final checkpoint only |

The inherited optimizer, scheduler, RNG, loader, precision, gradient accumulation, pass schedule, learning-rate semantics, and all trainable parameters were preserved. There was no optimizer reset, scheduler reset, warmup reset, RNG reseed, parameter freezing, auxiliary loss, or extra training.

All 191 training-log rows were present and covered updates 2100–2290 exactly. Active gradient groups remained finite and nonzero. The final checkpoint strict-reopen audit passed all 14 checks.

## O2 final checkpoint

- Remote path: `/workspace/exp2d8_overlap_width/run/checkpoints/O2/scientific_cumulative_001200619520.pt`
- Local archive: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d8_trained_overlap_width_n_o1_o2/O2/scientific_cumulative_001200619520.pt`
- Size: `1,493,951,755` bytes
- SHA-256: `792b85f99164e8d1a096e2913e9d116f7f544e84d63c5a48c9a20e136cd9b69f`
- Global update: `2290`
- Cumulative targets: `1,200,619,520`
- Optimizer/scheduler/RNG/loader state included: yes
- Remote and local SHA equality: PASS

## Terminal stream identity

| Identity | Required | O2 observed | Result |
|---|---|---|---|
| Final loader cursor SHA | `682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3` | exact match | PASS |
| Next-global-batch SHA | `a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe` | exact match | PASS |
| Next-stream SHA | `7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567` | exact match | PASS |

Matched-training status: **MATCH**.

## Fresh evaluation panel

The panel was constructed only after O2 had been sealed and before any model was scored.

| Property | Value |
|---|---|
| Panel name | fresh disjoint 2D8 matched panel |
| Dataset | `edu_fineweb10B/edufineweb_val_000000.npy` |
| Dataset SHA-256 | `8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f` |
| Split | validation |
| Selection seed | `20260904` |
| Candidate panels constructed | 1 |
| Batches | 64 |
| Sequences per batch | 64 |
| Total sequences | 4,096 |
| Targets per sequence | 1,024 |
| Targets per condition | 4,194,304 |
| Panel SHA | `7a32c6b84eb09f453af5e8b4e119e2edd4f401cdee8662744919f7307391702f` |

Selection used a seeded permutation of complete canonical 64×1024 validation batches. It rejected every recovered historical target span and the reserved prefix of batches 0–127, then accepted the first 64 eligible batches.

Disjointness evidence:

- Training data were disjoint by dataset split.
- `239` prior/reserved target spans were recovered and excluded before selection.
- Historical provenance was recovered from five JSON artifact sources.
- All selected batch indices were unique.
- All 4,096 sequence identities were unique.
- No checkpoint loss was inspected during panel selection.
- The manifest was frozen and copied locally before scoring.
- Panel disjointness audit: PASS.

## Evaluation execution

Exactly three conditions were evaluated, sequentially, using the same frozen panel:

1. `N_REAL`
2. `O1_REAL`
3. `O2_REAL`

Every condition used identical sequence identities, source offsets, target positions, ordering, BF16 model execution, FP32 token CE, FP64 accumulation, true-incremental evaluation, and cache semantics.

No G, OFF, SHUFFLED, route-off, HellaSwag, per-link ablation, alternate panel, additional seed, representation analysis, or attention visualization was run.

| Condition | Checkpoint SHA verified | Sequences | Targets | Wall time | Peak allocated VRAM | Peak reserved VRAM | Result |
|---|---|---:|---:|---:|---:|---:|---|
| N_REAL | yes | 4,096 | 4,194,304 | 19.007 min | 5,020.7 MiB | 23,694 MiB | PASS |
| O1_REAL | yes | 4,096 | 4,194,304 | 19.282 min | 5,020.6 MiB | 22,644 MiB | PASS |
| O2_REAL | yes | 4,096 | 4,194,304 | 19.058 min | 5,021.2 MiB | 22,598 MiB | PASS |

Total sequential evaluation time was approximately `57.35` minutes.

## Primary fresh-panel results

| Arm | Mean CE | Perplexity |
|---|---:|---:|
| N | 3.030318233011 | 20.703820180166 |
| O1 | **3.030163707525** | **20.700621159467** |
| O2 | 3.030168359876 | 20.700717466242 |

Numerical CE ranking:

1. O1
2. O2
3. N

The primary statistics used a paired per-sequence percentile bootstrap:

- Resamples: `50,000`
- Sampling unit: sequence
- Paired sequences: `4,096`
- Bootstrap seed: `20260902`
- Shared resample indices across all three contrasts: yes
- Confidence interval: 95% percentile interval

Positive `A−B` means B has lower CE.

| Contrast | Estimate | Paired 95% CI | Perplexity ratio `exp(A−B)` | Classification |
|---|---:|---:|---:|---|
| N − O1 | +0.000154525486 | [+0.000103576253, +0.000205237518] | 1.000154537425 | O1 superior beyond `delta_CE` |
| O1 − O2 | −0.000004652351 | [−0.000051136085, +0.000041085229] | 0.999995347660 | Practical equivalence established |
| N − O2 | +0.000149873135 | [+0.000099390197, +0.000200113966] | 1.000149884367 | O2 statistically superior; beyond-margin superiority not established |

## Per-sequence wins

A win means the named arm had the lower per-sequence CE.

| Pair | First-arm wins | Second-arm wins | Ties |
|---|---:|---:|---:|
| N vs O1 | N: 1,891 | O1: 2,205 | 0 |
| O1 vs O2 | O1: 2,070 | O2: 2,026 | 0 |
| N vs O2 | N: 1,886 | O2: 2,210 | 0 |

The win counts agree directionally with the aggregate ranking but are not used as a substitute for the paired confidence intervals.

## Practical-difference interpretation

The preregistered practical margin was:

`delta_CE = 0.0001`

- O1 vs N: the lower bound is `+0.000103576253`, so O1 superiority extends beyond the practical margin.
- O2 vs N: the lower bound is `+0.000099390197`. This is positive, establishing statistical superiority, but it is `0.000000609803` below the beyond-margin threshold.
- O1 vs O2: the entire CI lies inside `[−0.0001,+0.0001]`, establishing practical equivalence.

The O1/O2 point estimate favors O1 by `0.000004652351` CE, but the recommendation does not treat that tiny numerical difference as a meaningful superiority claim.

## N-vs-O1 independent confirmation

The original 2D7 N/O1 arrays were reused only for secondary analysis. They were not reevaluated.

The saved arrays reproduced the original 2D7 result:

| Panel | Sequences | N − O1 | 95% CI |
|---|---:|---:|---|
| Old reused sealed 2D7 panel | 2,048 | +0.000149591044 | [+0.000077991189, +0.000221717781] |
| Fresh primary 2D8 panel | 4,096 | +0.000154525486 | [+0.000103576253, +0.000205237518] |

The fresh estimate differs from the old estimate by only `+0.000004934442` CE and independently crosses the practical-margin threshold.

## Secondary stratified pooled analysis

The pooled analysis used:

- Old paired sequences: `2,048`
- Fresh paired sequences: `4,096`
- Total paired sequences: `6,144`
- Resamples: `50,000`
- Independent resampling within each panel
- Fixed sequence-count weights: old `1/3`, fresh `2/3`
- Bootstrap seed: `20260902`

| Quantity | Estimate | 95% CI |
|---|---:|---:|
| Old-panel N − O1 | +0.000149591044 | stratified replicate CI [+0.000077646868, +0.000221257153] |
| Fresh-panel N − O1 | +0.000154525486 | stratified replicate CI [+0.000103624127, +0.000204893070] |
| Pooled N − O1 | **+0.000152880672** | **[+0.000110933481, +0.000193975395]** |

Pooled classification: **O1 superior to N beyond `delta_CE`**.

The primary evidence remains the fresh panel. The pooled result is supporting confirmation and does not replace it.

## Panel heterogeneity

The preregistered heterogeneity estimator was:

`H = D_fresh - D_old`

Observed result:

- H: `+0.000004934442`
- 95% bootstrap CI: `[−0.000082897880, +0.000092526309]`

The interval includes zero. No material panel heterogeneity is established, and there is no directional reversal: both panels favor O1 by nearly the same amount.

## Persistent-state accounting

| Arm | Physical bytes/sequence | Difference vs N | Difference vs O1 |
|---|---:|---:|---:|
| N | 33,289,728 | 0 | 0 |
| O1 | 33,289,728 | 0 | 0 |
| O2 | 33,289,728 | 0 | 0 |

O2 increases eligible B3/B5 attention candidates but does not extend the retained lag-1023 horizon. As measured, it requires no additional persistent historical context.

## Scientific interpretation

### 1. Did fresh data replicate the O1 advantage over N?

**Yes.** The fresh N−O1 estimate is `+0.000154525486` with CI `[+0.000103576253,+0.000205237518]`. The lower bound exceeds `+0.0001`, upgrading the previous statistical result to an independently replicated advantage beyond the preregistered practical margin.

### 2. Is O2 better than O1?

**No established improvement.** O1−O2 is `−0.000004652351` with CI `[−0.000051136085,+0.000041085229]`. The interval establishes practical equivalence, and the numerical point estimate slightly favors O1.

### 3. Does O2 improve over N?

**Statistically, yes.** N−O2 is `+0.000149873135` with CI `[+0.000099390197,+0.000200113966]`. O2 is statistically better than N, but superiority beyond `0.0001` is not established because the lower bound is narrowly below the threshold.

### 4. Is there an overlap-width dose response?

**No.** The hypothesized dose-response ordering `O2 < O1 < N` was not observed. The actual fresh ordering is `O1 < O2 < N`, and O1/O2 practical equivalence is established.

### 5. Is O1 the optimum overlap width?

The evidence supports O1 over the tested alternatives N and O2. It does not globally establish a one-token optimum for every recurrent layer, because O2 widened overlap only at B3 and B5 while B1 remained identical to O1.

## Architecture recommendation

Preferred geometry: **O1**.

Reasoning:

1. O1 independently establishes superiority over N beyond the practical margin.
2. O2 does not establish any improvement over O1.
3. O1 and O2 have identical persistent-state costs.
4. O1 changes fewer eligibility boundaries and is therefore the narrower supported intervention.
5. The old and fresh N/O1 panels agree closely, with no established heterogeneity.

O1 should become the preferred trained boundary-alignment geometry among N, O1, and O2.

## Is further overlap-width testing warranted?

**No, not currently.**

Wider B3/B5 overlap is practically equivalent to O1 and slightly worse numerically. Experiment 2D8 therefore supplies no positive evidence for continuing to O3/O4-style expansion. Additional GPU expenditure should not be escalated automatically.

No follow-up experiment was launched.

## Scope compliance

The run performed exactly:

- one new O2 training arm;
- 191 optimizer updates;
- one fresh 4,096-sequence panel;
- three fresh REAL evaluations: N, O1, O2;
- one fresh shared-index bootstrap;
- one secondary stratified N/O1 pooled bootstrap;
- one panel-heterogeneity calculation;
- one minimal persistent-state accounting.

It did not perform:

- N or O1 retraining;
- G training or evaluation;
- O3, O4, or any wider-overlap arm;
- fixed-recurrent-width variants;
- B1/B3/B5 decomposition;
- joint/common softmax;
- source-index shifting;
- OFF, SHUFFLED, or route-off evaluation;
- HellaSwag;
- per-head or representation diagnostics;
- attention visualization;
- additional seeds or panels;
- more than 191 O2 optimizer updates;
- automatic follow-up work.

## Storage operations

The retained network volume was close to its original 150 GB quota before execution.

Before training, storage was cleaned only under the user's explicit authorization:

- Four remote checkpoint copies were removed only after their SHA-256 values exactly matched retained local backups:
  - 2D7 G final checkpoint
  - 2D5C milestones at cumulative targets 1,025,507,328; 1,050,673,152; and 1,075,838,976
- One disposable failed-preflight checkpoint was removed.
- One zero-byte temporary file was removed.
- The sealed 2D6 parent, 2D7 N, 2D7 O1, compact provenance, and result artifacts were retained.
- No unique scientific checkpoint was deleted.
- The user subsequently reported expanding the volume to 160 GB.
- The final O2 checkpoint and all remote result artifacts remain on volume `yhzyb27fb5`.
- A hash-verified O2 checkpoint backup is also retained locally.

Storage audit artifact: `STORAGE_CLEANUP_MANIFEST.json`.

## GPU shutdown

- Pod name: `serious_indigo_swordfish`
- Pod ID: `e8nd7m6piw5km2`
- GPU: 1 × A100-SXM4-80GB
- User reported stop completion: yes
- Independent SSH probe: connection refused
- Confirmation time: `2026-09-03T14:34:44Z`
- Persistent volume deleted: no
- Final status: **GPU STOPPED: confirmed**

The 50,000-resample statistics, pooled analysis, heterogeneity calculation, and report generation were performed locally only after shutdown confirmation.

## Test and validation evidence

- Pod unit tests: `8 passed`
- Preflight: PASS
- O2 training summary: PASS
- Final checkpoint strict reopen: PASS
- Joint three-evaluation identity/size audit: PASS
- Old 2D7 effect reproduction: PASS
- Fresh shared-index bootstrap: PASS
- Stratified pooled bootstrap: PASS
- Final machine-readable audit: PASS
- Local diff-format check: PASS
- Local O2 checkpoint rehash: exact
- Working tree after scientific result commit: clean

The local Python environments did not contain `pytest`; the authoritative test run was completed on the pod before scientific execution.

## Final audit

All 35 machine-readable final checks passed.

| Audit check | Result |
|---|---|
| parent_sha_exact | PASS |
| parent_counters_exact | PASS |
| o2_independent_from_sealed_2d6 | PASS |
| n_sha_exact_strict | PASS |
| o1_sha_exact_strict | PASS |
| n_o1_not_retrained | PASS |
| b6_w1024_b7_recurrence_absent | PASS |
| separate_softmax_preserved | PASS |
| existing_gates_preserved_trainable | PASS |
| no_new_gates_or_parameters | PASS |
| o2_minima_exact | PASS |
| o2_runtime_boundaries_exact | PASS |
| no_source_shift_or_future_access | PASS |
| continuation_hashes_exact | PASS |
| preflight_pass | PASS |
| first_batch_exact | PASS |
| training_exact_191_updates | PASS |
| training_targets_exact | PASS |
| training_continuation_exact | PASS |
| inherited_state_no_reset | PASS |
| ce_only_no_freezing | PASS |
| final_counters_exact | PASS |
| terminal_stream_exact | PASS |
| o2_checkpoint_sealed_exported | PASS |
| fresh_panel_disjoint_frozen | PASS |
| fresh_panel_exact_size | PASS |
| same_ordered_fresh_panel | PASS |
| three_4096_loss_arrays | PASS |
| exactly_three_allowed_evaluations | PASS |
| fresh_shared_bootstrap_50000 | PASS |
| old_arrays_reproduce_2d7 | PASS |
| stratified_bootstrap_50000 | PASS |
| heterogeneity_reported | PASS |
| persistent_state_accounted | PASS |
| gpu_stopped_volume_retained | PASS |

Final audit status: **PASS**.

## Artifact inventory

Primary result directory:

`/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b/results/experiment_2d8_trained_overlap_width_n_o1_o2`

Key files:

- `EXPERIMENT_2D8_FINAL_REPORT.md` — this full report
- `EXPERIMENT_2D8_CONCISE_REPORT.md` — machine-generated concise result
- `SCIENTIFIC_RESULT_SUMMARY.json` — complete structured result
- `FINAL_AUDIT.json` — 35-check final audit
- `FRESH_PAIRED_BOOTSTRAP.json` — primary 50,000-resample analysis
- `STRATIFIED_POOLED_N_O1.json` — secondary pooled analysis
- `PANEL_HETEROGENEITY.json` — fresh-minus-old panel effect
- `PERSISTENT_STATE_SUMMARY.json` — storage accounting
- `EVALUATION_PANEL_MANIFEST.json` — frozen panel identities and offsets
- `EVALUATION_PANEL_DISJOINTNESS_AUDIT.json` — panel provenance
- `STOP_VERIFICATION.json` — shutdown evidence
- `STORAGE_CLEANUP_MANIFEST.json` — authorized storage actions
- `gpu_artifacts/preflight/PREFLIGHT_AUDIT.json`
- `gpu_artifacts/training/O2/TRAINING_COMPLETE_O2.json`
- `gpu_artifacts/training/O2/TRAINING_LOG_O2.jsonl`
- `gpu_artifacts/evaluation/N_REAL.json`
- `gpu_artifacts/evaluation/O1_REAL.json`
- `gpu_artifacts/evaluation/O2_REAL.json`

Local checkpoint archive:

`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d8_trained_overlap_width_n_o1_o2/O2/scientific_cumulative_001200619520.pt`

## Git provenance

- Branch: `experiment-2d8-trained-overlap-width-n-o1-o2`
- Implementation commit: `1fe6761d7bfacd71be32da16c467fb081b172ef7`
- Scientific results commit: `0bded0119cab97863a6a689ec436678a7f321ea0`
- Sealed scientific tag: `experiment-2d8-trained-overlap-width-n-o1-o2-final`
- Tag target: `0bded0119cab97863a6a689ec436678a7f321ea0`
- Full-report expansion commit: the documentation-only commit containing this file
- The sealed scientific tag was not moved.

## Final conclusion

Experiment 2D8 provides a clean independent confirmation that O1 improves on N and now establishes that advantage beyond the preregistered practical margin. Widening overlap at B3 and B5 does not improve performance over O1, does not reduce persistent state, and does not support a monotonic overlap-width dose response.

**Adopt O1 among N/O1/O2. Do not launch a wider-overlap follow-up under the present evidence.**

# EXPERIMENT 2D8 COMPLETE
