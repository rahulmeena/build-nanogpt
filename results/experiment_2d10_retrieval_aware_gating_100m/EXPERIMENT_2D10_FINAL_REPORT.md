# EXPERIMENT 2D10 — RETRIEVAL-AWARE GATING 100M COMPLETE

**100M screening classification:** NEW CANDIDATE BENEFIT ESTABLISHED
**Candidates establishing benefit over D:** H
**Candidates clearing delta_CE=0.0001:** H
**Direct tanh versus softmax:** H establishes benefit over T and clears the practical margin
**Current mature baseline:** sealed 2D9 250M Dynamic.

Exactly two new training arms, each 191 updates / 100,139,008 targets; two reused sealed 100M controls; exactly four final evaluations. No continuation, window expansion, from-scratch comparison, or further training was launched.

| Condition | CE | Perplexity |
|---|---:|---:|
| S | 3.037265944699 | 20.848165206806 |
| D | 3.037199508684 | 20.846780183806 |
| T | 3.037197403652 | 20.846736300699 |
| H | 3.036814821361 | 20.838762234040 |

## Paired comparisons

Positive favors the second condition. Primary decisions use Bonferroni-adjusted **98.333333% marginal bootstrap intervals** (nominal 95% family coverage for three comparisons). Ordinary 95% intervals are descriptive.

| Contrast | Mean CE | Raw 95% CI | Adjusted 98.333333% CI | exp(contrast) | Second / first wins / ties |
|---|---:|---|---|---:|---|
| D-T | +0.000002105033 | [-0.000043940381, +0.000047987716] | [-0.000054686209, +0.000058844481] | 1.000002105035 | 2071 / 2025 / 0 |
| D-H | +0.000384687323 | [+0.000309556577, +0.000460108888] | [+0.000293674423, +0.000476805992] | 1.000384761325 | 2332 / 1764 / 0 |
| T-H | +0.000382582291 | [+0.000307313283, +0.000458700253] | [+0.000289903849, +0.000474652899] | 1.000382655484 | 2317 / 1779 / 0 |
| S-D | +0.000066436015 | [+0.000015733860, +0.000117146694] | Secondary / descriptive | 1.000066438222 | 2097 / 1999 / 0 |
| S-T | +0.000068541047 | [+0.000018231923, +0.000119191308] | Secondary / descriptive | 1.000068543396 | 2101 / 1995 / 0 |
| S-H | +0.000451123338 | [+0.000376472694, +0.000525287890] | Secondary / descriptive | 1.000451225109 | 2368 / 1728 / 0 |

| Primary contrast | Positive L>0 | Margin L>δ | Negative U<0 | Harm U<−δ | Equivalence | Second noninferior |
|---|---|---|---|---|---|---|
| D-T | False | False | False | False | True | True |
| D-H | True | False | False | True | False | True |
| T-H | True | False | False | True | False | True |

50,000 paired sequence resamples; NumPy default_rng(20260910); identical indices across all contrasts; linear percentiles. Primary percentiles: [0.833333333333, 99.166666666667]. Strict boundaries apply: touching a boundary does not pass. Absence of significance is not equivalence. S-based contrasts cannot override the primary decision. Differing significance against D does not establish a T/H difference.

T and D establish practical equivalence within ±0.0001 using the adjusted interval. H clears the practical margin over both D and T.

## Initialization, gates, and costs

T has an additive tanh gate with the inherited linear h path and retrieval-aware MLP. H uses two-branch softmax. For each projected q, completed local output, and completed recurrent output, heads are concatenated in c_proj order into a width768 vector. Those three vectors receive separate FP32 affine-free LayerNorm (epsilon 1e-5), then are concatenated into the width2304 router input. Coefficients are cast to BF16 immediately before combining original branch outputs. The shared c_proj and bias execute once.

T initially reproduced the exact parent BF16 logits and CE in both parallel and incremental checks. H intentionally scales initial eligible combined outputs by 1/(1+tanh(g0)), preserving branch ratio but changing the function. H returns exactly local output without eligible memory. Its three compatibility gate scalars and their optimizer states remain untouched.

| Disposable training-batch diagnostic | Parent CE | T CE | H CE |
|---|---:|---:|---:|
| parallel | 3.440364122 | 3.440364122 | 3.427489519 |
| incremental | 3.432337046 | 3.432337046 | 3.415507555 |

This small fixed training batch is an initialization diagnostic, not an additional validation condition or scientific comparison. Smoke state was discarded and original scientific state reloaded independently.

| T destination | Raw g0 | w norm | u mean | delta mean | FP32 g mean / std / range | BF16 g mean / std / range |
|---|---:|---:|---:|---:|---|---|
| B1 | 0.303131253 | 0.013313452 | 0.31294445 | 0.01535221 | 0.31684633 / 0.02036966 / [0.10863435, 0.39138380] | 0.31684636 / 0.02037736 / [0.10839844, 0.39062500] |
| B3 | 0.012080547 | 0.005983629 | 0.00939083 | 0.00051666 | 0.00990666 / 0.00677261 / [-0.01173111, 0.04158271] | 0.00990663 / 0.00677263 / [-0.01171875, 0.04150391] |
| B5 | 0.041995563 | 0.010035963 | 0.04078272 | 0.00013591 | 0.04088889 / 0.01260313 / [-0.00699302, 0.10271206] | 0.04088881 / 0.01260330 / [-0.00698853, 0.10253906] |

| H destination | Learned b2 | Logit difference mean | FP32 λL mean | FP32 λR mean / std / range | BF16 λR mean / std / range | Entropy mean |
|---|---|---:|---:|---|---|---:|
| B1 | [-0.010228628292679787, -1.2059063911437988] | -1.19785132 | 0.76799398 | 0.23200602 / 0.00989794 / [0.18152581, 0.27255303] | 0.23200508 / 0.00990070 / [0.18164062, 0.27343750] | 0.54141345 |
| B3 | [0.018001170828938484, -4.466044902801514] | -4.57184239 | 0.98976466 | 0.01023534 / 0.00021580 / [0.00949527, 0.01131716] | 0.01023533 / 0.00021652 / [0.00952148, 0.01129150] | 0.05707787 |
| B5 | [0.016544075682759285, -3.1314923763275146] | -3.20123215 | 0.96087149 | 0.03912851 / 0.00086380 / [0.03557832, 0.04291437] | 0.03912854 / 0.00086666 / [0.03564453, 0.04296875] | 0.16515463 |

Gate table values cover eligible-memory positions. GATE_STATISTICS.json contains separate unavailable-position summaries, full-panel means/std/extrema/negative fractions, and fixed-sample quantiles (131,072 eligible positions per destination, seed 20260911; identical positions across arms). No attention matrices or residuals were exported; detached scalar diagnostics used one device transfer per batch. Router weight norms are retained in T_REAL.json/H_REAL.json. Nonzero router weights or changing coefficients alone do not prove retrieval utility.

| Architecture | Registered parameters | Added FP32 parameter bytes versus S | Persistent state bytes per sequence |
|---|---:|---:|---:|
| S | 124,475,908 | 0 | 33,289,728 |
| D | 124,478,212 | 9,216 | 33,289,728 |
| T | 124,699,588 | 894,720 | 33,289,728 |
| H | 124,697,386 | 885,912 | 33,289,728 |

All four first full-length physical cache audits passed: persistent state delta **0**. Router activations are transient; parameters and training optimizer state are separate from historical KV/raw recurrent cache accounting.

## Training and evaluation integrity

Both new arms restored original 2D7 O1, global update 2290 / 1,200,619,520 targets. Original backbone tensors, optimizer groups/moments/individual counters, scheduler metadata, Python/NumPy/Torch CPU/CUDA RNG, and loader cursor were restored. The older source predates explicit optimizer-name mapping: its unchanged source inventory was reconstructed before adding parameters, state was transplanted by parameter name, and all states and named counters verified. New parameters used fresh state and isolated initialization generators; matching hidden-layer hashes are in SOURCE_AND_PARAMETER_AUDIT.json.

Replay used B32×T1024, accumulation16, global updates 2291–2481: 185 two-pass and six three-pass updates at 2304, 2336, 2368, 2400, 2432, 2464. Every batch, stream, cursor, pass count, and target count matched the original ledger. New counters reached 191; inherited active counters advanced 191; H retired g0 and both dormant B6 states remained identical. Final cumulative targets: 1,300,758,528. Training logs include the common weighted multipass objective separately from final incremental validation.

Focused tests: 12 CPU and 12 GPU tests passed, covering T equivalence, H closed-form/simplex/cast and empty-memory behavior, valid zero-valued memory, nonzero-router future suffix and row isolation in both modes past lag63, attached q/local/recurrent/earlier-writer gradients, expected first-zero hidden gradients, nonzero hidden gradients after output update, and activation-checkpoint gradient equivalence. Full-size GPU smoke verified batch fit and complete model/optimizer/RNG reload. Final strict reopens passed.

Fresh panel: 722cbffc8ab96c42137c174672849ccf50a0f0527055a28265391e95aaea8b2b. Seed 20260909; 64 canonical B64 batches  / 4096 sequences  / 4,194,304 targets per condition. Both historical 2D9 panels and 431 recovered historical/reserved spans were excluded. No reselection, old scores, midpoint or extra full evaluations were used. BF16 incremental execution, FP32 token CE, FP64 accumulation; four finite paired arrays and ordered identities preserved.

| Terminal identity (both new arms and historical controls) | SHA256 |
|---|---|
| Cursor | `d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d` |
| Next batch | `400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649` |
| Next stream | `0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642` |

## Runtime, backups, and shutdown

| Workload | Current minutes |
|---|---:|
| T training | 30.445 |
| H training | 30.136 |
| S evaluation | 18.404 |
| D evaluation | 18.704 |
| T evaluation | 21.022 |
| H evaluation | 21.729 |

Historical S/D training times were 28.440 / 28.987 minutes; neither was retrained here. Current T/H evaluation includes diagnostics and does not isolate production router overhead.

Supervised two-GPU interval: 77.290 wall minutes / 2.576332 aggregate GPU-hours. Pod resume through verified stop (including preparation): 95.734 wall minutes / 3.191128 aggregate GPU-hours.

Pod `nagj1hv18p3z2c` (`electrical_aqua_worm`) verified EXITED/stopped at 2026-09-05T10:36:22.029549+00:00. Persistent volume `yhzyb27fb5` retained. Both new checkpoints were exported and independently hash-verified while evaluation ran; remaining outputs were exported after GPU completion. All 19 artifact hashes match persistent copies. Statistics/reporting occurred locally after shutdown.

| Checkpoint | SHA256 |
|---|---|
| O1 | `c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6` |
| S | `676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b` |
| D | `c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44` |
| T | `7ed29fb5adc1e5aade2fad0e8db8e90233951fec5922ec13e75a7e861b2e6019` |
| H | `d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c` |

Full checkpoint paths and independent backup evidence: ALL_CHECKPOINT_IDENTITIES.json, CHECKPOINT_MANIFESTS.json, ARTIFACT_BACKUP_VERIFICATION.json, STOP_VERIFICATION.json. Large checkpoints and scalar arrays remain outside Git. Persistent runtime: `/workspace/exp2d10_retrieval_gating_100m/`.

## Interpretation and recommendation

Consider a separately authorized matched 250M continuation of H before any promotion over the mature Dynamic baseline.

This is a 100M architecture screen. The accepted 250M Dynamic checkpoint remains the mature baseline; its old-panel score is not compared here. These contrasts do not isolate retrieved-output inputs from extra query-dependent MLP capacity. T/H also differ in parameterization, signed versus convex combination, and initial function. No parameter-matched query-only MLP or inference-ablation control was included. Sequence bootstrap intervals are not training-seed replication, and contrasts share checkpoints. An unresolved 100M result does not establish permanent mechanism failure.

Scientific implementation commit: `6ad955b2f6b30cb43d5697a8a118176cd2f8010a`. Branch: `codex/experiment-2d10-retrieval-aware-gating-100m`. Final immutable tag: `experiment-2d10-retrieval-aware-gating-100m-final`. See GIT_REFERENCES.json for the final pushed commit/tag verification.
