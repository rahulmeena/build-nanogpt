# EXPERIMENT 2D9 — 250M MATCHED CONTINUATION COMPLETE

Primary classification: **TOKEN-CONDITIONED GATING ESTABLISHES UTILITY**
Preferred architecture recommendation: **Dynamic**

| Condition | CE | Perplexity |
|---|---:|---:|
| STATIC_REAL | 3.049130365956 | 21.096989782250 |
| DYNAMIC_REAL | 3.048975700608 | 21.093727061307 |
| DYNAMIC_STATICIZED | 3.049212379604 | 21.098720094297 |

| Contrast | Mean CE | 95% paired CI | exp(contrast) | Second / first wins / ties |
|---|---:|---|---:|---|
| A: Static − Dynamic | +0.000154665348 | [+0.000102589600, +0.000206530858] | 1.000154677309 | 2200 / 1896 / 0 |
| P: Staticized − Dynamic | +0.000236678996 | [+0.000183391404, +0.000290188942] | 1.000236707007 | 2251 / 1845 / 0 |
| R: Static − Staticized | -0.000082013648 | [-0.000133856788, -0.000029670076] | 0.999917989715 | 1948 / 2148 / 0 |

delta_CE = 0.0001. A measures the matched architecture benefit. P measures the inference benefit of the complete learned w term on D weights. R is descriptive; A = R + P. Positive contrasts favor the second named condition.

| Primary contrast | Positive utility | Beyond margin | Negative utility | Material harm | Equivalent | Second condition noninferior |
|---|---|---|---|---|---|---|
| A | True | True | False | False | False | True |
| P | True | True | False | False | False | True |

All flags use the original strict inequalities. An interval touching a boundary does not clear it. Architecture adoption requires A lower CI > +0.0001 and P lower CI > 0, passing integrity, and zero persistent-state growth.

## Descriptive comparison with 100M

| Contrast | 100M mean [95% CI] | 250M mean [95% CI] | Descriptive mean change |
|---|---|---|---:|
| A | +0.000054258626 [+0.000004552163, +0.000104531155] | +0.000154665348 [+0.000102589600, +0.000206530858] | +0.000100406722 |
| P | +0.000139331197 [+0.000090475812, +0.000188356938] | +0.000236678996 [+0.000183391404, +0.000290188942] | +0.000097347799 |
| R | -0.000085072571 [-0.000134928755, -0.000035385256] | -0.000082013648 [-0.000133856788, -0.000029670076] | +0.000003058923 |

The panels differ and are disjoint. These mean changes are descriptive, not a paired test of effect growth. Losses were not pooled, and absolute CEs across stages are not interpreted as a learning curve. Intervals quantify evaluation-sequence uncertainty for these continued training trajectories, not replication across training seeds. A and P share Dynamic and are not independent replications.

## Learned gates

| Destination | 100M raw g0 | 250M raw g0 | Change | 100M ‖w‖₂ | 250M ‖w‖₂ | Change |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.308653027 | 0.314035445 | +0.005382419 | 0.014254109 | 0.019666972 | +0.005412863 |
| B3 | 0.012151183 | 0.012935475 | +0.000784292 | 0.005996462 | 0.007802825 | +0.001806363 |
| B5 | 0.042001247 | 0.041949447 | -0.000051800 | 0.010406611 | 0.016615810 | +0.006209199 |

| Destination | Final tanh(g0) | FP32 gate mean / std / range | BF16 coefficient mean / std / range |
|---|---:|---|---|
| B1 | 0.304103792 | 0.324220275 / 0.023843344 / [0.101350389, 0.417327195] | 0.324219878 / 0.023849328 / [0.101562500, 0.417968750] |
| B3 | 0.012934753 | 0.010843919 / 0.007281515 / [-0.015596947, 0.048286434] | 0.010843901 / 0.007281539 / [-0.015625000, 0.048339844] |
| B5 | 0.041924857 | 0.041582144 / 0.017563824 / [-0.034001339, 0.124682561] | 0.041581977 / 0.017563890 / [-0.033935547, 0.124511719] |

B1 gates use token embedding plus position embedding; B3/B5 additionally contain preceding contextual processing. Gates use the current pre-attention residual with FP32 RMS epsilon 1e-5, dot product and tanh, then cast the coefficient to the attention dtype. The attention and gating kernels are unchanged from the sealed 100M experiment.

All statistics use the 4,194,304 positions per destination. Quantiles are exact full-panel linear percentiles, with no sampling. Across-token standard deviation, standard deviation of per-sequence means, and mean within-sequence standard deviation all use population definitions (ddof=0). Intrinsic gates are also reported at positions without eligible recurrent memory, where they scale zero recurrent output. B1 first has eligible memory at position 1, B3 at 31, and B5 at 63.

FP32 preactivation summaries (all panel positions):

| Destination | Mean | Std | Minimum | Maximum |
|---|---:|---:|---:|---:|
| B1 | 0.336578150 | 0.026447865 | 0.101699561 | 0.444451123 |
| B3 | 0.010845007 | 0.007283189 | -0.015598211 | 0.048324015 |
| B5 | 0.041619497 | 0.017604399 | -0.034014452 | 0.125334740 |

Exact gate percentiles:

| Destination / precision | p1 | p5 | p25 | Median | p75 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 / FP32 | 0.261261794 | 0.280186836 | 0.311452419 | 0.326597542 | 0.341792829 | 0.355203211 | 0.367298722 |
| B1 / BF16 | 0.261718750 | 0.279296875 | 0.310546875 | 0.326171875 | 0.341796875 | 0.355468750 | 0.367187500 |
| B3 / FP32 | -0.002773438 | 0.000926300 | 0.005617296 | 0.009728128 | 0.014984026 | 0.024824465 | 0.030669060 |
| B3 / BF16 | -0.002777100 | 0.000926971 | 0.005615234 | 0.009704590 | 0.014953613 | 0.024780273 | 0.030639648 |
| B5 / FP32 | 0.001098582 | 0.013059271 | 0.030375902 | 0.041030513 | 0.052355439 | 0.070737441 | 0.088094767 |
| B5 / BF16 | 0.001098633 | 0.013061523 | 0.030395508 | 0.041015625 | 0.052246094 | 0.070800781 | 0.087890625 |

Gate variation and sign summaries:

| Destination / precision | Negative fraction | Mean abs. deviation from tanh(g0) | Std of sequence means | Mean within-sequence std |
|---|---:|---:|---:|---:|
| B1 / FP32 | 0.000000000 | 0.026589179 | 0.003751378 | 0.023493849 |
| B1 / BF16 | 0.000000000 | 0.026593897 | 0.003751002 | 0.023500000 |
| B3 / FP32 | 0.034505844 | 0.006264726 | 0.001174635 | 0.007160191 |
| B3 / BF16 | 0.034505844 | 0.006264721 | 0.001174645 | 0.007160213 |
| B5 / FP32 | 0.008235693 | 0.013632589 | 0.002999686 | 0.017279402 |
| B5 / BF16 | 0.008235693 | 0.013632706 | 0.002999610 | 0.017279483 |

Position-bin summaries (0, 1–31, 32–63, 64–127, 128–255, 256–511, 512–767, 768–1023) and eligible-memory fractions are preserved in GATE_STATISTICS.json. First eligible positions are B1: 1, B3: 31, B5: 63.

Gate scalars were collected during DYNAMIC_REAL with at most one batch on-device and one CPU transfer per batch. No attention matrices or extra evaluation condition were collected. Nonzero w and gate variance alone are not evidence of usefulness.

Staticization removes the entire learned w term, including its average shift and token variation. A positive P establishes that term’s inference utility; it does not establish superiority to every optimally refitted constant gate.

## Matched continuation and provenance

**286 additional updates / 149,946,368 additional targets per arm; 477 updates / 250,085,376 total 2D9 targets per arm.**

Each arm resumed its own sealed 100M checkpoint. S and D model/optimizer tensors were not required to equal one another. Both sources restored their exact model, optimizer groups and moments, individual Adam counters, scheduler metadata, RNG states, and shared loader cursor. D’s learned w vectors and their moments were preserved.

Continuation global updates: 2482–2767 inclusive. Source inherited target counter: 1,300,758,528. Final inherited target counter: 1,450,704,896. B32×T1024, accumulation16 and 524,288 targets/update were unchanged. The stage used 277 two-pass updates and 9 three-pass updates, at 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720 and 2752, with inherited loss weights.

Every active parameter’s Adam counter advanced 286 from its own source value; D’s w counters reached 477. Dormant B6 parameters and optimizer state were unchanged. Base and w LR remained 3e-5, scalar-gate LR 3e-4, and w weight decay 0. Warmup was not restarted.

Committed continuation implementation: `38e6edce5a0391ea0e14f373fe9714bd232ebd49`
Sealed 100M result commit: `482ad55637c2a0adb5c7c268b37c7be243ac15c8`

Fresh panel SHA: `d1caaf166ebae8bef5729704fc8077af138558e91c10cc7b6fd0ef1e7b2bc3b5`
Panel selection seed: 20260907. Exactly 64 canonical B64 validation batches, 4096 sequences and 4,194,304 targets per condition. All historical exclusions and the sealed 100M panel were applied before continuation training or scoring.

Exactly three final conditions used true incremental inference, BF16 execution, FP32 token CE and FP64 accumulation. All model/cache state resets between sequence batches and conditions. Staticized generated its entire trajectory with the w term omitted, using D’s final g0 and base weights; it did not reuse Dynamic-Real caches or mutate/refit the checkpoint.

CPU analysis used 50,000 paired sequence-level bootstrap resamples, RNG seed 20260906, shared indices across A/P/R, and 95% NumPy linear-percentile intervals. The original ordered classification and adoption rules were applied unchanged.

## Memory, runtime, checkpoints and shutdown

**Persistent state: 33,289,728 BF16 bytes per B=1 sequence for all conditions; delta 0.** Dynamic still adds only 2,304 parameters: 9,216 FP32 bytes, or 4,608 bytes if stored in BF16. Optimizer state is training-only; gate arrays are analysis artifacts.

| Workload | Minutes |
|---|---:|
| S | 41.64 |
| D | 42.44 |
| STATIC_REAL | 18.44 |
| DYNAMIC_REAL | 19.59 |
| DYNAMIC_STATICIZED | 18.42 |

Timings describe these workloads. Dynamic Real includes gate collection; training update times include state audits and exclude recovery-checkpoint writing. No isolated benchmark was run.

One complete recovery checkpoint was written at continuation update 144 for each arm. Recovery saves preserve scientific RNG state. Final checkpoints passed strict reopen and independent local/persistent SHA verification. All historical checkpoints, results, manifests and sealed tags were retained.

| Arm | Source 100M SHA-256 | Final 250M SHA-256 |
|---|---|---|
| S | `676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b` | `265ad44e39b513d939c43ea6b71fdac1f6a6c65df04f7dba9fd724df8f7a0ddb` |
| D | `c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44` | `9714b2e3f53a8c15dfecfed3e9b56c358176c1f9f609bcce7e28c35b8a358a9b` |

S local final checkpoint: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/S/scientific_cumulative_001450704896.pt`
S retained persistent final checkpoint: `/workspace/exp2d9_dynamic_gating_250m/run/checkpoints/S/scientific_cumulative_001450704896.pt`

D local final checkpoint: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/D/scientific_cumulative_001450704896.pt`
D retained persistent final checkpoint: `/workspace/exp2d9_dynamic_gating_250m/run/checkpoints/D/scientific_cumulative_001450704896.pt`

Terminal loader cursor SHA: `f549cf0a45863e65391147d4439c1ae0af2ba1c8bc1d943e1a33f532c7b3d1d0`
Next global batch SHA: `c1d217a6af6379263b035c47a158bdeada49f9611cb2fa721ae6c939aa35fe27`
Next stream SHA: `c9025e28b5ef35a00c35fde4e89d38da8bfdae50235211d88516f14abebb4196`
Terminal equality across S, D and the frozen continuation: **PASS**.

Final audit: **PASS, 47/47 checks**.
GPU pod `grand_amber_catshark` (`7kk5yyti00rnrp`): desired status `EXITED`, runtime status `stopped`. Stop verification: **PASS**. Persistent volume `yhzyb27fb5` retained.

## Decision and one next recommendation

Matched architecture benefit: statistically established; benefit beyond the practical margin: established.
The learned w term’s inference utility: statistically established.
Prefer Dynamic among the tested architectures: the existing adoption rule is met. Next recommendation: use the sealed 250M Dynamic checkpoint as the candidate baseline for the next separately scoped experiment.

No automatic 500M extension or further experiment was launched. This continuation is complete and stopped.
