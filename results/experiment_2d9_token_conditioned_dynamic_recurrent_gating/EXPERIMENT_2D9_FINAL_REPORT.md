# EXPERIMENT 2D9 — TOKEN-CONDITIONED DYNAMIC RECURRENT GATING COMPLETE

Primary classification: **TOKEN-CONDITIONED GATING ESTABLISHES UTILITY**
Preferred architecture recommendation: **Retain static O1 provisionally**

| Condition | CE | Perplexity |
|---|---:|---:|
| STATIC_REAL | 3.047944432829 | 21.071984993122 |
| DYNAMIC_REAL | 3.047890174203 | 21.070841687183 |
| DYNAMIC_STATICIZED | 3.048029505400 | 21.073777717319 |

| Contrast | Mean CE | 95% paired CI | exp(contrast) | Second / first wins / ties |
|---|---:|---|---:|---|
| A: Static − Dynamic | +0.000054258626 | [+0.000004552163, +0.000104531155] | 1.000054260098 | 2121 / 1975 / 0 |
| P: Staticized − Dynamic | +0.000139331197 | [+0.000090475812, +0.000188356938] | 1.000139340904 | 2171 / 1925 / 0 |
| R: Static − Staticized | -0.000085072571 | [-0.000134928755, -0.000035385256] | 0.999914931047 | 1955 / 2141 / 0 |

delta_CE = 0.0001. Positive A favors Dynamic over matched Static; positive P favors the learned w term on D weights. R is descriptive; A = R + P.

| Primary contrast | Positive utility | Beyond margin | Negative utility | Material harm | Equivalent | Second condition noninferior |
|---|---|---|---|---|---|---|
| A | True | False | False | False | False | True |
| P | True | False | False | False | False | True |

All flags use strict inequalities. Statistical utility and practical equivalence can coexist. Unresolved significance is not equivalence.

## Learned gates

| Destination | Parent raw g0 | Final raw g0 | tanh(g0) | ‖w‖₂ | FP32 gate mean / std / range | BF16 coefficient mean / std / range |
|---|---:|---:|---:|---:|---|---|
| B1 | 0.305539072 | 0.308653027 | 0.299211174 | 0.014254109 | 0.310348922 / 0.017647441 / [0.139317036, 0.410523266] | 0.310347060 / 0.017655327 / [0.139648438, 0.410156250] |
| B3 | 0.011701962 | 0.012151183 | 0.012150585 | 0.005996462 | 0.009693904 / 0.006634473 / [-0.012053467, 0.040354349] | 0.009693868 / 0.006634470 / [-0.012023926, 0.040283203] |
| B5 | 0.044410013 | 0.042001247 | 0.041976567 | 0.010406611 | 0.039669858 / 0.012164474 / [-0.009126551, 0.093307376] | 0.039669717 / 0.012164665 / [-0.009155273, 0.093261719] |

B1 gates see token embedding plus position embedding. B3 and B5 gates also see preceding contextual processing. The FP32 RMS epsilon is 1e-5, with no sqrt(768) scaling. The coefficient is cast to the attention dtype immediately before multiplying the recurrent output. Local and recurrent softmaxes remain separate and the destination projection, including its bias, is applied once.

Gate summaries use all 4,194,304 positions per destination, including positions without eligible memory. An intrinsic gate at those positions has no recurrent output to scale. Percentiles are exact full-panel linear percentiles. Global token std, std of sequence means, and mean within-sequence std are population measures (ddof=0). Full moments, percentiles, negative fractions, deviation from tanh(g0), and position bins for both precisions are in GATE_STATISTICS.json. Nonzero vectors or gate variance alone are not evidence of utility.

FP32 preactivation summaries (all panel positions):

| Destination | Mean | Std | Minimum | Maximum |
|---|---:|---:|---:|---:|
| B1 | 0.321047714 | 0.019462241 | 0.140229017 | 0.436240375 |
| B3 | 0.009694683 | 0.006635599 | -0.012054051 | 0.040376276 |
| B5 | 0.039696824 | 0.012188134 | -0.009126805 | 0.093579583 |

Exact gate percentiles:

| Destination / precision | p1 | p5 | p25 | Median | p75 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 / FP32 | 0.266149375 | 0.282186122 | 0.299123466 | 0.311770827 | 0.322498918 | 0.336483300 | 0.349225968 |
| B1 / BF16 | 0.265625000 | 0.281250000 | 0.298828125 | 0.312500000 | 0.322265625 | 0.335937500 | 0.349609375 |
| B3 / FP32 | -0.002500003 | 0.000132958 | 0.004685912 | 0.009140172 | 0.013673285 | 0.022065512 | 0.026843549 |
| B3 / BF16 | -0.002502441 | 0.000132561 | 0.004699707 | 0.009155273 | 0.013671875 | 0.022094727 | 0.026855469 |
| B5 / FP32 | 0.013888264 | 0.021076406 | 0.031297036 | 0.038836807 | 0.047287011 | 0.060541044 | 0.072769916 |
| B5 / BF16 | 0.013916016 | 0.021118164 | 0.031250000 | 0.038818359 | 0.047363281 | 0.060546875 | 0.072753906 |

Gate variation and sign summaries:

| Destination / precision | Negative fraction | Mean abs. deviation from tanh(g0) | Std of sequence means | Mean within-sequence std |
|---|---:|---:|---:|---:|
| B1 / FP32 | 0.000000000 | 0.016501692 | 0.002406183 | 0.017460111 |
| B1 / BF16 | 0.000000000 | 0.016509187 | 0.002406004 | 0.017468131 |
| B3 / FP32 | 0.046692610 | 0.005841983 | 0.000894861 | 0.006560232 |
| B3 / BF16 | 0.046692610 | 0.005841961 | 0.000894862 | 0.006560228 |
| B5 / FP32 | 0.000032425 | 0.009895534 | 0.001669768 | 0.012032889 |
| B5 / BF16 | 0.000032425 | 0.009895535 | 0.001669775 | 0.012033079 |

Position-bin summaries (0, 1–31, 32–63, 64–127, 128–255, 256–511, 512–767, 768–1023) and eligible-memory fractions are preserved in GATE_STATISTICS.json. First eligible positions are B1: 1, B3: 31, B5: 63.

## Training, evaluation and provenance

Exactly **191 updates / 100,139,008 new targets per arm**, independently from the accepted O1 parent. Updates 2291–2481; final cumulative targets 1,300,758,528. B32×T1024, 16 microbatches/update; 185 two-pass and 6 three-pass updates. Three-pass updates: 2304, 2336, 2368, 2400, 2432, 2464. Old parameter values, moments, individual Adam counters, constant LR metadata, all RNG states and the loader were restored. The new vectors use a separate group with the inherited base no-decay settings and fresh moments. Dormant g_rec_b6 and its old optimizer state remained unchanged.

Source SHA: `c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6`
Committed scientific implementation: `12d9ebbc0e86584b6f35088881865e3a8a05a798`
Fresh panel SHA: `aa27fff24ddd5776d357f81b3197a8e44bccbb3ac30ca6ee7e93bc9c3822531f`
Panel seed: 20260905; 64 canonical B64 batches, 4096 sequences, 4,194,304 targets per condition. Recovered historical target spans and the reserved prefix were excluded before training/scoring. Exactly three final evaluations used BF16 execution, FP32 token CE, and FP64 accumulation.

Staticized resets all caches and generates its entire trajectory with the w term omitted. D-trained g0 and base weights are retained. The control does not mutate D tensors or its checkpoint.

50,000 paired sequence bootstrap resamples use NumPy RNG seed 20260906, shared indices for A/P/R, and 95% linear percentile intervals. These intervals quantify evaluation-sequence uncertainty for two sealed training trajectories. A and P share Dynamic and are not independent replications; this is not a training-seed study.

Staticization removes the complete learned w term, including an average shift as well as token variation. A positive P establishes the term’s inference utility; it does not show token variation beats every optimally refitted constant gate.

## State, runtime and artifacts

Persistent inference state is **33,289,728 BF16 bytes per B=1 sequence in all conditions; delta 0**. Dynamic adds 2,304 parameters (9,216 FP32 bytes; 4,608 bytes if stored in BF16). Adam state is training-only. Gate diagnostic arrays are analysis artifacts, not persistent model state.

| Workload | Wall time (minutes) |
|---|---:|
| S | 28.44 |
| D | 28.99 |
| STATIC_REAL | 18.33 |
| DYNAMIC_REAL | 21.30 |
| DYNAMIC_STATICIZED | 18.54 |

These are descriptive workload timings, not an isolated gate timing benchmark. Dynamic Real includes gate collection; training includes state audits. The optional timing campaign was omitted.

S checkpoint SHA: `676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b`
Local checkpoint: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating/S/scientific_cumulative_001300758528.pt`
Retained persistent checkpoint: `/workspace/exp2d9_dynamic_gating/run/checkpoints/S/scientific_cumulative_001300758528.pt`

D checkpoint SHA: `c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44`
Local checkpoint: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating/D/scientific_cumulative_001300758528.pt`
Retained persistent checkpoint: `/workspace/exp2d9_dynamic_gating/run/checkpoints/D/scientific_cumulative_001300758528.pt`

Terminal loader cursor SHA: `d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d`
Next global batch SHA: `400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649`
Next stream SHA: `0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642`
Terminal identities match S versus D and the frozen continuation.

Final audit: **PASS, 44/44 checks**. Final checkpoints passed strict reopen and independent local/persistent SHA verification before GPU shutdown.
GPU pod `lazy_tan_louse` (`rx11t3e4lvfuhf`) is verified **stopped**, with desired status EXITED and runtime status stopped. Persistent volume `yhzyb27fb5` is retained.

## Decision and next recommendation

Matched architecture benefit: established statistically; beyond-margin benefit: not established.
Learned w-term inference benefit: established statistically.
Retain static O1 provisionally because the matched architecture benefit does not clear the practical margin. Next recommendation: consider a separately authorized matched 250M continuation.

No follow-up launched. No extra seeds, panels, conditions, overlap-width sweeps or diagnostic campaigns were run. Checkpoints, results, panel manifests and the retained volume must not be deleted without explicit authorization.
