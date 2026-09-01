# Experiment 2D5C — comprehensive scientific review

## Document status

This is an append-only scientific-review companion to `EXPERIMENT_2D5C_FINAL_REPORT.md`. The original 94-line report is the protocol-mandated terminal handoff and remains unchanged. This document expands the same sealed result; it does not revise the measurements, classification, checkpoint, or scientific tag.

- Experiment: `2D5C — FIXED-WRITER B3/B5 W2 REPRESENTATION PRESSURE`
- Review prepared: `2026-09-01`
- Scientific-results commit: `f6e45dcf6bca7f78b03c56b1a695e641c27d1af9`
- Sealed scientific tag: `experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final`
- Terminal postflight commit: `3240d98420a1989aca00e2514b2d3aa6195c6dbd`
- Final audit: `PASS`
- Final classification: **W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL**

## Executive summary

Experiment 2D5C asked whether mature fixed deep recurrent writers could replace almost all native historical same-layer KV at both B3 and B5. The trained C arm reduced both native local windows to W2 while expanding the existing fixed recurrent links to cover the displaced history. It then received exactly the same 191 logical batches and 100,139,008 new targets as the matched Fixed-100M control.

The result has two simultaneously strong components:

1. **The recurrent representations became more useful under pressure.** In C, turning both B3/B5 recurrence links off increased CE by `+0.001866062064`, and shuffling their aligned recurrent sequences increased CE by `+0.002060319627`. Both effects and both Fixed-to-C lifts were established above zero.
2. **The recurrence did not fully replace the removed native KV.** C finished at CE `3.095354986487`, versus `3.044323022936` for Fixed, a C penalty of `+0.051031963552` with 95% CI `[+0.050290981786, +0.051787810044]`. This is about a `1.05236×` perplexity ratio, or `+5.236%`, and is far outside the preregistered `+0.001` CE noninferiority margin.

The architecture nevertheless recovered `97.185%` of its initial geometry shock on the immutable core panel. Gates, recurrent/local contribution ratios, and gradients reaching the actual B10/B8 writers all rose during adaptation. These diagnostics agree with the causal OFF/SHUFFLED evidence that the recurrence was used. They do not erase the large absolute performance deficit.

The correct reading is therefore partial substitution: deep recurrence carried useful aligned historical information, especially at B5, but W2/W2 removed too much recent native KV for 100M targets of adaptation to close the gap.

## 1. Preregistered scientific question

The experiment tested whether already-learned fixed deep recurrent writers could replace almost all native historical same-layer KV at two destinations:

- B3, where the native window changed from W32 to W2 and `B10 → B3` became eligible for lags 2–1023.
- B5, where the native window changed from W64 to W2 and `B8 → B5` became eligible for lags 2–1023.

The central hypotheses were:

- representation pressure would increase dependence on `B10 → B3`;
- representation pressure would increase dependence on `B8 → B5`;
- after matched adaptation, C might approach Fixed in absolute true-incremental CE;
- OFF and aligned-SHUFFLED interventions would distinguish recurrent replacement from recovery mediated by other paths.

The protocol explicitly required separation of absolute model quality from recurrent-path utility. A model could learn to use recurrence more strongly and still remain worse overall; that is exactly what occurred.

## 2. Scope, lineage, and controls

Only one new arm was trained: C. No A, B, Fixed, Routed, or 250M continuation was executed.

| Role | Artifact | SHA-256 | Training in 2D5C |
|---|---|---|---:|
| C source | Accepted 2D3A parent | `de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b` | Source only |
| Matched control | Existing Fixed-100M checkpoint | `e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8` | 0 steps |
| C terminal | W2/W2 checkpoint at local update 191 | `f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe` | 191 steps |

The C arm started from the accepted 2D3A source, not from the sealed 2D4A Fixed or Routed 250M checkpoints. The Fixed checkpoint was used only as the existing matched scientific control.

The model retained exactly `124,475,908` parameters. The state-dict key set, parameter names and shapes, optimizer, scheduler, RNG, loader state, CE-only objective, recurrent writers, and gates were preserved. The architecture change was mask/index/configuration-only. Its fingerprint was:

`019d822dd89986c269e985fba8d1277a15d476dd73a0dac0d8c35e07e7315c12`

## 3. Architecture under test

| Block | C native local window | Fixed recurrent writer | C eligible recurrent lags |
|---|---:|---|---:|
| B1 | W2 | B12 → B1 | 2–1023 |
| B2 | W1024 | none | none |
| B3 | W2 | B10 → B3 | 2–1023 |
| B4 | W1024 | none | none |
| B5 | W2 | B8 → B5 | 2–1023 |
| B6 | W512 | B7 → B6 | 512–1023 |
| B7–B12 | W1024 | none | none |

For B3 and B5, local lags 0–1 and recurrent lags 2–1023 formed an exact, causal, nonoverlapping partition. No lag was missing and no lag had two routes. Relative to Fixed, only these geometry choices changed:

| Destination | Fixed local | C local | Fixed recurrence | C recurrence |
|---|---:|---:|---:|---:|
| B3 | W32 | W2 | 32–1023 | 2–1023 |
| B5 | W64 | W2 | 64–1023 | 2–1023 |

The recurrent source rings already retained the newly eligible short-lag representations. They were neither resized nor reinitialized.

## 4. Training design and execution

### 4.1 Exact matched continuation

C consumed the exact 191 logical global batches used by the existing matched Fixed arm, in the same order and with the same pass cadence.

| Quantity | Start | Finish | Change |
|---|---:|---:|---:|
| Local update | 0 | 191 | 191 |
| Global update | 1908 | 2099 | 191 |
| Cumulative targets | 1,000,341,504 | 1,100,480,512 | 100,139,008 |
| Targets per update | — | — | 524,288 |

The replay ledger contained exactly 191 batches. Its SHA-256 was `429f1d11b2af285fafab8aaf48341f6098a983b7f89598b1465974f4e969b6c0`, and the terminal replay chain was `6a5ab6dd6aee669d7e415b6ba456bf5e7e940940ab624cf1a957b103cee72d0e`.

Pass cadence was reproduced exactly: 185 updates used two passes and 6 scheduled updates used three passes. There were no skipped, overflowed, replayed, or post-terminal optimizer updates.

### 4.2 Mandatory process boundary

The protocol required a fresh-process restart at update 96. It completed successfully:

- pre-restart training process: PID `14730`;
- post-restart training process: PID `15031`;
- restart boundary: local update 96;
- sentinel maximum absolute difference: `0.0`;
- optimizer, scheduler, loader, RNG, next batch, and next pass-count decisions matched exactly.

The final checkpoint also passed a strict fresh-process reopen test.

### 4.3 Training performance

The actual optimizer run took `1,753.112` seconds, or about `29.22` minutes, at a mean `57,454.91` targets/second. The milestone rows below are training-batch measurements; the fresh evaluation panels later in this report are the scientific estimands.

| Local update | Global update | New targets | Train CE | B3 gate | B5 gate | Passes | Targets/s | Wall/update (s) | PID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1909 | 524,288 | 4.987733811 | 0.011509430 | 0.044627141 | 2 | 57,222 | 9.16 | 14730 |
| 48 | 1956 | 25,165,824 | 3.218224719 | 0.023598950 | 0.056541972 | 2 | 57,733 | 9.08 | 14730 |
| 96 | 2004 | 50,331,648 | 3.166130215 | 0.032215629 | 0.069468215 | 2 | 58,162 | 9.01 | 14730 |
| 144 | 2052 | 75,497,472 | 3.121343672 | 0.036691681 | 0.082832560 | 2 | 58,150 | 9.02 | 15031 |
| 191 | 2099 | 100,139,008 | 3.165572032 | 0.038644470 | 0.096225284 | 2 | 58,095 | 9.02 | 15031 |

Peak allocated VRAM was approximately `49.06 GiB`; peak reserved VRAM was approximately `57.37 GiB`. The larger peak came from scheduled three-pass work.

### 4.4 Why the task occupied more than one day

The greater-than-one-day elapsed task time was not greater-than-one-day model optimization. Optimization was about 29 minutes. The remaining work comprised protocol implementation and preflight, source/control/checkpoint verification, exact replay preparation, disposable smoke tests, immutable-panel freezing, the mandatory restart audit, milestone and terminal checkpoint transfers, repeated true-incremental evaluations, the 14-condition large panel, 50,000-resample paired bootstrap analysis, 224 sequence-level representation diagnostics, strict reopen and cache tests, a 1.494 GB local checkpoint backup, preservation and adjudication of two audit-tool false alarms, large-artifact packaging, Git verification, and safe pod shutdown.

This distinction matters: the scientific cost was dominated by verification and measurement breadth, while the avoidable operational problem was that monitoring did not surface those phases clearly enough and the GPU was left idle during part of the post-training workflow.

## 5. Evaluation and statistical design

### 5.1 True-incremental evaluation

The primary evaluation mode was deployment-equivalent true incremental execution. The large confirmation panel contained:

- `2,048` paired sequences;
- `1,024` targets per sequence;
- `2,097,152` targets per condition;
- `14` conditions: seven interventions for Fixed and seven for C;
- exact per-sequence pairing across every condition.

The seven conditions per architecture were ALL_REAL, B3 OFF, B3 SHUFFLED, B5 OFF, B5 SHUFFLED, B3+B5 OFF, and B3+B5 SHUFFLED.

The secondary path-matched parallel evaluations at C0, C96, and C191 passed their audits, but were not numerically mixed with or substituted for the primary true-incremental results.

### 5.2 Meaning of the interventions

- `OFF gain = CE(OFF) − CE(ALL_REAL)`. Positive values mean the live recurrent path improves CE.
- `SHUFFLED gap = CE(SHUFFLED) − CE(ALL_REAL)`. Positive values mean the correctly aligned recurrent sequence is better than a shuffled recurrent sequence.
- `Pressure lift = C intervention effect − Fixed intervention effect`. Positive values mean the W2/W2 architecture became more dependent on that recurrent mechanism.

OFF tests recurrent-path utility. SHUFFLED tests whether utility depends on aligned sequence content rather than merely path activation or distributional side effects.

### 5.3 Bootstrap and decision boundaries

All final contrasts used a paired per-sequence cluster percentile bootstrap:

- 50,000 resamples;
- seed `2026083003`;
- the same resampled sequence indices for all contrasts;
- direct per-sequence difference-in-differences for pressure lifts;
- no subtraction of independently bootstrapped summaries.

The preregistered practical noninferiority margin was `delta_CE = 0.001`. C could be called noninferior only if the upper 95% CI of `C − Fixed` was below `+0.001`. A 95% CI crossing zero was treated as not established.

## 6. Principal result: absolute quality did not recover to Fixed

| Estimand | Point estimate | Paired 95% CI | Positive sequences | Paired SE |
|---|---:|---:|---:|---:|
| Fixed ALL_REAL CE | 3.044323022936 | — | — | — |
| C ALL_REAL CE | 3.095354986487 | — | — | — |
| Fixed − C | -0.051031963552 | [-0.051787810044, -0.050290981786] | 0 / 2,048 | 0.000381703639 |
| C − Fixed penalty | +0.051031963552 | [+0.050290981786, +0.051787810044] | 2,048 / 2,048 | 0.000381703639 |

The architecture result is unusually unambiguous:

- every one of the 2,048 sequence-level C-minus-Fixed differences was positive;
- the entire CI is above zero, establishing statistical worsening;
- the entire CI is above `+0.001`, establishing material worsening beyond the practical margin;
- the upper CI is nowhere near the noninferiority boundary;
- exponentiating the CE penalty gives an approximate perplexity ratio of `1.0523565`, or a `5.236%` penalty.

C therefore did not approach Fixed closely enough to support strong substitution at W2/W2 after 100,139,008 matched targets.

## 7. Causal recurrent-mechanism results

### 7.1 Effects within each architecture

| Link | Intervention | Fixed effect | Fixed 95% CI | C effect | C 95% CI |
|---|---|---:|---:|---:|---:|
| B3 | OFF | -0.000023921401 | [-0.000079872377, +0.000031808718] | +0.000184993947 | [+0.000119602674, +0.000249774084] |
| B3 | SHUFFLED | +0.000012785384 | [-0.000041599412, +0.000067331932] | +0.000161060065 | [+0.000096326783, +0.000225035995] |
| B5 | OFF | -0.000015549722 | [-0.000076629165, +0.000044863491] | +0.001636330241 | [+0.001547201585, +0.001724891819] |
| B5 | SHUFFLED | +0.000080251648 | [+0.000023095814, +0.000137116621] | +0.001852160512 | [+0.001763843695, +0.001940769808] |
| Combined | OFF | +0.000006169170 | [-0.000056450589, +0.000068275572] | +0.001866062064 | [+0.001769525401, +0.001963154336] |
| Combined | SHUFFLED | +0.000099006173 | [+0.000040726413, +0.000157835316] | +0.002060319627 | [+0.001967473816, +0.002153794933] |

All six C effects were established above zero. The Fixed OFF effects were not established, while Fixed B5 and combined SHUFFLED gaps were small but established. The main preregistered combined results show that C both used the recurrent paths and benefited from their correct sequence alignment.

B5 accounted for most of the combined mechanism effect. That is a localization diagnostic, not proof that the B5 W2 change alone caused the absolute architecture penalty.

### 7.2 Fixed-to-C representation-pressure lifts

| Link | Intervention | Pressure lift | Paired 95% CI | Positive sequences | Paired SE |
|---|---|---:|---:|---:|---:|
| B3 | OFF | +0.000208915347 | [+0.000126389546, +0.000291684542] | 1,128 / 2,048 | 0.000042155983 |
| B3 | SHUFFLED | +0.000148274681 | [+0.000065757393, +0.000230574496] | 1,090 / 2,048 | 0.000042051416 |
| B5 | OFF | +0.001651879963 | [+0.001550919876, +0.001753040191] | 1,557 / 2,048 | 0.000051630946 |
| B5 | SHUFFLED | +0.001771908863 | [+0.001673931196, +0.001870972187] | 1,603 / 2,048 | 0.000050444571 |
| Combined | OFF | +0.001859892894 | [+0.001754336163, +0.001965946167] | 1,581 / 2,048 | 0.000053972206 |
| Combined | SHUFFLED | +0.001961313454 | [+0.001857341056, +0.002065912818] | 1,613 / 2,048 | 0.000053283682 |

Every lift was established above zero. This is the strongest causal evidence for the intended representation-pressure effect: forcing W2/W2 increased the marginal value of both recurrent links relative to the matched Fixed control.

The magnitude is highly asymmetric. B5 pressure lifts are about an order of magnitude larger than B3 lifts. That motivates decomposition or intermediate-window work, but it does not by itself identify the causal source of the `+0.0510` absolute penalty.

## 8. Adaptation trajectory

### 8.1 Longitudinal immutable core

| C local update | ALL_REAL CE | B3 OFF | B3 SHUFFLED | B5 OFF | B5 SHUFFLED | Combined OFF | Combined SHUFFLED |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.927317558128 | +0.000058250410 | +0.000314177925 | -0.007036929700 | +0.002514301469 | -0.007159265382 | +0.002155889566 |
| 48 | 3.164011790841 | +0.000279178043 | +0.000202525375 | +0.001629632151 | +0.001564639778 | +0.001773604804 | +0.001905304505 |
| 96 | 3.122225893014 | +0.000167085519 | +0.000302361402 | +0.001633121276 | +0.001400641545 | +0.001657943644 | +0.001704998004 |
| 144 | 3.106592168803 | +0.000326397663 | +0.000295952573 | +0.001751312922 | +0.001840067375 | +0.001950196041 | +0.002026195161 |
| 191 | 3.098035731623 | +0.000150473709 | +0.000287152731 | +0.001722253811 | +0.001935884672 | +0.001962010245 | +0.002234896308 |

The parent’s update-0 ALL_REAL core CE was `3.047730705509`. Applying the W2/W2 geometry without training raised C0 CE to `4.927317558128`, a severe immediate shock. Most recovery occurred by update 48. Later training produced diminishing improvement and a persistent terminal gap.

The initially negative B5 OFF value means that, immediately after the geometry change, disabling the unretrained B5 recurrence could improve CE on this core panel. By update 48 the sign had reversed and B5 recurrence had become beneficial. The positive effect then remained stable through update 191. This is direct evidence of adaptation to the new geometry rather than a static masking artifact.

### 8.2 Interaction diagnostics

The combined effect minus the sum of individual effects was:

| C update | OFF interaction | SHUFFLED interaction |
|---:|---:|---:|
| 0 | -0.000180586092 | -0.000672589828 |
| 48 | -0.000135205390 | +0.000138139353 |
| 96 | -0.000142263150 | +0.000001995057 |
| 144 | -0.000127514544 | -0.000109824787 |
| 191 | +0.000089282726 | +0.000011858906 |

The terminal interactions were near zero, so the B3 and B5 contributions were approximately additive on this diagnostic. These interaction values are descriptive and were not treated as independent architectural proof.

### 8.3 Geometry shock and recovery fraction

| Derived core statistic | Estimate | Paired 95% CI | Positive sequences |
|---|---:|---:|---:|
| Initial shock: C0 − Parent0 | +1.879586852619 | [+1.851675481918, +1.907075269979] | 256 / 256 |
| Final matched penalty: C191 − Fixed191 | +0.052901769756 | [+0.051321161220, +0.054532945118] | 256 / 256 |
| Recovery fraction | 0.971854575551 | [0.970889234861, 0.972790941077] | derived |

The system recovered roughly 97.2% of the initial degradation but not the last, scientifically important 2.8%. Because the initial shock was enormous, a high recovery fraction can coexist with a material final deficit. Recovery fraction must therefore not be read as practical equivalence.

## 9. Representation-pressure diagnostics

The representation suite evaluated seven frozen model states on the same exact 32-sequence subset, totaling 224 model-sequence diagnostics and 32,768 targets per model. All finite-statistic, frozen-state, attached-gradient, and subset-identity checks passed.

- Diagnostic subset SHA-256: `4b8c0654f4f2173bfa0becb016d564fa0086f5d831a38c560bd3c9f57f954c9f`
- Core manifest SHA-256: `8befbf790b3e522747cd39da306ec124464bf8dde1604caf64f299efa7e36216`
- Gate transformation: `tanh(raw_gate_parameter)`

These diagnostics describe how the trained representation changed. The causal claims remain grounded in OFF and SHUFFLED CE interventions.

### 9.1 Gate adaptation

| Frozen state | B3 effective gate | B5 effective gate |
|---|---:|---:|
| Parent | 0.011555109 | 0.044753078 |
| C0 | 0.011555109 | 0.044753078 |
| C48 | 0.023598950 | 0.056541972 |
| C96 | 0.032215629 | 0.069468215 |
| C144 | 0.036691681 | 0.082832560 |
| C191 | 0.038644470 | 0.096225284 |
| Fixed100M | 0.012117636 | 0.043659437 |

At C191, the B3 gate was approximately `3.19×` the Fixed gate and the B5 gate was approximately `2.20×` the Fixed gate. Relative to C0, the factors were approximately `3.34×` and `2.15×`. This monotonic gate growth is consistent with rising recurrence use under local-history pressure.

### 9.2 Contribution magnitude

The table reports the aggregate pre-output-projection ratio of gated recurrent L2 to local L2, followed by the gated recurrent L2 magnitude itself.

| Frozen state | B3 recurrent/local L2 | B5 recurrent/local L2 | B3 gated recurrent L2 | B5 gated recurrent L2 |
|---|---:|---:|---:|---:|
| Parent | 0.027813808 | 0.052866679 | 0.135977074 | 0.288631893 |
| C0 | 0.016686885 | 0.037298838 | 0.159040816 | 0.389199647 |
| C48 | 0.030063614 | 0.038328941 | 0.280627621 | 0.398626312 |
| C96 | 0.039251125 | 0.045688164 | 0.384903252 | 0.484622421 |
| C144 | 0.042097985 | 0.053263818 | 0.431793551 | 0.578645375 |
| C191 | 0.042884905 | 0.060576010 | 0.454267229 | 0.670785498 |
| Fixed100M | 0.029946835 | 0.052308106 | 0.144144178 | 0.283459923 |

The C191 recurrent/local ratio was about `43.2%` above Fixed at B3 and `15.8%` above Fixed at B5. The absolute gated recurrent contribution was much larger: approximately `3.15×` Fixed at B3 and `2.37×` Fixed at B5.

At C0, the W2 geometry greatly changed local-path magnitudes before any adaptation, so the initial ratio dropped despite a modest rise in absolute recurrent L2. The later monotonic rise is the more informative trajectory.

C191 post-output-projection recurrent/local ratios were `0.026332957` at B3 and `0.045432307` at B5. Pre-projection local/recurrent cosine means were `0.015873651` and `0.072647587`; post-projection cosine means were `0.115789860` and `0.179080070`. These low-to-moderate cosines suggest the recurrent contribution was not simply collinear with the local contribution, but they are noncausal descriptive measurements.

### 9.3 Gradient attachment to the real writers

| Frozen state | B10 writer gradient L2 for B3 | B8 writer gradient L2 for B5 |
|---|---:|---:|
| Parent | 2.431975e-05 | 6.551101e-05 |
| C0 | 7.636587e-05 | 3.434983e-04 |
| C48 | 4.306069e-05 | 9.038683e-05 |
| C96 | 5.397293e-05 | 1.035744e-04 |
| C144 | 5.822346e-05 | 1.226653e-04 |
| C191 | 6.127995e-05 | 1.389461e-04 |
| Fixed100M | 2.633782e-05 | 6.311267e-05 |

C191 gradients reaching the actual temporal-ring write edges were approximately `2.33×` Fixed at B3 and `2.20×` Fixed at B5. At C191 they reached 32,704 of 32,768 eligible B3 positions and the corresponding audited B5 writer positions. The gradient diagnostic performed no optimizer step and left the model and optimizer states invariant.

### 9.4 Lag distribution at C191

The recurrent lag bins were fixed in advance: 2–7, 8–15, 16–31, 32–63, 64–127, 128–255, 256–511, and 512–1023. In Fixed, newly exposed short bins are correctly marked unavailable rather than treated as zero.

| Lag bin | B3 source-gradient fraction | B5 source-gradient fraction | B3 attention / uniform | B5 attention / uniform |
|---:|---:|---:|---:|---:|
| 2–7 | 0.119068 | 0.257041 | 2.587× | 5.688× |
| 8–15 | 0.116072 | 0.158077 | 2.455× | 3.757× |
| 16–31 | 0.167203 | 0.169573 | 2.226× | 2.671× |
| 32–63 | 0.182251 | 0.137155 | 1.740× | 1.582× |
| 64–127 | 0.147180 | 0.103722 | 1.162× | 0.844× |
| 128–255 | 0.112089 | 0.081856 | 0.721× | 0.436× |
| 256–511 | 0.083254 | 0.056283 | 0.443× | 0.239× |
| 512–1023 | 0.072884 | 0.036293 | 0.476× | 0.417× |

B5 concentrated much more strongly on newly recurrent short history: lags 2–31 carried `58.47%` of its normalized source-gradient mass, versus `40.23%` at B3. Attention opportunity normalization shows the same qualitative pattern, especially B5’s `5.688×`-uniform emphasis on lags 2–7.

This supports the interpretation that W2 created strong pressure to recover recent context through B8→B5. It does not prove that those short recurrent reads were as effective as retaining native B5 KV; the absolute CE result says they were not.

## 10. Persistent-state accounting

All deployment-equivalent persistent tensors were measured in BF16 at canonical batch size 1. Allocator-reserved but unused memory was excluded.

| Quantity | Fixed bytes | C bytes | C reduction |
|---|---:|---:|---:|
| Logical unique BF16 payload | 33,288,192 | 33,005,568 | 282,624 |
| Measured unique storage | 33,288,192 | 33,005,568 | 282,624 |

The exact reduction was `282,624` bytes, or `276 KiB`, at batch size 1 and scales linearly with active batch size. Only the B3 and B5 local KV caches changed. Recurrent-ring shapes, capacities, logical sizes, and physical allocations were unchanged.

This experiment is therefore not an efficiency success at the measured batch size. The persistent-state saving is exact but small relative to the `+0.0510` CE penalty.

## 11. Reliability, audit, and provenance

### 11.1 Scientific checks

The terminal audit passed all required substantive checks, including:

- source and control checkpoint identity;
- exactly one newly trained arm;
- exact 191-update and 100,139,008-target limits;
- exact batch and stream replay;
- two-/three-pass cadence;
- optimizer, scheduler, RNG, and loader continuity;
- mandatory fresh-process restart;
- no training after local update 191;
- unchanged parameter count and state-dict keys;
- exact B3/B5 lag coverage and local/recurrent nonoverlap;
- causality, cache equivalence, capacity, eviction, and control-specificity tests;
- attached gradients to actual B10/B8 writers;
- all required core and 14 large-panel conditions;
- 2,097,152 targets for every large condition;
- per-sequence pairing and shared bootstrap indices;
- strict final-checkpoint reopen;
- remote/local checkpoint hash equality;
- branch and tag push verification;
- stopped pod and retained persistent volume.

### 11.2 Preserved audit-tool false alarms

Two failed legacy artifacts were preserved rather than deleted:

1. **Condition-order audit.** A checker compared mapping insertion order after JSON had been serialized with sorted keys. The semantic condition set and the separately recorded requested control order were exact. The adjudication changed only the audit interpretation; it did not change checkpoints, evaluations, per-sequence data, or statistical results.
2. **Optimizer-step audit.** A legacy checker assumed the inherited Adam state should have a singleton terminal step. The inherited optimizer actually had two valid step populations. All 152 optimizer state entries advanced by exactly 191: 149 states from 2386 to 2577 and 3 gate states from 1908 to 2099. The adjudication changed no training state and caused no replay.

The corresponding legacy failed evidence remains available in `ANALYSIS_INPUT_IDENTITY_AUDIT_LEGACY_FAILED.json` and `FINAL_CHECKPOINT_PROVENANCE_LEGACY_FAILED.json`.

### 11.3 Large artifact packaging

The full representation aggregate was `106,195,702` bytes, exceeding GitHub’s single-object 100 MB limit. The raw JSON was preserved locally and on the persistent volume. A deterministic gzip copy of `12,836,191` bytes was committed instead.

- Raw JSON SHA-256: `3a289cf0c68a86c22fc05a442949db97fdd7bd76bddb4bb33a22b679b35b8f0d`
- Gzip SHA-256: `3298ff136a96a08a933df0f0c09426bca4f786f2cb63258ee6095da518349952`

This was a repository-packaging issue, not a scientific failure. No result was reduced or recomputed to make it fit.

## 12. Final checkpoint and reproducibility identities

| Item | Value |
|---|---|
| Remote checkpoint | `/workspace/exp2d5c_w2w2_100m/checkpoints/scientific_cumulative_001100480512.pt` |
| Local checkpoint backup | `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d5c_fixed_writer_b3_b5_w2_matched_100m/scientific_cumulative_001100480512.pt` |
| Checkpoint bytes | 1,493,950,155 |
| Checkpoint SHA-256 | `f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe` |
| Model-state SHA-256 | `0275f4a3a8fbdf3b0ab572ef7ad5f1300015526086e62e785a24dc76b5f8b40d` |
| Optimizer-state SHA-256 | `a928a86247b85b4083ccb9ed165c52b25698bdf9c97c3bff78325d850b7d09d2` |
| Final next-batch SHA-256 | `62800455f294aaf110fbfc024abaa601c30f45d96175acb795a4d162d53da097` |
| Final next-stream SHA-256 | `cdfd4afb20c268d69e3e3fbc1c39076af21719ba6d2a6636180f12b1afd5a157` |
| Final local/global update | 191 / 2099 |
| Final cumulative targets | 1,100,480,512 |

The original scientific-results tag remains fixed. This expanded review is a later documentation artifact and must not move or replace that tag.

## 13. Classification logic

The preregistered partial-replacement branch required four things, all observed:

| Decision requirement | Result |
|---|---|
| C statistically or materially worse than Fixed | Yes; penalty lower CI `+0.050291` > `+0.001` margin |
| C combined OFF gain established | Yes; CI `[+0.001770, +0.001963]` |
| C combined aligned-SHUFFLED gap established | Yes; CI `[+0.001967, +0.002154]` |
| Recurrent dependence increased relative to Fixed | Yes; both combined lift CIs entirely above zero |
| Meaningful longitudinal recovery | Yes; `97.185%` of initial shock recovered |

Accordingly, the classification is:

**W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL**

This rejects two tempting but incorrect summaries:

- It is not “substitution supported,” because C is materially worse than Fixed.
- It is not “recurrence failed,” because both utility and aligned sequence dependence are strongly established and larger than in Fixed.

## 14. What is and is not established

Established:

- W2/W2 caused a large initial degradation.
- Training recovered most, but not all, of that degradation.
- C remained materially worse than Fixed after exactly 100,139,008 matched targets.
- B3 and B5 recurrent paths were causally useful in C.
- Correct sequence alignment mattered in C.
- Recurrent dependence increased under W2/W2 pressure relative to Fixed.
- The B5 signal was substantially larger than the B3 signal.
- Gates, contribution magnitude, and attached writer gradients changed in the same direction as the causal evidence.

Not established:

- Full replacement of native recent KV by fixed deep recurrence.
- Practical noninferiority to Fixed.
- Which of the B3 or B5 window changes caused the absolute architecture penalty; only A/B decomposition can localize that causally.
- Whether more than 100M additional targets would close the remaining gap.
- Whether W4, W8, or another intermediate native window gives a better quality/state tradeoff.
- That attention weights, cosines, norms, or gradients alone prove mechanism; they remain descriptive.

## 15. Limitations

- This is one matched continuation and one training seed. The paired evaluation is precise for this run but does not measure training-seed variance.
- The terminal large panel is broad within the frozen dataset protocol, but external-domain generalization was not tested.
- W2 is an extreme intervention. The result does not identify the response curve between W2 and the Fixed W32/W64 windows.
- C combines two geometry changes. The larger B5 diagnostics cannot causally assign the absolute penalty without an A or B arm.
- The recovery curve was sampled at five milestones. It shows strong early recovery and later improvement, but does not establish the asymptotic limit.
- The memory calculation covers persistent BF16 inference state, not end-to-end serving latency, bandwidth, allocator behavior, or total model memory.

## 16. Recommended next scientific step

Frozen recommendation: **DIFFERENT_NEXT_DIAGNOSTIC**.

The most informative next design is to compare decomposition against modest intermediate windows, without automatically launching anything:

- A: B3 W2 / B5 W64, isolating the B3 pressure change.
- B: B3 W32 / B5 W2, isolating the B5 pressure change.
- Intermediate alternative: W4/W8 variants to measure whether a small amount of native recent KV recovers most of the `+0.0510` penalty while retaining pressure on recurrence.

Because both destinations show established pressure lifts, A/B decomposition is the cleanest causal localization. Because the absolute penalty is large while recurrent utility is real, W4/W8 may have higher engineering value by mapping the quality–state frontier. A sensible next protocol would preregister both goals and choose one; no new arm should be inferred or started from this report alone.

Before any new experiment, its protocol must explicitly name the intended source checkpoint. Neither the C191 checkpoint nor any Fixed/Routed 250M checkpoint should be continued by assumption.

## 17. Artifact index

The complete evidence is in this result directory. The most important review entry points are:

| Artifact | Purpose |
|---|---|
| `EXPERIMENT_2D5C_FINAL_REPORT.md` | Frozen compact terminal handoff |
| `SCIENTIFIC_RESULT_SUMMARY.json` | Machine-readable results and bootstrap summaries |
| `CLASSIFICATION.json` | Decision-tree flags and frozen recommendation |
| `FINAL_AUDIT.json` | Authoritative terminal scientific/Git/operational audit |
| `FINAL_CHECKPOINT_PROVENANCE.json` | Checkpoint, optimizer, loader, and process identities |
| `TRAINING_COMPLETE_ADJUDICATED.json` | Exact training completion and optimizer progression |
| `TRAINING_LOG.jsonl` | All 191 update records |
| `DATA_REPLAY_AUDIT.json` | Exact batch/stream replay verification |
| `MIDPOINT_RESTART_AUDIT.json` | Fresh-process update-96 restart evidence |
| `TRUE_INCREMENTAL_LONGITUDINAL_CORE.json` | C0/C48/C96/C144/C191 mechanism trajectory |
| `LARGE_FINAL_BOOTSTRAP.json` | Full paired bootstrap outputs |
| `LARGE_FINAL_PER_SEQUENCE_LOSSES.json` | Per-sequence large-panel pairing data |
| `REPRESENTATION_PRESSURE_DIAGNOSTICS.json.gz` | Complete compressed diagnostic aggregate |
| `REPRESENTATION_C0.json` … `REPRESENTATION_C191.json` | Per-milestone C diagnostics |
| `REPRESENTATION_FIXED100M.json` | Matched Fixed representation reference |
| `BF16_PERSISTENT_STATE_AUDIT.json` | Exact persistent-state accounting |
| `ANALYSIS_ORDER_ADJUDICATION.json` | Condition-order false-alarm adjudication |
| `LOCAL_BACKUP_AUDIT.json` | Local checkpoint and artifact backup verification |
| `GIT_VERIFICATION.json` | Branch/tag/commit verification |

## 18. Operational closure

- RunPod pod: `grand_amber_catshark` (`7kk5yyti00rnrp`)
- Terminal pod state: `STOPPED`
- Persistent volume: `yhzyb27fb5`, retained
- Major terminal checkpoint: backed up locally with matching SHA-256
- Sealed scientific branch and tag: pushed and verified

The stale pod identity/status embedded in the earlier scientific summary predates the replacement-pod postflight. `FINAL_AUDIT.json` and the terminal postflight commit are authoritative for shutdown status.

## Bottom line

2D5C produced a clear negative result for **full** W2/W2 replacement and a clear positive result for **increased recurrent use under representation pressure**. The experiment therefore answered more than “did CE improve?”: it showed that the model can redirect meaningful aligned historical computation through the mature deep recurrent writers, especially B8→B5, yet still needs more recent native KV than W2 supplies at this adaptation budget.

**Final classification: W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL.**

**STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS.**
