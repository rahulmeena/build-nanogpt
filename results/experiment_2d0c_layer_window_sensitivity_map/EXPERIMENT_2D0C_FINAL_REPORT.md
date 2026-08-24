# Experiment 2D0C — Full Layer × KV-Window Sensitivity Map

## Outcome

Descriptive classification: **MARGINAL KV SENSITIVITY IS IRREGULAR / MULTIMODAL**.

This is a single-layer marginal sensitivity map under an otherwise full-context Transformer. It is not a jointly shortened model and does not identify optimal simultaneous layer windows; interactions between shortened layers can be nonlinear.

## Validation-damage matrix

| Layer | W1024 | W512 | W256 | W128 | W64 | W32 | W16 | W8 | W4 | W2 | W1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | +0.000000 | +0.154916 | +0.557936 | +0.835810 | +0.990025 | +1.107044 | +1.227068 | +1.406887 | +1.664340 | +1.973811 | +2.219531 |
| B2 | +0.000000 | +0.012409 | +0.013257 | +0.013933 | +0.014583 | +0.017916 | +0.033303 | +0.105422 | +0.406299 | +1.511357 | +3.660851 |
| B3 | +0.000000 | +0.000784 | +0.001370 | +0.004332 | +0.017375 | +0.059219 | +0.152362 | +0.320024 | +0.734506 | +1.938957 | +4.322851 |
| B4 | +0.000000 | +0.001106 | +0.004520 | +0.018137 | +0.068326 | +0.136097 | +0.203690 | +0.285144 | +0.419752 | +0.822532 | +3.004057 |
| B5 | +0.000000 | +0.007683 | +0.009069 | +0.011339 | +0.015884 | +0.026270 | +0.052591 | +0.114252 | +0.288307 | +0.791766 | +1.794224 |
| B6 | +0.000000 | +0.001922 | +0.002448 | +0.004450 | +0.008592 | +0.016794 | +0.033359 | +0.071392 | +0.167272 | +0.428051 | +1.555582 |
| B7 | +0.000000 | +0.001356 | +0.005069 | +0.012263 | +0.024083 | +0.044075 | +0.074105 | +0.118367 | +0.189161 | +0.313261 | +0.746987 |
| B8 | +0.000000 | +0.019099 | +0.047010 | +0.077526 | +0.107063 | +0.138071 | +0.173312 | +0.221112 | +0.289415 | +0.386750 | +0.545039 |
| B9 | +0.000000 | +0.003022 | +0.013220 | +0.034367 | +0.073374 | +0.141434 | +0.244510 | +0.384396 | +0.559063 | +0.777767 | +1.112572 |
| B10 | +0.000000 | +0.002003 | +0.006212 | +0.013485 | +0.026158 | +0.046142 | +0.074035 | +0.109607 | +0.152544 | +0.211582 | +0.302859 |
| B11 | +0.000000 | +0.002309 | +0.007686 | +0.018432 | +0.037893 | +0.069520 | +0.109555 | +0.148999 | +0.182935 | +0.218982 | +0.279037 |
| B12 | +0.000000 | +0.001364 | +0.004399 | +0.010361 | +0.021058 | +0.037838 | +0.056747 | +0.073061 | +0.085135 | +0.096979 | +0.121337 |

## Marginal width-at-damage profiles

| Layer | W@0.001 | W@0.0025 | W@0.005 | W@0.01 | W@0.02 | W@0.05 | W@0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| B2 | 1024 | 1024 | 1024 | 1024 | 32 | 16 | 16 |
| B3 | 512 | 256 | 128 | 128 | 64 | 64 | 32 |
| B4 | 1024 | 512 | 256 | 256 | 128 | 128 | 64 |
| B5 | 1024 | 1024 | 1024 | 256 | 64 | 32 | 16 |
| B6 | 1024 | 256 | 128 | 64 | 32 | 16 | 8 |
| B7 | 1024 | 512 | 512 | 256 | 128 | 32 | 16 |
| B8 | 1024 | 1024 | 1024 | 1024 | 512 | 256 | 128 |
| B9 | 1024 | 1024 | 512 | 512 | 256 | 128 | 64 |
| B10 | 1024 | 512 | 512 | 256 | 128 | 32 | 16 |
| B11 | 1024 | 512 | 512 | 256 | 128 | 64 | 32 |
| B12 | 1024 | 512 | 256 | 256 | 128 | 32 | 2 |

The W@0.01 row is the primary *marginal W@0.01 profile*, not a set of optimal joint windows.

## Fixed-window depth summary

| Layer | damage@512 | damage@128 | damage@32 | damage@1 | argmax@1 |
|---|---:|---:|---:|---:|---:|
| B1 | +0.154916 | +0.835810 | +1.107044 | +2.219531 | 0.318172 |
| B2 | +0.012409 | +0.013933 | +0.017916 | +3.660851 | 0.128532 |
| B3 | +0.000784 | +0.004332 | +0.059219 | +4.322851 | 0.081500 |
| B4 | +0.001106 | +0.018137 | +0.136097 | +3.004057 | 0.215421 |
| B5 | +0.007683 | +0.011339 | +0.026270 | +1.794224 | 0.362906 |
| B6 | +0.001922 | +0.004450 | +0.016794 | +1.555582 | 0.417879 |
| B7 | +0.001356 | +0.012263 | +0.044075 | +0.746987 | 0.567913 |
| B8 | +0.019099 | +0.077526 | +0.138071 | +0.545039 | 0.630264 |
| B9 | +0.003022 | +0.034367 | +0.141434 | +1.112572 | 0.506151 |
| B10 | +0.002003 | +0.013485 | +0.046142 | +0.302859 | 0.728009 |
| B11 | +0.002309 | +0.018432 | +0.069520 | +0.279037 | 0.756389 |
| B12 | +0.001364 | +0.010361 | +0.037838 | +0.121337 | 0.827973 |

## Hypothetical marginal KV budgets — not joint-model results

| Damage threshold | Sum independently selected W | Fraction of 12×1024 | Min W | Max W |
|---:|---:|---:|---:|---:|
| 0.0010 | 11776 | 0.958333 | 512 | 1024 |
| 0.0025 | 8192 | 0.666667 | 256 | 1024 |
| 0.0050 | 6912 | 0.562500 | 128 | 1024 |
| 0.0100 | 5312 | 0.432292 | 64 | 1024 |
| 0.0200 | 2624 | 0.213542 | 32 | 1024 |
| 0.0500 | 1824 | 0.148438 | 16 | 1024 |
| 0.1000 | 1418 | 0.115397 | 2 | 1024 |

## Scientific questions

1. At W1 the most sensitive block is B3 (+4.322851); fixed-window rankings for W512/W128/W32/W1 are saved in `fixed_window_rankings.json`.
2. At W1 the least sensitive block is B12 (+0.121337).
3. Marginal W@0.01: B1=1024, B2=1024, B3=128, B4=256, B5=256, B6=64, B7=256, B8=1024, B9=512, B10=256, B11=256, B12=256.
4. Depth-vs-W@0.01 Spearman correlation is -0.176834; the classification above states whether widening with depth is supported.
5. The same correlation and adjacent-step counts distinguish decreasing sensitivity; see `shape_analysis.json`.
6. Middle-layer bulge evidence was assessed across W@0.005/.01/.02/.05 rather than from one threshold; the resulting descriptive category is shown above.
7. B11 ranks 11/12 in W1 damage, so its earlier curve is interpreted relative to the complete network rather than in isolation.
8. B12 minus B11 W1 damage is -0.157700; B12 follows eleven full-context blocks, while B11 follows ten.
9. B1 W1 damage is +2.219531; unlike later blocks, it has no contextualized lower-layer Transformer residual input.
10. The largest W128→W32 damage increases occur at B1, B4, B9, B8.
11. There are 0 monotonic violations; the largest is 0.0000000000.
12. In 100.0% of comparable layer/window bin cases, mean absolute damage after the removal boundary is at least as large as before it.
13. Independently selecting W@0.01 sums to 5312 (0.432292 of 12×1024). This is hypothetical because joint damage is not additive and was not measured.
14. The triangle hypothesis assessment is: MARGINAL KV SENSITIVITY IS IRREGULAR / MULTIMODAL. It is descriptive marginal evidence, not a final architecture decision.
15. The next controlled comparison should use matched total KV budgets to compare the empirical marginal profile, a monotonic triangle, and a uniform sliding-window baseline; it is not executed here.

## Interpretation

For every layer above B1, a low marginal sensitivity can reflect information already contextualized by lower full-context layers. It does not imply that the same layer will remain insensitive when lower layers are shortened simultaneously.

B1 and B12 are special endpoints: B1 cannot receive contextualized Transformer state from below, while B12 tests the final block's own historical attention after eleven full-context blocks.

## Integrity audit

- PASS — 2D0B frozen tag exact
- PASS — ~10B Standard checkpoint SHA exact
- PASS — Standard GPT-2 architecture exact
- PASS — Full AttnRes absent
- PASS — canonical validation manifest exact
- PASS — baseline full-context regression exact
- PASS — B11 W512 regression
- PASS — B11 W128 regression
- PASS — B11 W1 regression
- PASS — human/zero-based layer mapping exact
- PASS — one and only one shortened layer per cell
- PASS — all other 11 layers W1024
- PASS — window grid exact
- PASS — absolute positions and causal semantics exact
- PASS — same BF16 runtime and loss denominator
- PASS — same validation batches every cell
- PASS — all losses and predictions finite
- PASS — model tensor hashes before/after identical
- PASS — optimizer/scheduler/GradScaler/backward/steps/updates/training targets zero
- PASS — no recurrence/completion/writers/temporal or Full AttnRes/BPTT/HellaSwag
- PASS — all 120 shortened cells completed
- PASS — all four GPU workers exited successfully
- PASS — cross-artifact matrix consistency
- PASS — required artifacts present
- PASS — four worker logs present

## Performance

Total four-GPU elapsed wall time: 476.661 seconds.
Mean scientific cell time: 14.992 seconds.
Scientific validation targets: 157286400.
Mean targets/sec across cells: 87583.8.

## Next controlled experiment

**PROCEED TO MATCHED JOINT-KV GEOMETRY EVALUATION**

The recommendation was not executed.

# EXPERIMENT 2D0C COMPLETE
