# Experiment 2D1C — Frozen C1908 Residual-Fusion Alpha Sweep

## Final classifications

**Primary:** RESIDUAL FUSION REDUCES RECURRENT HARM BUT DOES NOT YET CREATE UTILITY

**Stability:** WIDE BOUNDED RESIDUAL-ALPHA RANGE

The diagnostic used exactly `X = E + alpha*F` on frozen C1908 weights with D12 windows. Optimizer objects, backward calls, parameter updates, training targets, and training-loader advances were all zero.

## T1 — Alpha sweep

| alpha | Plain | Real | Shuffled | Zero | Gain | Sequence gap | Zero dependency | Real>plain wins |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.1001356740 | 3.1001356740 | 3.1001356740 | 3.1001356740 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0 |
| 0.03125 | 3.1001356740 | 3.1262396877 | 3.1339612075 | 3.1001356740 | -0.0261040137 | 0.0077215198 | -0.0261040137 | 0 |
| 0.0625 | 3.1001356740 | 3.1681423450 | 3.1804268041 | 3.1001356740 | -0.0680066709 | 0.0122844592 | -0.0680066709 | 0 |
| rms_match | 3.1001356740 | 3.2065093706 | 3.2197222055 | 3.1001356740 | -0.1063736965 | 0.0132128349 | -0.1063736965 | 0 |
| 0.125 | 3.1001356740 | 3.3201213253 | 3.3334590633 | 3.1001356740 | -0.2199856512 | 0.0133377381 | -0.2199856512 | 0 |
| 0.25 | 3.1001356740 | 3.6367037395 | 3.6712014262 | 3.1001356740 | -0.5365680654 | 0.0344976867 | -0.5365680654 | 0 |
| 0.5 | 3.1001356740 | 3.8502073551 | 3.9761526798 | 3.1001356740 | -0.7500716811 | 0.1259453247 | -0.7500716811 | 0 |
| 0.75 | 3.1001356740 | 3.7593763282 | 3.9039910336 | 3.1001356740 | -0.6592406541 | 0.1446147055 | -0.6592406541 | 0 |
| 1 | 3.1001356740 | 3.5378252564 | 3.7129882658 | 3.1001356740 | -0.4376895824 | 0.1751630094 | -0.4376895824 | 0 |

## T2 — Scale and stability

| alpha | F/E | alphaF/E | X/E | Max RMS | Pass-32 RMS | Class |
|---|---|---|---|---|---|---|
| 0 | 12.079115 | 0.000000 | 1.000000 | 0.035730 | 0.0356732327 | STABLE / STATIONARY |
| 0.03125 | 12.079115 | 0.377472 | 1.087170 | 0.038879 | 0.0388105791 | BOUNDED OSCILLATORY |
| 0.0625 | 12.079115 | 0.754945 | 1.284075 | 0.046044 | 0.0459507629 | BOUNDED OSCILLATORY |
| rms_match | 12.079115 | 0.999879 | 1.450609 | 0.052137 | 0.0520322174 | BOUNDED OSCILLATORY |
| 0.125 | 12.079115 | 1.509889 | 1.854072 | 0.067151 | 0.0670187473 | BOUNDED OSCILLATORY |
| 0.25 | 12.079115 | 3.019779 | 3.230278 | 0.119851 | 0.1195955463 | STABLE / STATIONARY |
| 0.5 | 12.079115 | 6.039557 | 6.173129 | 0.230158 | 0.2296956107 | STABLE / STATIONARY |
| 0.75 | 12.079115 | 9.059257 | 9.166080 | 0.342191 | 0.3414183557 | BOUNDED OSCILLATORY |
| 1 | 12.079115 | 12.079115 | 12.172409 | 0.455488 | 0.4543445557 | EXPANSIVE |

## T3 — Embedding content

| alpha | cos(F,E) | cos(X,E) | cos(F,X) | cos(ZN,E) |
|---|---|---|---|---|
| 0 | 0.052868 | 1.000000 | 0.052868 | 0.180914 |
| 0.03125 | 0.052868 | 0.935902 | 0.394917 | 0.180914 |
| 0.0625 | 0.052868 | 0.808355 | 0.622544 | 0.180914 |
| rms_match | 0.052868 | 0.726668 | 0.716707 | 0.180914 |
| 0.125 | 0.052868 | 0.587811 | 0.832839 | 0.180914 |
| 0.25 | 0.052868 | 0.367464 | 0.945490 | 0.180914 |
| 0.5 | 0.052868 | 0.219738 | 0.984980 | 0.180914 |
| 0.75 | 0.052868 | 0.165685 | 0.993168 | 0.180914 |
| 1 | 0.052868 | 0.137966 | 0.996122 | 0.180914 |

## T4 — Matched legacy comparisons

| Comparison | Interface | Real CE | Gain | Sequence gap | Max RMS | X/E | cos(X,E) |
|---|---|---|---|---|---|---|---|
| MATCHED_075_COMPARISON | legacy | 3.1871706692 | -0.0870349951 | 0.2136753911 | 0.325660 | 9.075807 | 0.081481 |
| MATCHED_075_COMPARISON | residual | 3.7593763282 | -0.6592406541 | 0.1446147055 | 0.342191 | 9.166080 | 0.165685 |
| MATCHED_100_COMPARISON | legacy | 3.4835763178 | -0.3834406438 | 0.7349820635 | 0.433431 | 12.079155 | 0.052868 |
| MATCHED_100_COMPARISON | residual | 3.5378252564 | -0.4376895824 | 0.1751630094 | 0.455488 | 12.172409 | 0.137966 |

## T5 — Late-context utility

| alpha | Global gain | 769-1023 gain | 897-1023 gain |
|---|---|---|---|
| 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| 0.03125 | -0.02610401 | -0.06092167 | -0.08125094 |
| 0.0625 | -0.06800667 | -0.11957645 | -0.15763284 |
| rms_match | -0.10637370 | -0.15801892 | -0.19964233 |
| 0.125 | -0.21998565 | -0.24419900 | -0.28142048 |
| 0.25 | -0.53656807 | -0.41298816 | -0.39062089 |
| 0.5 | -0.75007168 | -0.54651765 | -0.49319570 |
| 0.75 | -0.65924065 | -0.50449016 | -0.48153848 |
| 1 | -0.43768958 | -0.33837953 | -0.32911061 |

## Local behavior and frontier

- `alpha_rms_match`: 0.0827875253078167
- Local REAL slope at zero: +0.8353284387
- Local shuffled slope at zero: +1.0824170715
- Best-any alpha: 0.03125
- Best-bounded alpha: 0.03125
- Maximum bounded alpha: 0.75
- Historical 10× RMS threshold: 0.3550996296107769

## Direct answers Q1–Q21

### Q1

Yes. Alpha=0 identity maximum canonical delta was 0.000e+00; deterministic tops and logits were exact.

### Q2

alpha_rms_match = 0.0827875253078167, derived before outcomes from E_RMS/F_RMS.

### Q3

At alpha=.03125, REAL CE changed by +0.0261040137 relative to plain.

### Q4

The descriptive local REAL slope is +0.8353284387; it is positive (initially CE-harming). The shuffled slope is +1.0824170715.

### Q5

No.

### Q6

Best positive-alpha CE occurs at alpha=0.03125.

### Q7

Best bounded alpha is 0.03125, REAL CE 3.1262396877.

### Q8

The largest bounded tested alpha is 0.75.

### Q9

The first alpha with gap >.01 and at least 15/20 paired wins over shuffled is 0.0625.

### Q10

The first tested expansive alpha is 1.

### Q11

Residual alpha=.75 changes REAL CE versus legacy rho=.75 by +0.5722056590.

### Q12

Residual alpha=1 changes REAL CE versus legacy rho=1 by +0.0542489386.

### Q13

Restoring E changes cos(X,E) by +0.084204 at .75 and +0.085098 at 1.

### Q14

Restoring E changes X/E by +0.090273 at .75 and +0.093254 at 1.

### Q15

At best bounded alpha=0.03125, global gain is -0.0261040137 and positions 769-1023 gain is -0.0609216669. There is no global-negative/late-positive split.

### Q16

At best bounded alpha, real-vs-shuffled is 20/20 wins.

### Q17

At best bounded alpha, real-vs-zero is 0/20 wins.

### Q18

Yes. Across the sweep the largest |zero-plain| delta was 0.000e+00.

### Q19

At best-any alpha=0.03125, sequence gap is +0.0077215198 and recurrent gain is -0.0261040137; sequence-specificity does not imply positive utility at that operating point.

### Q20

Use fixed alpha=0.03125 if the recommended future experiment is authorized.

### Q21

RETRAIN RESIDUAL RECURRENCE FROM AN EARLIER CLEAN CHECKPOINT.

## Exactly one next recommendation

**RETRAIN RESIDUAL RECURRENCE FROM AN EARLIER CLEAN CHECKPOINT**

No follow-on compute was launched.

## Integrity and provenance

- Frozen 2D1B tag `experiment-2d1b-rho-window-factorial-final` resolves exactly to `e2197377d84991d5ff13eb059e203a25f143d18b`.
- C1908 SHA-256 `1d8c5f96cfe5eadc0cdb458d7cdc40b8c8cb78b15ef91a142d1c6044cd6d3864` and strict reopen passed.
- Legacy D12 rho=.75 oracle regression passed before the residual sweep.
- Alpha manifest was frozen before residual outcomes.
- Model, base, W_u, and W_g hashes were unchanged.
- Exact canonical collection `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb` and fixed derangement were used.
- Alpha=0 identities, causal shift, row isolation, and reverse-order contamination checks passed.
- No new alpha, training, window sweep, normalization variant, teacher, reconstruction, AttnRes, or HellaSwag was used.

## Artifacts

All machine-readable artifacts and P1–P10 are under `results/experiment_2d1c_c1908_residual_alpha_sweep/`.

# EXPERIMENT 2D1C COMPLETE
