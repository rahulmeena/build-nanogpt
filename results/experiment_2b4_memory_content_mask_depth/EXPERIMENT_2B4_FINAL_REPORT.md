# Experiment 2B4 — Final Report

## Outcome

The zero-optimizer diagnostic completed with classification: **MASK PRESSURE DESTROYS RECURRENT UTILITY**. Part A measured a real recurrent loss of 4.8141904593 against gate-zero 5.9736480713. Part B's specificity trajectory was classified as decreasing.

The automated preliminary rule emitted `MASK PRESSURE DOES NOT INCREASE RECURRENT DEPENDENCE`; final protocol review corrected that label because real recurrent recovery fell from 0.6123003244 at depth 1 to 0.0134044409 at depth 4 (a 97.8108% loss), while the specific gap fell from 0.0402606964 to -0.0029426336. Section 44 defines this loss of useful feedback advantage with depth as destruction. Raw measurements were not changed.

No optimizer, scheduler, GradScaler, backward pass, optimizer step, parameter update, training continuation, or HellaSwag evaluation occurred.

## Part A — Memory content

| Control | Loss | Δ vs real | Recovery | Real recovery retained | Real wins |
|---|---:|---:|---:|---:|---:|
| zero | 5.9736480713 | 1.1594576120 | 0.0000000000 | 0.0000% | 20/20 |
| real | 4.8141904593 | 0.0000000000 | 1.1594576120 | 100.0000% | 0/20 |
| coherent shuffled | 4.8176936150 | 0.0035031557 | 1.1559544563 | 99.6979% | 16/20 |
| independent-source shuffled | 4.7925686121 | -0.0216218472 | 1.1810794592 | 101.8648% | 0/20 |
| leave-one-out batch mean | 4.7792271852 | -0.0349632740 | 1.1944208860 | 103.0155% | 0/20 |
| position-conditioned template | 4.7750704288 | -0.0391200304 | 1.1985776424 | 103.3740% | 0/20 |
| global template | 4.7776873112 | -0.0365031481 | 1.1959607601 | 103.1483% | 0/20 |
| norm-matched random | 5.7395672560 | 0.9253767967 | 0.2340808153 | 20.1888% | 20/20 |
| same-sequence lag-8 | 4.8211796045 | 0.0069891453 | 1.1524684668 | 99.3972% | 18/20 |
| same-sequence lag-32 | 4.8561311245 | 0.0419406652 | 1.1175169468 | 96.3827% | 20/20 |

### Paired statistics versus real

| Control | Real wins | Control wins | Ties | Mean | Median | Sample std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 20 | 0 | 0 | 1.1594576120 | 1.1575393677 | 0.0206375229 | 1.1092009544 | 1.1851148605 |
| real | 0 | 0 | 20 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| coherent shuffled | 16 | 4 | 0 | 0.0035031557 | 0.0045094490 | 0.0051618589 | -0.0094823837 | 0.0134239197 |
| independent-source shuffled | 0 | 20 | 0 | -0.0216218472 | -0.0213782787 | 0.0053036988 | -0.0332293510 | -0.0118942261 |
| leave-one-out batch mean | 0 | 20 | 0 | -0.0349632740 | -0.0340487957 | 0.0090854981 | -0.0515446663 | -0.0172481537 |
| position-conditioned template | 0 | 20 | 0 | -0.0391200304 | -0.0387849808 | 0.0061051519 | -0.0530295372 | -0.0280508995 |
| global template | 0 | 20 | 0 | -0.0365031481 | -0.0363216400 | 0.0061893874 | -0.0506086349 | -0.0253210068 |
| norm-matched random | 20 | 0 | 0 | 0.9253767967 | 0.9233431816 | 0.0192998038 | 0.8945460320 | 0.9563302994 |
| same-sequence lag-8 | 18 | 2 | 0 | 0.0069891453 | 0.0061986446 | 0.0049203742 | -0.0003576279 | 0.0175709724 |
| same-sequence lag-32 | 20 | 0 | 0 | 0.0419406652 | 0.0403604507 | 0.0062795659 | 0.0295319557 | 0.0553593636 |

### Lag-restricted comparisons

| Control | Target subset | Lag loss | Matched real loss | Specific delta | Real wins |
|---|---|---:|---:|---:|---:|
| same-sequence lag-8 | t >= 8 | 4.8159425259 | 4.8101121902 | 0.0058303356 | 17/20 |
| same-sequence lag-32 | t >= 32 | 4.8184588671 | 4.8009437799 | 0.0175150871 | 20/20 |

### Routing and state diagnostics

| Control | Input RMS v16/v17/v20/v24 | Routing v16/v17/v20/v24 | Entropy | Top-down RMS | Feedback RMS |
|---|---|---|---:|---:|---:|
| zero | 0.189993/0.184399/0.192164/0.179920 | 0.705913/0.014355/0.172452/0.107280 | 0.324431 | 0.167796 | 0.000000 |
| real | 0.196403/0.212166/0.200997/0.197318 | 0.448705/0.093113/0.277623/0.180560 | 0.496218 | 0.164860 | 0.026242 |
| coherent shuffled | 0.195245/0.212042/0.199979/0.196710 | 0.457192/0.099652/0.275109/0.168047 | 0.491337 | 0.163928 | 0.026093 |
| independent-source shuffled | 0.195254/0.212500/0.200343/0.197465 | 0.442040/0.103732/0.290563/0.163665 | 0.487594 | 0.160667 | 0.025574 |
| leave-one-out batch mean | 0.071094/0.179587/0.074400/0.060222 | 0.945846/0.000277/0.025618/0.028259 | 0.175100 | 0.067939 | 0.010799 |
| position-conditioned template | 0.067814/0.181783/0.071024/0.054922 | 0.960422/0.000258/0.016791/0.022530 | 0.165179 | 0.065436 | 0.010416 |
| global template | 0.065053/0.179573/0.068971/0.051684 | 0.981755/0.000250/0.006857/0.011148 | 0.101232 | 0.064151 | 0.010211 |
| norm-matched random | 0.189170/0.183427/0.192254/0.180288 | 0.249699/0.250322/0.250338/0.249641 | 0.727175 | 0.142276 | 0.022615 |
| same-sequence lag-8 | 0.194415/0.210566/0.198494/0.194298 | 0.448902/0.100875/0.280851/0.169371 | 0.500749 | 0.162780 | 0.025911 |
| same-sequence lag-32 | 0.189590/0.204299/0.193493/0.189156 | 0.450925/0.100325/0.278732/0.170017 | 0.516279 | 0.158964 | 0.025303 |

## Part A — What the recurrent signal contains

### A1 — Exact sequence identity

The measured coherent-shuffle penalty was 0.0035031557; real won 16/20 paired batches. Exact row identity therefore contributed exactly this measured amount; the control does not support attributing the remaining recovery to row identity.

### A2 — Coherence among sources

Breaking the common donor across v16/v17/v20/v24 changed loss by -0.0251250029 relative to coherent shuffling (independent-source loss 4.7925686121).

### A3 — Absolute-position-conditioned generic state

The position template retained 103.3740% of real recovery at loss 4.7750704288.

### A4 — Constant generic template

The global template retained 103.1483% of real recovery at loss 4.7776873112.

### A5 — Memory-vector norms

Norm-matched random directions retained 20.1888% of real recovery at loss 5.7395672560. This directly measures what norms alone can supply without learned direction/content.

### A6 — Temporal alignment versus row identity

Lag-8 increased restricted loss over matched real by 0.0058303356, versus the coherent cross-sequence identity gap of 0.0035031557; the difference was +0.0023271799.

## Part B — Mask-depth sweep

| Mask depth | Zero loss | Real loss | Shuffled loss | Specific gap | Recovery % | Specific share | Real wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.9736480713 | 5.3613477468 | 5.4016084433 | 0.0402606964 | 32.3115% | 6.5753% | 20/20 |
| 2 | 5.9744093418 | 5.4324810266 | 5.4381460905 | 0.0056650639 | 28.5864% | 1.0454% | 14/20 |
| 3 | 6.9039013624 | 6.8043104887 | 6.7918549061 | -0.0124555826 | 3.5250% | -12.5068% | 0/20 |
| 4 | 7.2173499584 | 7.2039455175 | 7.2010028839 | -0.0029426336 | 0.4271% | -21.9527% | 9/20 |

### Gap minus depth 1

| Depth | Gap minus depth 1 |
|---:|---:|
| 2 | -0.0345956326 |
| 3 | -0.0527162790 |
| 4 | -0.0432033300 |

## Gap trajectory

- depth 1: 0.0402606964
- depth 2: 0.0056650639
- depth 3: -0.0124555826
- depth 4: -0.0029426336

Trajectory: **decreasing**. Strong-support depths: none.

## Conditional Part C

**NOT TRIGGERED**

Part B did not satisfy STRONG MASK-PRESSURE SUPPORT

## Integrity

- 2B2A 10M checkpoint SHA: PASS
- 2B3 checkpoint SHA: PASS
- Block-1 d=1 regression: PASS
- HellaSwag: NOT RUN
- all losses finite: PASS
- all memory-control causality checks: PASS
- backward calls: 0
- calibration/evaluation data disjoint: PASS
- canonical validation SHA: PASS
- future causality: PASS
- generalized mask d=1 FP32 equivalence: PASS
- masked-cache absence: PASS
- model parameter hashes before/after: IDENTICAL
- no optimizer constructed for result path: PASS
- optimizer steps: 0
- row isolation: PASS
- unmasked-cache health: PASS

Final 2B3 checkpoint: `7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b`

2B2A 10M checkpoint: `de5e04f817dcfa5dd8a4dcc6e503ec86d8545d558d837b517c7259917218dff3`

Canonical validation: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`

## Classification

MASK PRESSURE DESTROYS RECURRENT UTILITY

## Recommendations

### B1 — Train Blocks 1–2 masked next?

No. The preregistered diagnostic did not establish strong sequence-specific mask pressure.

### B2 — Feedback destination

Do not add a second destination here. If masking is revisited, first preserve the single Block-1 destination so the comparison remains identified.

### B3 — Source checkpoint

Do not select a training source yet. If a follow-up diagnostic is approved, retain the 10M checkpoint as the high-specificity reference rather than silently advancing lineage.

### B4 — Temporal credit

Yes. Experiment 2B4 performed no training and provides no evidence authorizing a longer credit horizon; keep temporal credit at one token.

### B5 — Future control set

No. Keep coherent shuffled memory, and make the global-template, norm-matched-random, and same-sequence lag-8 controls mandatory when mechanism attribution is claimed.

### Observed classification

MASK PRESSURE DESTROYS RECURRENT UTILITY

No optimizer update may be launched automatically after this diagnostic.

# EXPERIMENT 2B4 COMPLETE
