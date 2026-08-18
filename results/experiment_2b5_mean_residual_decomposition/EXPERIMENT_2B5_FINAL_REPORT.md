# Experiment 2B5 — Final Report

## Outcome

The four-GPU zero-optimizer decomposition completed with classification: **GENERIC COMPENSATION DISPLACES SEQUENCE MEMORY**. The frozen rule was `Section 45 frozen rule`.

Each checkpoint ran on its own A100-SXM4-80GB. No optimizer, LR scheduler, GradScaler, backward pass, optimizer step, parameter update, additional training token, or HellaSwag evaluation occurred.

## Central longitudinal result

| Checkpoint | Tokens | Real | Shuffled | μ-only | Residual-only | Generic retention | Residual retention | Gap α=.25 | Gap α=.5 | Gap α=1 | Gap α=2 | Residual-only gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B2 5M | 5,242,880 | 5.5900331020 | 5.6259720802 | 5.3366457701 | 6.3019540071 | 166.0525% | -85.5821% | 0.0257770538 | 0.0427144527 | 0.0359389782 | -0.3993496656 | -0.1270784140 |
| 2B2A 10M | 10,485,760 | 5.3613477468 | 5.4016084433 | 5.1452856302 | 6.1502050638 | 135.2870% | -28.8350% | 0.0265152693 | 0.0428214550 | 0.0402606964 | -0.2725460529 | -0.1755123615 |
| 2B2A 15M | 15,204,352 | 5.0959878206 | 5.1250012159 | 4.9464744329 | 5.8624111176 | 117.0355% | 12.6743% | 0.0246182203 | 0.0356348753 | 0.0290133953 | -0.1547598839 | -0.0316424608 |
| 2B3 final | 19,922,944 | 4.8141904593 | 4.8176936150 | 4.7776873112 | 5.6313026428 | 103.1483% | 29.5263% | 0.0185552120 | 0.0225783825 | 0.0035031557 | -0.1537672520 | 0.0208942413 |

## Alpha sweeps

### 2B2 5M

| α | Real loss | Shuffled loss | Specific gap | Real wins | Shuffled wins | Ties |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5.3366457701 | 5.3366457701 | 0.0000000000 | 0 | 0 | 20 |
| 0.25 | 5.3690263033 | 5.3948033571 | 0.0257770538 | 20 | 0 | 0 |
| 0.50 | 5.4273474216 | 5.4700618744 | 0.0427144527 | 20 | 0 | 0 |
| 1.00 | 5.5900331020 | 5.6259720802 | 0.0359389782 | 20 | 0 | 0 |
| 2.00 | 6.3925687790 | 5.9932191133 | -0.3993496656 | 0 | 20 | 0 |

Residual-only paired gap: -0.1270784140; real wins 4/20, shuffled wins 16/20.

### 2B2A 10M

| α | Real loss | Shuffled loss | Specific gap | Real wins | Shuffled wins | Ties |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5.1452856302 | 5.1452856302 | 0.0000000000 | 0 | 0 | 20 |
| 0.25 | 5.1704826832 | 5.1969979525 | 0.0265152693 | 20 | 0 | 0 |
| 0.50 | 5.2183840513 | 5.2612055063 | 0.0428214550 | 20 | 0 | 0 |
| 1.00 | 5.3613477468 | 5.4016084433 | 0.0402606964 | 20 | 0 | 0 |
| 2.00 | 6.0317924976 | 5.7592464447 | -0.2725460529 | 0 | 20 | 0 |

Residual-only paired gap: -0.1755123615; real wins 0/20, shuffled wins 20/20.

### 2B2A 15M

| α | Real loss | Shuffled loss | Specific gap | Real wins | Shuffled wins | Ties |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 4.9464744329 | 4.9464744329 | 0.0000000000 | 0 | 0 | 20 |
| 0.25 | 4.9633262873 | 4.9879445076 | 0.0246182203 | 20 | 0 | 0 |
| 0.50 | 4.9923415184 | 5.0279763937 | 0.0356348753 | 20 | 0 | 0 |
| 1.00 | 5.0959878206 | 5.1250012159 | 0.0290133953 | 20 | 0 | 0 |
| 2.00 | 5.5787140131 | 5.4239541292 | -0.1547598839 | 0 | 20 | 0 |

Residual-only paired gap: -0.0316424608; real wins 2/20, shuffled wins 18/20.

### 2B3 final

| α | Real loss | Shuffled loss | Specific gap | Real wins | Shuffled wins | Ties |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 4.7776873112 | 4.7776873112 | 0.0000000000 | 0 | 0 | 20 |
| 0.25 | 4.7794759274 | 4.7980311394 | 0.0185552120 | 20 | 0 | 0 |
| 0.50 | 4.7759203434 | 4.7984987259 | 0.0225783825 | 20 | 0 | 0 |
| 1.00 | 4.8141904593 | 4.8176936150 | 0.0035031557 | 16 | 4 | 0 |
| 2.00 | 5.2000593424 | 5.0462920904 | -0.1537672520 | 0 | 20 | 0 |

Residual-only paired gap: 0.0208942413; real wins 19/20, shuffled wins 1/20.

## Decomposition controls

| Checkpoint | Control | Loss | Δ vs real | Recovery from zero | Recovery retained |
|---|---|---:|---:|---:|---:|
| 2B2 5M | zero | 5.9736480713 | 0.3836149693 | 0.0000000000 | 0.0000% |
| 2B2 5M | real | 5.5900331020 | 0.0000000000 | 0.3836149693 | 100.0000% |
| 2B2 5M | mu only | 5.3366457701 | -0.2533873320 | 0.6370023012 | 166.0525% |
| 2B2 5M | residual only | 6.3019540071 | 0.7119209051 | -0.3283059359 | -85.5821% |
| 2B2 5M | residual only shuffled | 6.1748755932 | 0.5848424911 | -0.2012275219 | -52.4556% |
| 2B2 5M | independent source residual shuffle | 5.6000916481 | 0.0100585461 | 0.3735564232 | 97.3780% |
| 2B2A 10M | zero | 5.9736480713 | 0.6123003244 | 0.0000000000 | 0.0000% |
| 2B2A 10M | real | 5.3613477468 | 0.0000000000 | 0.6123003244 | 100.0000% |
| 2B2A 10M | mu only | 5.1452856302 | -0.2160621166 | 0.8283624411 | 135.2870% |
| 2B2A 10M | residual only | 6.1502050638 | 0.7888573170 | -0.1765569925 | -28.8350% |
| 2B2A 10M | residual only shuffled | 5.9746927023 | 0.6133449554 | -0.0010446310 | -0.1706% |
| 2B2A 10M | independent source residual shuffle | 5.3786869764 | 0.0173392296 | 0.5949610949 | 97.1682% |
| 2B2A 15M | zero | 5.9736480713 | 0.8776602507 | 0.0000000000 | 0.0000% |
| 2B2A 15M | real | 5.0959878206 | 0.0000000000 | 0.8776602507 | 100.0000% |
| 2B2A 15M | mu only | 4.9464744329 | -0.1495133877 | 1.0271736383 | 117.0355% |
| 2B2A 15M | residual only | 5.8624111176 | 0.7664232969 | 0.1112369537 | 12.6743% |
| 2B2A 15M | residual only shuffled | 5.8307686567 | 0.7347808361 | 0.1428794146 | 16.2796% |
| 2B2A 15M | independent source residual shuffle | 5.1024324417 | 0.0064446211 | 0.8712156296 | 99.2657% |
| 2B3 final | zero | 5.9736480713 | 1.1594576120 | 0.0000000000 | 0.0000% |
| 2B3 final | real | 4.8141904593 | 0.0000000000 | 1.1594576120 | 100.0000% |
| 2B3 final | mu only | 4.7776873112 | -0.0365031481 | 1.1959607601 | 103.1483% |
| 2B3 final | residual only | 5.6313026428 | 0.8171121836 | 0.3423454285 | 29.5263% |
| 2B3 final | residual only shuffled | 5.6521968842 | 0.8380064249 | 0.3214511871 | 27.7243% |
| 2B3 final | independent source residual shuffle | 4.7925686121 | -0.0216218472 | 1.1810794592 | 101.8648% |

## Memory geometry

| Checkpoint | Source | μ RMS | Memory RMS | Residual RMS | Residual/Memory | Mean cosine(memory, μ) |
|---|---|---:|---:|---:|---:|---:|
| 2B2 5M | v16 | 0.078742 | 0.193348 | 0.176076 | 0.910675 | 0.411990 |
| 2B2 5M | v17 | 0.163616 | 0.191949 | 0.097490 | 0.507870 | 0.856423 |
| 2B2 5M | v20 | 0.072405 | 0.194241 | 0.179511 | 0.924166 | 0.382425 |
| 2B2 5M | v24 | 0.059103 | 0.186654 | 0.177020 | 0.948395 | 0.320153 |
| 2B2A 10M | v16 | 0.074177 | 0.193516 | 0.178164 | 0.920667 | 0.389261 |
| 2B2A 10M | v17 | 0.165538 | 0.195242 | 0.100974 | 0.517147 | 0.851582 |
| 2B2A 10M | v20 | 0.070852 | 0.195700 | 0.181628 | 0.928089 | 0.372819 |
| 2B2A 10M | v24 | 0.055467 | 0.189045 | 0.180634 | 0.955519 | 0.298415 |
| 2B2A 15M | v16 | 0.069176 | 0.194616 | 0.181305 | 0.931604 | 0.362491 |
| 2B2A 15M | v17 | 0.170434 | 0.201734 | 0.105971 | 0.525267 | 0.847290 |
| 2B2A 15M | v20 | 0.069442 | 0.198057 | 0.184598 | 0.932041 | 0.362790 |
| 2B2A 15M | v24 | 0.052551 | 0.192652 | 0.185219 | 0.961423 | 0.278961 |
| 2B3 final | v16 | 0.065124 | 0.196595 | 0.184922 | 0.940626 | 0.338614 |
| 2B3 final | v17 | 0.179760 | 0.212373 | 0.112353 | 0.529004 | 0.845496 |
| 2B3 final | v20 | 0.069049 | 0.201193 | 0.187987 | 0.934357 | 0.356599 |
| 2B3 final | v24 | 0.051738 | 0.197511 | 0.190484 | 0.964429 | 0.268623 |

## Generic-direction cosine matrices

### v16

| | 5M | 10M | 15M | Final |
|---|---:|---:|---:|---:|
| 5M | 1.000000 | 0.995408 | 0.974479 | 0.923367 |
| 10M | 0.995408 | 1.000000 | 0.991379 | 0.954978 |
| 15M | 0.974479 | 0.991379 | 1.000000 | 0.985294 |
| Final | 0.923367 | 0.954978 | 0.985294 | 1.000000 |

μ RMS trajectory: 5M=0.078742, 10M=0.074177, 15M=0.069176, Final=0.065124.

### v17

| | 5M | 10M | 15M | Final |
|---|---:|---:|---:|---:|
| 5M | 1.000000 | 0.996550 | 0.982394 | 0.953356 |
| 10M | 0.996550 | 1.000000 | 0.994283 | 0.973729 |
| 15M | 0.982394 | 0.994283 | 1.000000 | 0.992111 |
| Final | 0.953356 | 0.973729 | 0.992111 | 1.000000 |

μ RMS trajectory: 5M=0.163616, 10M=0.165538, 15M=0.170434, Final=0.179760.

### v20

| | 5M | 10M | 15M | Final |
|---|---:|---:|---:|---:|
| 5M | 1.000000 | 0.994888 | 0.974418 | 0.930096 |
| 10M | 0.994888 | 1.000000 | 0.992005 | 0.961426 |
| 15M | 0.974418 | 0.992005 | 1.000000 | 0.988101 |
| Final | 0.930096 | 0.961426 | 0.988101 | 1.000000 |

μ RMS trajectory: 5M=0.072405, 10M=0.070852, 15M=0.069442, Final=0.069049.

### v24

| | 5M | 10M | 15M | Final |
|---|---:|---:|---:|---:|
| 5M | 1.000000 | 0.989323 | 0.946057 | 0.859747 |
| 10M | 0.989323 | 1.000000 | 0.982821 | 0.922249 |
| 15M | 0.946057 | 0.982821 | 1.000000 | 0.976798 |
| Final | 0.859747 | 0.922249 | 0.976798 | 1.000000 |

μ RMS trajectory: 5M=0.059103, 10M=0.055467, 15M=0.052551, Final=0.051738.

## Reader routing under decomposition

| Checkpoint | Control | Routing v16/v17/v20/v24 | Entropy | Input RMS v16/v17/v20/v24 | Top-down RMS | Feedback RMS |
|---|---|---|---:|---|---:|---:|
| 2B2 5M | real | 0.626302/0.017401/0.141689/0.214608 | 0.479758 | 0.193159/0.191762/0.194051/0.186472 | 0.162416 | 0.025694 |
| 2B2 5M | mu only | 0.989819/0.000287/0.000810/0.009077 | 0.057178 | 0.078671/0.163446/0.072344/0.059040 | 0.078125 | 0.012359 |
| 2B2 5M | residual only | 0.282016/0.226989/0.299004/0.191991 | 0.582283 | 0.177331/0.150306/0.176206/0.174617 | 0.159238 | 0.025191 |
| 2B2 5M | alpha 0.5 real | 0.807458/0.002953/0.049848/0.139742 | 0.335908 | 0.117582/0.172214/0.116128/0.107673 | 0.104419 | 0.016519 |
| 2B2 5M | alpha 1 real | 0.626302/0.017401/0.141689/0.214608 | 0.479758 | 0.193159/0.191762/0.194051/0.186472 | 0.162416 | 0.025694 |
| 2B2 5M | alpha 2 real | 0.418867/0.127401/0.208902/0.244830 | 0.601947 | 0.375832/0.309835/0.362860/0.357227 | 0.315339 | 0.049887 |
| 2B2A 10M | real | 0.601998/0.014763/0.170834/0.212405 | 0.468923 | 0.193327/0.195052/0.195509/0.188860 | 0.163117 | 0.025805 |
| 2B2A 10M | mu only | 0.988097/0.000254/0.001104/0.010553 | 0.065797 | 0.074100/0.165406/0.070780/0.055419 | 0.073262 | 0.011592 |
| 2B2A 10M | residual only | 0.341669/0.160551/0.291618/0.206161 | 0.545908 | 0.175965/0.132409/0.177669/0.174158 | 0.154670 | 0.024469 |
| 2B2A 10M | alpha 0.5 real | 0.782749/0.001814/0.071050/0.144387 | 0.345298 | 0.115587/0.174422/0.116080/0.107431 | 0.102113 | 0.016154 |
| 2B2A 10M | alpha 1 real | 0.601998/0.014763/0.170834/0.212405 | 0.468923 | 0.193327/0.195052/0.195509/0.188860 | 0.163117 | 0.025805 |
| 2B2A 10M | alpha 2 real | 0.431370/0.089051/0.230570/0.249009 | 0.566196 | 0.374615/0.288063/0.366072/0.359821 | 0.308223 | 0.048761 |
| 2B2A 15M | real | 0.553503/0.024143/0.218052/0.204301 | 0.465643 | 0.194426/0.201537/0.197864/0.192464 | 0.164257 | 0.025985 |
| 2B2A 15M | mu only | 0.986769/0.000248/0.001774/0.011204 | 0.073187 | 0.069101/0.170278/0.069383/0.052501 | 0.068343 | 0.010812 |
| 2B2A 15M | residual only | 0.417602/0.071161/0.286048/0.225189 | 0.496975 | 0.175800/0.109748/0.179586/0.175511 | 0.147222 | 0.023291 |
| 2B2A 15M | alpha 0.5 real | 0.735292/0.002458/0.117118/0.145131 | 0.364032 | 0.113876/0.179236/0.116464/0.107976 | 0.099829 | 0.015793 |
| 2B2A 15M | alpha 1 real | 0.553503/0.024143/0.218052/0.204301 | 0.465643 | 0.194426/0.201537/0.197864/0.192464 | 0.164257 | 0.025985 |
| 2B2A 15M | alpha 2 real | 0.418325/0.083240/0.261791/0.236645 | 0.530407 | 0.373454/0.279741/0.370435/0.364326 | 0.305399 | 0.048314 |
| 2B3 final | real | 0.448705/0.093113/0.277623/0.180560 | 0.496218 | 0.196403/0.212166/0.200997/0.197318 | 0.164860 | 0.026242 |
| 2B3 final | mu only | 0.981755/0.000250/0.006857/0.011148 | 0.101232 | 0.065053/0.179573/0.068971/0.051684 | 0.064151 | 0.010211 |
| 2B3 final | residual only | 0.461046/0.042940/0.274150/0.221864 | 0.425117 | 0.178443/0.108418/0.181941/0.178923 | 0.150266 | 0.023919 |
| 2B3 final | alpha 0.5 real | 0.629055/0.011745/0.219220/0.139979 | 0.400762 | 0.113039/0.187490/0.117615/0.109579 | 0.098478 | 0.015675 |
| 2B3 final | alpha 1 real | 0.448705/0.093113/0.277623/0.180560 | 0.496218 | 0.196403/0.212166/0.200997/0.197318 | 0.164860 | 0.026242 |
| 2B3 final | alpha 2 real | 0.348346/0.163454/0.285141/0.203058 | 0.516388 | 0.377725/0.289811/0.376684/0.371758 | 0.301714 | 0.048026 |

## Scientific questions

### Q1. Does generic recovery increase over training?

Not monotonically: generic-recovery retention was 166.0525% → 135.2870% → 117.0355% → 103.1483%.

### Q2. Does residual-only utility decrease?

Not monotonically: residual-recovery retention was -85.5821% → -28.8350% → 12.6743% → 29.5263%, while the α=1 sequence gaps were 0.0359389782 → 0.0402606964 → 0.0290133953 → 0.0035031557.

### Q3. At final 2B3, is sequence memory gone or underweighted?

The frozen classification is **GENERIC COMPENSATION DISPLACES SEQUENCE MEMORY**. Final α=1 and α=2 gaps were 0.0035031557 and -0.1537672520; real α=2 won 0/20 paired batches.

### Q4. Does the generic corrective direction converge?

The 15M→final per-source mean cosines were v16=0.985294, v17=0.992111, v20=0.988101, v24=0.976798. These values, together with the full matrices above, measure directional convergence without equating geometry with utility.

### Q5. Is the final generic correction already present early?

At 5M and 10M, μ-only retained 166.0525% and 135.2870% of real recovery, versus 103.1483% at final 2B3. This directly locates how early the calibration-derived correction became useful.

## Architectural decisions

### A. Split generic correction and sequence memory into two branches?

Yes. The measured generic and centered-residual interventions should be independently controllable in the next approved architecture.

### B. Freeze or static-initialize the generic branch?

Yes, initially. A frozen calibration-derived branch would isolate whether optimization can preserve sequence residuals without generic drift.

### C. Mean-center the sequence-memory branch?

Yes. Centering should be evaluated as a distinct branch operation, not interpreted as additive attribution.

### D. Is next-token cross entropy alone sufficient for actual memory?

No: the α=1 correct-sequence gap declined materially across the observed optimization trajectory.

### E. Keep mask-depth experiments paused?

Yes. Keep deeper-mask training paused until generic compensation and sequence-specific recurrence are separated.

### F. Keep one-token temporal credit?

Yes. This diagnostic performed no training and provides no controlled evidence authorizing a longer horizon.

## Interpretation limits

μ is a checkpoint-specific estimate from the frozen, disjoint calibration set. It is not a learned model parameter, not the true FineWeb expectation, and not necessarily the only generic component. μ-only, residual-only, and combined effects are nonlinear interventions; their recoveries must not be added or treated as Shapley attributions.

## Integrity and performance

- 2B2A_10M_SHA_exact: PASS
- 2B2A_15M_SHA_exact: PASS
- 2B2_5M_SHA_exact: PASS
- 2B3_final_SHA_exact: PASS
- 2B4_calibration_manifest_reused_exactly: PASS
- C1_real_shuffled_regression: PASS
- C3_global_template_regression: PASS
- C3_independent_source_regression: PASS
- C3_real_regression: PASS
- C3_shuffled_regression: PASS
- HellaSwag: NOT RUN
- all_losses_finite: PASS
- all_memory_states_finite: PASS
- alpha_0_mu_identity: PASS
- alpha_1_decomposition_identity: PASS
- backward_calls_zero: PASS
- calibration_canonical_data_disjoint: PASS
- canonical_validation_hash_exact: PASS
- closed_loop_rollout_semantics: PASS
- future_causality: PASS
- grad_scalers_created_zero: PASS
- hellaswag_not_run: PASS
- model_hashes_before_after_identical: PASS
- optimizer_objects_created_zero: PASS
- optimizer_steps_zero: PASS
- parameter_updates_zero: PASS
- row_isolation: PASS
- scheduler_objects_created_zero: PASS

Total four-GPU wall time: 25069.2 seconds.

2B4 frozen tag: `experiment-2b4-memory-content-mask-depth-final`

2B4 parent commit: `692fd80ba9fb5e81731397dcd4bf149c3c705d41`

2B5 branch: `experiment-2b5-mean-residual-decomposition-4gpu`

Implementation commit: `17156ad83c080efd407c51a3475cf74837189473`

Results commit: `8aac1db9a82163c625d0c1d4ee4b239135a48b6b`

Optimizer updates: `0`

Additional training tokens: `0`

## Classification

GENERIC COMPENSATION DISPLACES SEQUENCE MEMORY

# EXPERIMENT 2B5 COMPLETE
