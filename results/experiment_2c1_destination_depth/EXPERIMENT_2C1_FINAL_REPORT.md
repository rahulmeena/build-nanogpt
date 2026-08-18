# Experiment 2C1 — Final Report

## Outcome

All four independent destination workers completed the frozen 25M-token protocol. The largest final sequence-specific gap was 0.0412521124 at D1/Block 1. The preregistered classification is DESTINATION-DEPTH RESULT IS MIXED.

## Frozen model/data provenance

2C0 frozen tag: `experiment-2c0-separated-b1-final`  
2C0 parent commit: `677d711bc00dba0da1b80cb6369f33841ec29a51`  
2C1 branch: `experiment-2c1-destination-depth-sweep`  
Implementation commit: `cbf847f8ad43d59f38cd9cf43008562b3c64fb13`  
Results commit: `4328d3ed6cdffa4d5bbed96ba58e3c06302333a1`
Final report commit: `the immutable commit containing this file`  
Experiment 1B checkpoint SHA: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`  
Canonical validation SHA: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`  
All four workers used private clones of the same four replay-loader states.

## Damage and final destination results

| Destination | Masked | Damage | Real | Shuffled | Generic | Specific gap | Recovery % | Specific share | Real wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1/B1 | 5.9736744881 | 1.8950200796 | 5.8353391409 | 5.8765912533 | 5.8995734453 | 0.0412521124 | 7.2999% | 0.298204 | 20/20 |
| D5/B5 | 4.9225068569 | 0.8438524485 | 4.9182632208 | 4.9233091354 | 4.9221667528 | 0.0050459146 | 0.5029% | 1.189055 | 20/20 |
| D9/B9 | 5.1411294222 | 1.0624750137 | 5.1024558544 | 5.1026531458 | 5.1190280199 | 0.0001972914 | 3.6400% | 0.005101 | 12/20 |
| D12/B12 | 4.1168492794 | 0.0381948709 | 4.1140136480 | 4.1143971205 | 4.1153440714 | 0.0003834724 | 7.4241% | 0.135234 | 20/20 |

## Training trajectories

### D1 / Block 1

| Tokens | Real | Shuffled | Specific gap |
|---:|---:|---:|---:|
| 5,242,880 | 5.9533051014 | 5.9617962360 | 0.0084911346 |
| 10,485,760 | 5.9229358435 | 5.9391546011 | 0.0162187576 |
| 15,204,352 | 5.8944202185 | 5.9179884195 | 0.0235682011 |
| 25,165,824 | 5.8353391409 | 5.8765912533 | 0.0412521124 |

### D5 / Block 5

| Tokens | Real | Shuffled | Specific gap |
|---:|---:|---:|---:|
| 5,242,880 | 4.9224073172 | 4.9224833965 | 0.0000760794 |
| 10,485,760 | 4.9218292236 | 4.9224813938 | 0.0006521702 |
| 15,204,352 | 4.9208682775 | 4.9225823402 | 0.0017140627 |
| 25,165,824 | 4.9182632208 | 4.9233091354 | 0.0050459146 |

### D9 / Block 9

| Tokens | Real | Shuffled | Specific gap |
|---:|---:|---:|---:|
| 5,242,880 | 5.1379298449 | 5.1376224756 | -0.0003073692 |
| 10,485,760 | 5.1304411411 | 5.1301825523 | -0.0002585888 |
| 15,204,352 | 5.1221600533 | 5.1220158100 | -0.0001442432 |
| 25,165,824 | 5.1024558544 | 5.1026531458 | 0.0001972914 |

### D12 / Block 12

| Tokens | Real | Shuffled | Specific gap |
|---:|---:|---:|---:|
| 5,242,880 | 4.1164325476 | 4.1164831877 | 0.0000506401 |
| 10,485,760 | 4.1158476830 | 4.1159740686 | 0.0001263857 |
| 15,204,352 | 4.1153579712 | 4.1155261993 | 0.0001682281 |
| 25,165,824 | 4.1140136480 | 4.1143971205 | 0.0003834724 |

## Router specialization

| Destination | Gate | Query norm | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.02877882 | 0.60527110 | 0.73760774 | 0.40614135 | 0.16503048 | 0.16900915 | 0.25981902 | 0.00517919 |
| D5 | -0.02407707 | 0.43522668 | 0.94075834 | 0.55901042 | 0.09838491 | 0.15267877 | 0.18992589 | 0.00374314 |
| D9 | -0.02937355 | 0.62092155 | 0.51660769 | 0.14902180 | 0.31549089 | 0.39465753 | 0.14082979 | 0.00599802 |
| D12 | 0.02927095 | 0.43491694 | 0.80026288 | 0.11338999 | 0.49702841 | 0.16037853 | 0.22920307 | 0.00553269 |

## Conditional self-recurrent transfer

- D1: teacher gap 0.0412521124; self real 5.8695135747; self shuffled 5.8797442779; self gap 0.0102307031; self/teacher recovery ratio 0.752764.
- D5: SELF TEST NOT TRIGGERED
- D9: SELF TEST NOT TRIGGERED
- D12: SELF TEST NOT TRIGGERED

## Integrity

- all_preflights_passed: PASS
- trainable_parameters_exactly_1537: PASS
- base_and_teacher_gradients_none: PASS
- teacher_eval_no_grad: PASS
- future_causality_pass: PASS
- row_isolation_pass: PASS
- only_target_block_masked: PASS
- zero_gate_equals_masked: PASS
- global_targets_per_update_524288: PASS
- batch_hash_sequence_identical: PASS
- forced_fresh_process_restart_after_20: PASS
- checkpoint_strict_reload_pass: PASS
- all_evaluations_finite_and_paired: PASS
- generic_means_identical: PASS
- d1_historical_regression_pass: PASS
- writers_never_active: PASS
- no_auxiliary_loss: PASS
- no_bptt_or_temporal_gradient: PASS
- hellaswag_not_run: PASS

## Classification

DESTINATION-DEPTH RESULT IS MIXED

## Key scientific questions

Q1. NO; the latter gap is smaller by 0.0362061977.  
Q2. NO; the latter gap is smaller by 0.0048486233.  
Q3. YES; the latter gap is larger by 0.0001861811.  
Q4. YES; generic-vs-real deltas decrease overall with depth.  
Q5. YES; final source-weight vectors differ across destinations.  
Q6. D1 / Block 1 has the largest direct specific gap; it does not pass the frozen strong-support rule.  
Q7. D1 / Block 1 is the leading iterative-loop candidate by direct specific gap.

## Next-experiment recommendations

A. NO destination passed the frozen strong direct-signal rule.  
B. NO; no destination earned alternating reader→writer optimization.  
C. Retain the generic branch for Block 1; test it elsewhere only in a separate controlled protocol.  
D. NO for this experiment; a future multi-destination model requires separate approval and preregistration.  
E. YES; keep temporal credit zero for direct readers and limit it to one token only after writers are introduced.  
F. NO; direct sequence specificity is not strong enough.

No writers, reader continuation beyond 25M, multi-destination model, BPTT, iterative loops, auxiliary objectives, or HellaSwag were launched.

# EXPERIMENT 2C1 COMPLETE
