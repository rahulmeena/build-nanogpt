# Experiment 2C2 Final Report

Classification: **MATCHED FEEDBACK DOES NOT RESCUE CUMULATIVE MASKING**

Frozen rule: Matched-destination gain is < 0.020 for C2, C3, and C4.

## Provenance

- 2C1 frozen tag: `experiment-2c1-destination-depth-final`
- 2C1 parent commit: `e4a5eec76181db0581d486e0f5724f196c22db64`
- 2C2 branch: `experiment-2c2-cumulative-low-kv-matched-feedback`
- Implementation commit: `48c4bf3d7c327484b2ca0037b3d1a175aa0f6df5`
- Evaluation-only finalize commits: `d0736000235f76c2dd9cea277c7c741c11d27b2a`, `f6969b388f58675b300f9f12745600c20f84470e`
- Results commit: `404ab486a47f891030789496819a2717fd4a5491`
- Final-report commit: `the immutable commit containing this file`
- Base checkpoint SHA256: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`

## Main final table

| Config | Masked blocks | Readers | Masked | Real | Shuffled | Generic | Specific gap | Recovery % | Specific share | Real wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | B1 | 1 | 5.9736744881 | 5.8353391409 | 5.8765912533 | 5.8995734453 | 0.0412521124 | 7.299941 | 0.298204 | 20/20 |
| C2 | B1-B2 | 2 | 5.9744511843 | 5.8394160986 | 5.8794318676 | 5.9065647364 | 0.0400157690 | 7.122867 | 0.296336 | 20/20 |
| C3 | B1-B2-B3 | 3 | 6.9038509846 | 6.7783385992 | 6.8367046356 | 6.8770476580 | 0.0583660364 | 4.442607 | 0.465022 | 20/20 |
| C4 | B1-B2-B3-B4 | 4 | 7.2172823668 | 7.1104239941 | 7.1660341024 | 7.1948127031 | 0.0556101084 | 3.404621 | 0.520409 | 20/20 |

## Matched-feedback incremental table

| Config | B1-only loss | All-reader real loss | Gain from matched readers | Positive batches |
|---|---:|---:|---:|---:|
| C2 | 5.8428148031 | 5.8394160986 | 0.0033987045 | 20/20 |
| C3 | 6.7910120010 | 6.7783385992 | 0.0126734018 | 20/20 |
| C4 | 7.1258494139 | 7.1104239941 | 0.0154254198 | 20/20 |

## Training trajectory

| Config | Tokens | Masked | Real | Shuffled | Specific gap | Recovery % |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5242880 | 5.9736744881 | 5.9533051014 | 5.9617962360 | 0.0084911346 | 1.074890 |
| C1 | 10485760 | 5.9736744881 | 5.9229358435 | 5.9391546011 | 0.0162187576 | 2.677473 |
| C1 | 15204352 | 5.9736744881 | 5.8944202185 | 5.9179884195 | 0.0235682011 | 4.182239 |
| C1 | 25165824 | 5.9736744881 | 5.8353391409 | 5.8765912533 | 0.0412521124 | 7.299941 |
| C2 | 5242880 | 5.9744511843 | 5.9547081232 | 5.9635372400 | 0.0088291168 | 1.041412 |
| C2 | 10485760 | 5.9744511843 | 5.9253852606 | 5.9419242620 | 0.0165390015 | 2.588143 |
| C2 | 15204352 | 5.9744511843 | 5.8973955154 | 5.9208139658 | 0.0234184504 | 4.064553 |
| C2 | 25165824 | 5.9744511843 | 5.8394160986 | 5.8794318676 | 0.0400157690 | 7.122867 |
| C3 | 5242880 | 6.9038509846 | 6.8802060127 | 6.8934556723 | 0.0132496595 | 0.836932 |
| C3 | 10485760 | 6.9038509846 | 6.8541112900 | 6.8835628033 | 0.0294515133 | 1.760575 |
| C3 | 15204352 | 6.9038509846 | 6.8332338095 | 6.8759441376 | 0.0427103281 | 2.499549 |
| C3 | 25165824 | 6.9038509846 | 6.7783385992 | 6.8367046356 | 0.0583660364 | 4.442607 |
| C4 | 5242880 | 7.2172823668 | 7.1963164091 | 7.2073237181 | 0.0110073090 | 0.667998 |
| C4 | 10485760 | 7.2172823668 | 7.1725871801 | 7.1967770815 | 0.0241899014 | 1.424036 |
| C4 | 15204352 | 7.2172823668 | 7.1528661489 | 7.1885401726 | 0.0356740236 | 2.052369 |
| C4 | 25165824 | 7.2172823668 | 7.1104239941 | 7.1660341024 | 0.0556101084 | 3.404621 |

## Progressive reader activation

| Config | Active readers | Loss | Delta from previous |
|---|---|---:|---:|
| C2 | none | 5.9744511843 | 0.0000000000 |
| C2 | B1 | 5.8428148031 | 0.1316363811 |
| C2 | B1+B2 | 5.8394160986 | 0.0033987045 |
| C3 | none | 6.9038509846 | 0.0000000000 |
| C3 | B1 | 6.7910120010 | 0.1128389835 |
| C3 | B1+B2 | 6.7885278702 | 0.0024841309 |
| C3 | B1+B2+B3 | 6.7783385992 | 0.0101892710 |
| C4 | none | 7.2172823668 | 0.0000000000 |
| C4 | B1 | 7.1258494139 | 0.0914329529 |
| C4 | B1+B2 | 7.1240014076 | 0.0018480062 |
| C4 | B1+B2+B3 | 7.1173286676 | 0.0066727400 |
| C4 | B1+B2+B3+B4 | 7.1104239941 | 0.0069046736 |

## Leave-one-reader-out

| Config | Reader removed | Ablated loss | Delta vs all-real | Positive batches |
|---|---|---:|---:|---:|
| C1 | B1 | 5.9736744881 | 0.1383353472 | 20/20 |
| C2 | B1 | 5.9704186440 | 0.1310025454 | 20/20 |
| C2 | B2 | 5.8428148031 | 0.0033987045 | 20/20 |
| C3 | B1 | 6.8907103062 | 0.1123717070 | 20/20 |
| C3 | B2 | 6.7807003975 | 0.0023617983 | 20/20 |
| C3 | B3 | 6.7885278702 | 0.0101892710 | 20/20 |
| C4 | B1 | 7.2017721415 | 0.0913481474 | 20/20 |
| C4 | B2 | 7.1122249603 | 0.0018009663 | 20/20 |
| C4 | B3 | 7.1170634508 | 0.0066394567 | 20/20 |
| C4 | B4 | 7.1173286676 | 0.0069046736 | 20/20 |

## Generic-template control

| Config | Masked | Generic | Shuffled | Real | Generic-real | Shuffled-real |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5.9736744881 | 5.8995734453 | 5.8765912533 | 5.8353391409 | 0.0642343044 | 0.0412521124 |
| C2 | 5.9744511843 | 5.9065647364 | 5.8794318676 | 5.8394160986 | 0.0671486378 | 0.0400157690 |
| C3 | 6.9038509846 | 6.8770476580 | 6.8367046356 | 6.7783385992 | 0.0987090588 | 0.0583660364 |
| C4 | 7.2172823668 | 7.1948127031 | 7.1660341024 | 7.1104239941 | 0.0843887091 | 0.0556101084 |

## Final readers

| Config | Destination | Gate | Query norm | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | B1 | 0.0287788194 | 0.6052711010 | 0.7376077384 | 0.4061413527 | 0.1650304757 | 0.1690091498 | 0.2598190166 | 0.0051791935 |
| C2 | B1 | 0.0287782904 | 0.5994858146 | 0.7650935531 | 0.3858857229 | 0.1913199544 | 0.1599657089 | 0.2628286131 | 0.0051580094 |
| C2 | B2 | -0.0294043999 | 0.5964139700 | 0.5193847626 | 0.3726048827 | 0.0246090968 | 0.2849340513 | 0.3178519696 | 0.0057902629 |
| C3 | B1 | 0.0287692901 | 0.4221085906 | 0.7865854532 | 0.1100261644 | 0.6072065294 | 0.0987665549 | 0.1840007469 | 0.0055082500 |
| C3 | B2 | -0.0300003383 | 0.5991156697 | 0.6693145096 | 0.5722347528 | 0.0143611242 | 0.1620681651 | 0.2513359629 | 0.0054554815 |
| C3 | B3 | -0.0298277363 | 0.5029664636 | 0.6417841434 | 0.2930488601 | 0.0167379823 | 0.4316183925 | 0.2585947663 | 0.0057819930 |
| C4 | B1 | 0.0287670735 | 0.4329772592 | 0.3853874221 | 0.0265033858 | 0.8647929877 | 0.0400583327 | 0.0686452916 | 0.0063101114 |
| C4 | B2 | -0.0298649222 | 0.5499238968 | 0.5982266277 | 0.4572518066 | 0.0183461408 | 0.2154938973 | 0.3089081630 | 0.0057625362 |
| C4 | B3 | -0.0298368931 | 0.4740877151 | 0.5456555158 | 0.3170889169 | 0.0174108127 | 0.3204460844 | 0.3450541869 | 0.0056728740 |
| C4 | B4 | -0.0294666681 | 0.5022071004 | 0.5451370955 | 0.6555583954 | 0.0069766948 | 0.1889318667 | 0.1485330433 | 0.0056514415 |

## B1 reader evolution

| Config | Gate | Query norm | Entropy | v16 | v17 | v20 | v24 | RMS displacement | Feedback RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1/B1 | 0.0287788194 | 0.6052711010 | 0.7376077384 | 0.4061413527 | 0.1650304757 | 0.1690091498 | 0.2598190166 | 0.5813720822 | 0.0051791935 |
| C2/B1 | 0.0287782904 | 0.5994858146 | 0.7650935531 | 0.3858857229 | 0.1913199544 | 0.1599657089 | 0.2628286131 | 0.5726953149 | 0.0051580094 |
| C3/B1 | 0.0287692901 | 0.4221085906 | 0.7865854532 | 0.1100261644 | 0.6072065294 | 0.0987665549 | 0.1840007469 | 0.3931796551 | 0.0055082500 |
| C4/B1 | 0.0287670735 | 0.4329772592 | 0.3853874221 | 0.0265033858 | 0.8647929877 | 0.0400583327 | 0.0686452916 | 0.4106724262 | 0.0063101114 |

- Pairwise B1 query cosines: {'C1-C2': 0.9869624376296997, 'C1-C3': 0.589069664478302, 'C1-C4': 0.48263856768608093, 'C2-C3': 0.6344634294509888, 'C2-C4': 0.5265815258026123, 'C3-C4': 0.9471350908279419}

## Conditional self-recurrent transfer

| Config | Status | Teacher real | Teacher shuffled | Teacher gap | Teacher recovery | Self real | Self shuffled | Self gap | Self recovery | Self/teacher recovery | Self B1-only | Self all-readers | Self matched gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | TRIGGERED | 5.8353391409 | 5.8765912533 | 0.0412521124 | 0.1383353472 | 5.8695135747 | 5.8797442779 | 0.0102307031 | 0.1041609133 | 0.752959 | 5.8695135747 | 5.8695135747 | 0.0000000000 |
| C2 | SELF TEST NOT TRIGGERED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| C3 | SELF TEST NOT TRIGGERED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| C4 | SELF TEST NOT TRIGGERED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Scientific questions

- Q1: NO under the frozen 0.020 threshold; gain is 0.0033987045.
- Q2: NO under the frozen 0.020 threshold; gain is 0.0126734018.
- Q3: NO under the frozen 0.020 threshold; gain is 0.0154254198.
- Q4: C1: gap 0.0412521124, generic-real 0.0642343044; C2: gap 0.0400157690, generic-real 0.0671486378; C3: gap 0.0583660364, generic-real 0.0987090588; C4: gap 0.0556101084, generic-real 0.0843887091
- Q5: Recovery fractions by C1-C4 are 0.072999, 0.071229, 0.044426, 0.034046.
- Q6: Destination-specific routing differs within C2, C3, C4.
- Q7: YES; final B1 routing vectors differ across C1-C4.
- Q8: Self transfer ran for C1; see self-transfer table.

## Next-experiment decisions

- A: NO cumulative depth passed the frozen matched-feedback rule.
- B: C1 is strongest by recovery fraction among eligible configurations.
- C: A separately preregistered self-training test is supported most strongly at C1; do not launch here.
- D: NO; matched direct-reader evidence is insufficient for writers.
- E: If writers are later authorized, alternate reader/writer phases rather than co-training; do not launch here.
- F: A separate iterative-loop protocol is supportable only for a successful self-transfer configuration; do not launch here.
- G: YES; keep temporal credit limited to one token when writers are eventually introduced.

## Integrity and stopping

All frozen audit checks: **PASS**.

| Audit check | Result |
|---|---|
| 2c1_frozen_tag_exact | PASS |
| all_gradients_finite | PASS |
| all_later_blocks_retain_kv | PASS |
| all_losses_finite | PASS |
| all_optimizer_updates_exactly_48 | PASS |
| all_preflights_passed | PASS |
| all_smokes_passed_and_discarded | PASS |
| base_checkpoint_sha_exact | PASS |
| base_frozen | PASS |
| c1_historical_trajectory_regression | PASS |
| canonical_validation_hash_exact | PASS |
| checkpoint_strict_reload | PASS |
| finalize_added_zero_optimizer_updates | PASS |
| forced_fresh_process_restart | PASS |
| future_causality | PASS |
| generic_calibration_disjoint | PASS |
| generic_calibration_identical | PASS |
| hellaswag_not_run | PASS |
| identical_training_batch_sequence | PASS |
| no_auxiliary_objective | PASS |
| no_bptt | PASS |
| only_intended_low_blocks_masked | PASS |
| reader_destination_mapping_exact | PASS |
| row_isolation | PASS |
| self_state_resume_equivalence | PASS |
| single_implementation_commit | PASS |
| teacher_frozen | PASS |
| trainable_parameter_counts_exact | PASS |
| writers_never_active | PASS |
| zero_gate_equals_cumulative_mask | PASS |

- C1 optimizer updates: 48
- C2 optimizer updates: 48
- C3 optimizer updates: 48
- C4 optimizer updates: 48
- No writers, auxiliary objective, BPTT, reader continuation, iterative loops, additional masks, HellaSwag, or follow-on optimization were run.

# EXPERIMENT 2C2 COMPLETE