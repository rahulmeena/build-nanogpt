# Experiment 2D5C — Fixed-writer B3/B5 W2 representation pressure

**Classification:** W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL

- Fixed-100M ALL_REAL CE: `3.044323022936`
- C-W2/W2 ALL_REAL CE: `3.095354986487`
- Fixed−C: `-0.051031963552`; paired 95% CI [-0.051787810044, -0.050290981786]
- C B3 OFF gain / SHUFFLED gap: `+0.000184993947` / `+0.000161060065`
- C B5 OFF gain / SHUFFLED gap: `+0.001636330241` / `+0.001852160512`
- C combined OFF gain / SHUFFLED gap: `+0.001866062064` / `+0.002060319627`
- B3 pressure lifts (OFF / SHUFFLED): `+0.000208915347` / `+0.000148274681`
- B5 pressure lifts (OFF / SHUFFLED): `+0.001651879963` / `+0.001771908863`
- Combined pressure lifts (OFF / SHUFFLED): `+0.001859892894` / `+0.001961313454`
- Initial geometry shock: `+1.879586852619`
- Recovery fraction: `+0.971854575551`
- BF16 logical / measured physical reduction: `282,624` / `282,624` bytes
- Final checkpoint: `/workspace/exp2d5c_w2w2_100m/checkpoints/scientific_cumulative_001100480512.pt`
- Final checkpoint SHA-256: `f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe`
- Audit: `PASS`
- Git: `experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m` / `f6e45dcf6bca7f78b03c56b1a695e641c27d1af9` / `experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final`
- RunPod `7kk5yyti00rnrp`: `STOPPED`; volume `yhzyb27fb5` retained

## Scientific interpretation

compare A/B decomposition with modest intermediate W4/W8 windows; execute nothing. No A, B, Fixed, or 250M continuation was executed.

## True-incremental longitudinal core

| C local update | ALL_REAL CE | B3 OFF | B3 SHUFFLED | B5 OFF | B5 SHUFFLED | Combined OFF | Combined SHUFFLED |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.927317558128 | +0.000058250410 | +0.000314177925 | -0.007036929700 | +0.002514301469 | -0.007159265382 | +0.002155889566 |
| 48 | 3.164011790841 | +0.000279178043 | +0.000202525375 | +0.001629632151 | +0.001564639778 | +0.001773604804 | +0.001905304505 |
| 96 | 3.122225893014 | +0.000167085519 | +0.000302361402 | +0.001633121276 | +0.001400641545 | +0.001657943644 | +0.001704998004 |
| 144 | 3.106592168803 | +0.000326397663 | +0.000295952573 | +0.001751312922 | +0.001840067375 | +0.001950196041 | +0.002026195161 |
| 191 | 3.098035731623 | +0.000150473709 | +0.000287152731 | +0.001722253811 | +0.001935884672 | +0.001962010245 | +0.002234896308 |

## Final large-panel paired contrasts

| Contrast | Estimate | Paired 95% CI | Positive sequences | Paired SE |
|---|---:|---:|---:|---:|
| architecture_c_minus_fixed_penalty | +0.051031963552 | [+0.050290981786, +0.051787810044] | 2048 / 2048 | 0.000381703639 |
| architecture_fixed_minus_c | -0.051031963552 | [-0.051787810044, -0.050290981786] | 0 / 2048 | 0.000381703639 |
| b3_off_gain_lift | +0.000208915347 | [+0.000126389546, +0.000291684542] | 1128 / 2048 | 0.000042155983 |
| b3_sequence_gap_lift | +0.000148274681 | [+0.000065757393, +0.000230574496] | 1090 / 2048 | 0.000042051416 |
| b5_off_gain_lift | +0.001651879963 | [+0.001550919876, +0.001753040191] | 1557 / 2048 | 0.000051630946 |
| b5_sequence_gap_lift | +0.001771908863 | [+0.001673931196, +0.001870972187] | 1603 / 2048 | 0.000050444571 |
| c_b3_off_gain | +0.000184993947 | [+0.000119602674, +0.000249774084] | 1137 / 2048 | 0.000033135993 |
| c_b3_sequence_gap | +0.000161060065 | [+0.000096326783, +0.000225035995] | 1105 / 2048 | 0.000032896517 |
| c_b5_off_gain | +0.001636330241 | [+0.001547201585, +0.001724891819] | 1619 / 2048 | 0.000045394412 |
| c_b5_sequence_gap | +0.001852160512 | [+0.001763843695, +0.001940769808] | 1683 / 2048 | 0.000045055116 |
| c_combined_off_gain | +0.001866062064 | [+0.001769525401, +0.001963154336] | 1658 / 2048 | 0.000049346457 |
| c_combined_sequence_gap | +0.002060319627 | [+0.001967473816, +0.002153794933] | 1701 / 2048 | 0.000047778310 |
| combined_off_gain_lift | +0.001859892894 | [+0.001754336163, +0.001965946167] | 1581 / 2048 | 0.000053972206 |
| combined_sequence_gap_lift | +0.001961313454 | [+0.001857341056, +0.002065912818] | 1613 / 2048 | 0.000053283682 |
| f_b3_off_gain | -0.000023921401 | [-0.000079872377, +0.000031808718] | 1001 / 2048 | 0.000028571514 |
| f_b3_sequence_gap | +0.000012785384 | [-0.000041599412, +0.000067331932] | 1044 / 2048 | 0.000027719864 |
| f_b5_off_gain | -0.000015549722 | [-0.000076629165, +0.000044863491] | 1011 / 2048 | 0.000030988026 |
| f_b5_sequence_gap | +0.000080251648 | [+0.000023095814, +0.000137116621] | 1055 / 2048 | 0.000029161916 |
| f_combined_off_gain | +0.000006169170 | [-0.000056450589, +0.000068275572] | 1041 / 2048 | 0.000031878778 |
| f_combined_sequence_gap | +0.000099006173 | [+0.000040726413, +0.000157835316] | 1071 / 2048 | 0.000030004677 |

## Fixed-versus-C pressure-lift table

Positive lift means the intervention cost is larger for C than for the matched Fixed control. Each lift is a paired-sequence difference-in-differences using the shared bootstrap index stream.

| Link | Intervention | Fixed effect | C effect | Fixed-to-C lift | Paired 95% CI of lift |
|---|---|---:|---:|---:|---:|
| B3 | OFF | -0.000023921401 | +0.000184993947 | +0.000208915347 | [+0.000126389546, +0.000291684542] |
| B3 | SHUFFLED | +0.000012785384 | +0.000161060065 | +0.000148274681 | [+0.000065757393, +0.000230574496] |
| B5 | OFF | -0.000015549722 | +0.001636330241 | +0.001651879963 | [+0.001550919876, +0.001753040191] |
| B5 | SHUFFLED | +0.000080251648 | +0.001852160512 | +0.001771908863 | [+0.001673931196, +0.001870972187] |
| Combined | OFF | +0.000006169170 | +0.001866062064 | +0.001859892894 | [+0.001754336163, +0.001965946167] |
| Combined | SHUFFLED | +0.000099006173 | +0.002060319627 | +0.001961313454 | [+0.001857341056, +0.002065912818] |

## Lag, gradient, and contribution diagnostics

All seven requested models passed the 32-sequence diagnostic audit. Full per-head lag bins, opportunity normalization, entropy, source/K/V gradients, actual B8/B10 writer gradients, contribution norms, ratios, and cosines are in `REPRESENTATION_PRESSURE_DIAGNOSTICS.json`.

## Memory accounting

| Quantity | Fixed bytes | C bytes | Reduction bytes |
|---|---:|---:|---:|
| Logical unique BF16 payload | 33,288,192 | 33,005,568 | 282,624 |
| Measured unique storage | 33,288,192 | 33,005,568 | 282,624 |

## Integrity, restart, and replay

The passed terminal audit binds the source/control lineage, sealed final checkpoint, exact 191-row training and replay evidence, optimizer/scheduler continuity, fresh-process update-96 restart, paired evaluations, local backup, scientific-results Git commit/tag, exact stopped pod, and retained persistent volume.

## Recommendation

**DIFFERENT_NEXT_DIAGNOSTIC** — compare A/B decomposition with modest intermediate W4/W8 windows; execute nothing. Execute nothing automatically.

STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS
