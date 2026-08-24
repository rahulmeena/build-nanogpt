# Experiment 2D0D — Matched Joint-KV Geometry Evaluation

## Feed-forward geometry conclusion

**EMPIRICAL PROFILE IS BEST FEED-FORWARD GEOMETRY**

The numerically best matched-budget geometry was **EMPIRICAL**. Its advantage over **REVERSE_TRIANGLE** was 0.2109783325, with 20/20 paired-batch wins. The preregistered clear-winner rule was met.

## Joint-interaction conclusion

**STRONG SUPER-ADDITIVE JOINT DAMAGE**

Empirical joint damage was +0.0892095322, versus an exact 2D0C single-layer marginal sum of +0.0528998779. Their difference was +0.0363096543; the joint/marginal ratio was 1.686384. This is a controlled descriptive difference, not a formal interaction decomposition.

## Joint geometry results

| Geometry | ΣW | Fraction full | Val loss | Damage | Argmax agreement | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 12288 | 1.000000 | 3.0750437753 | +0.0000000000 | 1.0000000000 | — |
| EMPIRICAL | 5312 | 0.432292 | 3.1642533075 | +0.0892095322 | 0.8556137085 | 1 |
| TOP_WIDE_TRIANGLE | 5312 | 0.432292 | 5.0145652452 | +1.9395214698 | 0.4320304871 | 4 |
| REVERSE_TRIANGLE | 5312 | 0.432292 | 3.3752316400 | +0.3001878647 | 0.7691207886 | 2 |
| UNIFORM_MATCHED | 5312 | 0.432292 | 4.1171698762 | +1.0421261009 | 0.6573837280 | 3 |

All four candidates use 43.22916667% of the nominal 12×1024 layer-window budget, a 56.77083333% reduction. This is a KV-capacity proxy, not an exact total-memory or optimized serving-throughput measurement.

## Paired comparisons

| A | B | Mean A-B | A wins | B wins | Ties | 95% descriptive interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EMPIRICAL | TOP_WIDE_TRIANGLE | -1.8503119111 | 20 | 0 | 0 | [-1.8653025773, -1.8353212449] |
| EMPIRICAL | REVERSE_TRIANGLE | -0.2109783292 | 20 | 0 | 0 | [-0.2185825806, -0.2033740778] |
| EMPIRICAL | UNIFORM_MATCHED | -0.9529165387 | 20 | 0 | 0 | [-0.9637763869, -0.9420566905] |
| TOP_WIDE_TRIANGLE | REVERSE_TRIANGLE | +1.6393335819 | 0 | 20 | 0 | [+1.6250592885, +1.6536078754] |
| TOP_WIDE_TRIANGLE | UNIFORM_MATCHED | +0.8973953724 | 0 | 20 | 0 | [+0.8891459050, +0.9056448398] |
| REVERSE_TRIANGLE | UNIFORM_MATCHED | -0.7419382095 | 20 | 0 | 0 | [-0.7517050158, -0.7321714032] |

The intervals describe these fixed evaluation batches; they are not formal IID confidence intervals.

## Empirical additive-versus-joint decomposition

| Block | Empirical W | Exact marginal damage |
| --- | ---: | ---: |
| B1 | 1024 | +0.0000000000 |
| B2 | 1024 | +0.0000000000 |
| B3 | 128 | +0.0043317871 |
| B4 | 256 | +0.0045203627 |
| B5 | 256 | +0.0090687642 |
| B6 | 64 | +0.0085915039 |
| B7 | 256 | +0.0050690748 |
| B8 | 1024 | +0.0000000000 |
| B9 | 512 | +0.0030217072 |
| B10 | 256 | +0.0062117539 |
| B11 | 256 | +0.0076859903 |
| B12 | 256 | +0.0043989337 |

- FULL loss: 3.0750437753
- Empirical joint loss: 3.1642533075
- Empirical joint damage: +0.0892095322
- Sum of exact single-layer marginal damages: +0.0528998779
- Interaction (joint minus marginal sum): +0.0363096543
- Joint/marginal ratio: 1.6863844626

Interaction decompositions for Triangle, Reverse, and Uniform are **NOT DIRECTLY AVAILABLE** because their unmeasured widths were not interpolated or evaluated separately.

## Position-dependent damage

| Geometry | Highest-damage common bin | Damage in bin |
| --- | --- | ---: |
| EMPIRICAL | 897-1023 | +0.2523794607 |
| TOP_WIDE_TRIANGLE | 897-1023 | +3.9269249254 |
| REVERSE_TRIANGLE | 897-1023 | +0.8818235723 |
| UNIFORM_MATCHED | 897-1023 | +3.2693538567 |

Machine-readable per-position arrays and every common-bin result are stored in `per_position_loss.json` and `position_bin_loss.json`.

## Argmax retention

| Geometry | Agreement vs FULL |
| --- | ---: |
| EMPIRICAL | 0.8556137085 |
| REVERSE_TRIANGLE | 0.7691207886 |
| UNIFORM_MATCHED | 0.6573837280 |
| TOP_WIDE_TRIANGLE | 0.4320304871 |

**EMPIRICAL** retained full-model argmax predictions best.

## Direct answers to Q1–Q15

1. **Joint damage:** Empirical +0.0892095322; Triangle +1.9395214698; Reverse +0.3001878647; Uniform +1.0421261009.
2. **Lowest loss:** EMPIRICAL, at 3.1642533075.
3. **Clear winner:** Yes under the fixed 0.01 plus 15/20 rule.
4. **Empirical vs Uniform:** EMPIRICAL beat UNIFORM_MATCHED by 0.9529165687 validation loss.
5. **Top-wide Triangle vs Uniform:** UNIFORM_MATCHED beat TOP_WIDE_TRIANGLE by 0.8973953690 validation loss.
6. **Reverse vs Top-wide Triangle:** REVERSE_TRIANGLE beat TOP_WIDE_TRIANGLE by 1.6393336052 validation loss.
7. **Large early vs late windows:** Yes in this ordinary no-recurrence model: the reverse triangle outperformed the top-wide triangle while preserving wider early-layer windows.
8. **Empirical joint vs marginal sum:** +0.0892095322 vs +0.0528998779; interaction +0.0363096543.
9. **Empirical interaction:** STRONG SUPER-ADDITIVE JOINT DAMAGE, ratio 1.686384.
10. **Where damage accumulates most:** EMPIRICAL: 897-1023, TOP_WIDE_TRIANGLE: 897-1023, REVERSE_TRIANGLE: 897-1023, UNIFORM_MATCHED: 897-1023.
11. **Best argmax preservation:** EMPIRICAL, 0.8556137085.
12. **Use the 2D0C marginal profile directly:** No: the simultaneous profile should not be adopted directly because its measured joint damage exceeded its marginal prediction.
13. **Evidence of cross-layer compensation:** Yes: joint damage exceeded the exact marginal sum, consistent with compensation disappearing under simultaneous shortening.
14. **Raw no-recurrence triangle deficit:** +1.9395214698 validation loss, with 0.4320304871 argmax agreement.
15. **Next experiment:** INCREASE / REDISTRIBUTE JOINT KV BUDGET BEFORE RECURRENT TRAINING.

## Geometry manifest and budget

| Layer | Empirical | Triangle | Reverse | Uniform |
| --- | ---: | ---: | ---: | ---: |
| B1 | 1024 | 128 | 1024 | 443 |
| B2 | 1024 | 152 | 844 | 443 |
| B3 | 128 | 184 | 700 | 442 |
| B4 | 256 | 224 | 580 | 443 |
| B5 | 256 | 272 | 480 | 442 |
| B6 | 64 | 328 | 396 | 443 |
| B7 | 256 | 396 | 328 | 443 |
| B8 | 1024 | 480 | 272 | 442 |
| B9 | 512 | 580 | 224 | 443 |
| B10 | 256 | 700 | 184 | 442 |
| B11 | 256 | 844 | 152 | 443 |
| B12 | 256 | 1024 | 128 | 443 |
| **ΣW** | **5312** | **5312** | **5312** | **5312** |

These rows were generated from the frozen `geometry_manifest.json`, not retyped as a separate source of truth.

## Interpretation boundaries

2D0D identifies which geometry the frozen ordinary feed-forward Standard GPT-2 tolerates naturally without recurrent repair. It does not determine the best geometry after recurrent training.

The triangle was evaluated without the high→low recurrent repair mechanism that motivates preserving wider upper layers. A poor 2D0D triangle result measures the burden placed on future recurrence; it does not by itself falsify the recurrent triangle hypothesis.

Likewise, a Reverse win demonstrates natural feed-forward compatibility, not the optimal architecture for recurrent top-down memory. B1 is structurally special because it receives no contextualized Transformer state from a lower block.

The best candidate's absolute joint damage was +0.0892095322. Winning this four-way comparison does not by itself make that degradation deployment-ready.

## Exactly one recommended next experiment

**INCREASE / REDISTRIBUTE JOINT KV BUDGET BEFORE RECURRENT TRAINING**

This recommendation was not executed.

## Integrity and provenance

- Frozen 2D0C tag: `experiment-2d0c-layer-window-sensitivity-map-final` → `752bdc8e0f1a8b0694692ad0b0ae37f4edbeead0`
- 2D0D branch: `experiment-2d0d-matched-joint-kv-geometries`
- Source checkpoint SHA-256: `924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871`
- Validation shard SHA-256: `8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`
- Canonical batch collection SHA-256: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`
- Full-context oracle: 3.0750437753
- Standard GPT-2 only; no Full AttnRes, recurrence, completion, reader, writer, optimizer, backward call, parameter update, or training target
- Four independent single-GPU processes; no DDP or NCCL
- 20 identical B=64, T=1024 batches per candidate; 1,310,720 targets each
- All preflight and pre-final integrity checks: PASS

## Git commits

- Implementation commit: `234563545219a1e6a551dd50a0ea0bdcee828526`
- Results commit: `eb8e3ae9f52f346ad4c5912e15b6019e43e8ffa2`
- Final-report commit: recorded in the synchronized experiment handoff after this report is committed

## Performance

- Four-GPU candidate elapsed wall time: 20.097 seconds
- Masked-evaluator throughput is not optimized deployment throughput.

# EXPERIMENT 2D0D COMPLETE
