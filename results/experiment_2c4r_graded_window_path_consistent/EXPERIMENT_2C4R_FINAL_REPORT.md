# Experiment 2C4R — Path-Consistent Graded-KV-Window Self-Recurrence Rerun

## Opening result

Path-consistent S0 regression: **PASS** at absolute tolerance `5e-06`.

| Schedule | B1-B4 windows | No feedback | Teacher real | Teacher shuffled | Teacher gap | Self real | Self recovery | Self shuffled-real gap | Real-vs-shuffled wins | Extra-reader gain | Mean source cosine | Mean source RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1/1/1/1 | 7.2173504099 | 6.8263555762 | 6.9925456587 | +0.1661900824 | 7.3266605202 | -0.1093101103 | -0.0323840480 | 2/20 | +0.0534682458 | 0.29127653 | 0.25618213 |
| S1 | 1/2/4/8 | 6.9123014597 | 6.5328185435 | 6.7063873779 | +0.1735688344 | 6.9284622064 | -0.0161607467 | -0.0527642472 | 1/20 | +0.0486954872 | 0.36862073 | 0.24109655 |
| S2 | 1/4/16/64 | 6.5146212910 | 6.1350264967 | 6.3166498995 | +0.1816234028 | 6.4535695963 | +0.0610516947 | -0.0698948368 | 0/20 | +0.0479228892 | 0.47203644 | 0.21978590 |
| S3 | 1/8/64/256 | 6.1869757262 | 5.8726367776 | 6.0502017283 | +0.1775649507 | 6.1455257057 | +0.0414500205 | -0.0782787642 | 0/20 | +0.0305164137 | 0.53343936 | 0.20588795 |

Best schedule by recurrent recovery: **S2**.
Best schedule by KV/recovery tradeoff: **S2**.
Recurrence itself rescued: **YES**.
Aligned sequence memory rescued: **NO**.
Classification: **GRADED WINDOWS RESCUE RECURRENT UTILITY BUT NOT ALIGNED MEMORY**.
Frozen rule: S2 passes the frozen recurrent-utility thresholds, but correct sequence identity is not beneficial.

Local-window gain, teacher-feedback gain, self-feedback gain, aligned-sequence gain, and extra B2-B4 reader gain are reported as separate quantities throughout. Absolute loss improvement from a larger window is not treated as recurrent rescue.

## Primary result table

| Schedule | Windows | No-feedback | Teacher real | Teacher shuffled | Teacher gap | Self real | Self shuffled | Self gap | Self recovery | Window-only gain | Self B1-only | Self matched gain | Self/teacher recovery | Self-vs-no-feedback wins | Real-vs-shuffled wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1/1/1/1 | 7.2173504099 | 6.8263555762 | 6.9925456587 | +0.1661900824 | 7.3266605202 | 7.2942764722 | -0.0323840480 | -0.1093101103 | +0.0000000000 | 7.3801287660 | +0.0534682458 | -0.279569 | 0/20 | 2/20 |
| S1 | 1/2/4/8 | 6.9123014597 | 6.5328185435 | 6.7063873779 | +0.1735688344 | 6.9284622064 | 6.8756979592 | -0.0527642472 | -0.0161607467 | +0.3050489502 | 6.9771576937 | +0.0486954872 | -0.042586 | 8/20 | 1/20 |
| S2 | 1/4/16/64 | 6.5146212910 | 6.1350264967 | 6.3166498995 | +0.1816234028 | 6.4535695963 | 6.3836747594 | -0.0698948368 | +0.0610516947 | +0.7027291189 | 6.5014924855 | +0.0479228892 | +0.160834 | 17/20 | 0/20 |
| S3 | 1/8/64/256 | 6.1869757262 | 5.8726367776 | 6.0502017283 | +0.1775649507 | 6.1455257057 | 6.0672469415 | -0.0782787642 | +0.0414500205 | +1.0303746836 | 6.1760421194 | +0.0305164137 | +0.131864 | 16/20 | 0/20 |

## Critical decomposition

### S0

- Local-window benefit: +0.0000000000
- Teacher-feedback benefit: +0.3909948336
- Self-feedback benefit: -0.1093101103
- Aligned self-sequence benefit: -0.0323840480
- Extra B2-B4 reader benefit: +0.0534682458

### S1

- Local-window benefit: +0.3050489502
- Teacher-feedback benefit: +0.3794829162
- Self-feedback benefit: -0.0161607467
- Aligned self-sequence benefit: -0.0527642472
- Extra B2-B4 reader benefit: +0.0486954872

### S2

- Local-window benefit: +0.7027291189
- Teacher-feedback benefit: +0.3795947943
- Self-feedback benefit: +0.0610516947
- Aligned self-sequence benefit: -0.0698948368
- Extra B2-B4 reader benefit: +0.0479228892

### S3

- Local-window benefit: +1.0303746836
- Teacher-feedback benefit: +0.3143389486
- Self-feedback benefit: +0.0414500205
- Aligned self-sequence benefit: -0.0782787642
- Extra B2-B4 reader benefit: +0.0305164137

## Aggregate teacher↔self drift

| Schedule | Mean source cosine | Mean source RMS diff | Δ cosine vs S0 | Δ RMS vs S0 | Mean source norm ratio | Self recovery | Self gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.29127653 | 0.25618213 | +0.00000000 | +0.00000000 | 0.87746907 | -0.1093101103 | -0.0323840480 |
| S1 | 0.36862073 | 0.24109655 | +0.07734420 | -0.01508557 | 0.86589436 | -0.0161607467 | -0.0527642472 |
| S2 | 0.47203644 | 0.21978590 | +0.18075990 | -0.03639623 | 0.85644605 | +0.0610516947 | -0.0698948368 |
| S3 | 0.53343936 | 0.20588795 | +0.24216283 | -0.05029418 | 0.85269111 | +0.0414500205 | -0.0782787642 |

The full 4 schedules × 4 sources × 7 bins source table is in `source_drift.json`; the receiver table is in `receiver_drift.json`. Teacher similarity is interpreted only as a diagnostic correlated with transfer, not as an intrinsic objective.

## Receiver-state drift

| Schedule | Receiver | Mean teacher/student cosine | RMS difference | Norm ratio |
|---|---|---:|---:|---:|
| S0 | B1 | 0.99999999 | 0.00000000 | 1.00000000 |
| S0 | B2 | 0.47327703 | 0.11527901 | 1.78578855 |
| S0 | B3 | 0.53678150 | 0.12019589 | 1.04767484 |
| S0 | B4 | 0.43264359 | 0.11343087 | 0.89998542 |
| S1 | B1 | 0.99999999 | 0.00000000 | 1.00000000 |
| S1 | B2 | 0.48452974 | 0.11251198 | 1.75981002 |
| S1 | B3 | 0.54324064 | 0.12303650 | 1.11181125 |
| S1 | B4 | 0.51098646 | 0.10765635 | 0.93909190 |
| S2 | B1 | 0.99999999 | 0.00000000 | 1.00000000 |
| S2 | B2 | 0.49598377 | 0.11030071 | 1.74221371 |
| S2 | B3 | 0.55280893 | 0.12366683 | 1.14514602 |
| S2 | B4 | 0.56772552 | 0.10358316 | 0.98021403 |
| S3 | B1 | 0.99999999 | 0.00000000 | 1.00000000 |
| S3 | B2 | 0.49969487 | 0.10967832 | 1.73826341 |
| S3 | B3 | 0.55531463 | 0.12495698 | 1.16996586 |
| S3 | B4 | 0.58777125 | 0.10354183 | 1.02498075 |

## Cache budget and physical storage

| Schedule | B1 | B2 | B3 | B4 | Sum windows | Ratio vs 4×1024 | Max actual historical KV lengths B1-B4 | B5-B12 max |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| S0 | 1 | 1 | 1 | 1 | 4 | 0.000977 | 0/0/0/0 | 1023 |
| S1 | 1 | 2 | 4 | 8 | 15 | 0.003662 | 0/1/3/7 | 1023 |
| S2 | 1 | 4 | 16 | 64 | 85 | 0.020752 | 0/3/15/63 | 1023 |
| S3 | 1 | 8 | 64 | 256 | 329 | 0.080322 | 0/7/63/255 | 1023 |

These are low-block window budgets, not claims about exact end-to-end memory savings; B5-B12 and fixed model memory remain.

## Paired controls and monotonicity

| Schedule | Self-vs-no-feedback wins | Mean recovery |
|---|---:|---:|
| S0 | 0/20 | -0.1093101103 |
| S1 | 8/20 | -0.0161607467 |
| S2 | 17/20 | +0.0610516947 |
| S3 | 16/20 | +0.0414500205 |

| Schedule | Real-vs-shuffled wins | Specific gap |
|---|---:|---:|
| S0 | 2/20 | -0.0323840480 |
| S1 | 1/20 | -0.0527642472 |
| S2 | 0/20 | -0.0698948368 |
| S3 | 0/20 | -0.0782787642 |

B1-only versus all-reader paired results:

- S0: all readers win 20/20 with mean gain +0.0534682458.
- S1: all readers win 20/20 with mean gain +0.0486954872.
- S2: all readers win 20/20 with mean gain +0.0479228892.
- S3: all readers win 20/20 with mean gain +0.0305164137.

- self_recovery: nondecreasing=False, nonincreasing=False.
- self_specific_gap: nondecreasing=False, nonincreasing=True.
- source_RMS_difference: nondecreasing=False, nonincreasing=True.
- source_cosine: nondecreasing=True, nonincreasing=False.

## Reproducibility comparison to invalidated debug run

This diagnostic comparison was produced only after the 2C4R classification was frozen. The old 2C4 values are not scientific baselines and did not influence classification.

| Schedule | Condition | Old invalid debug value | New valid rerun value | Absolute delta |
|---|---|---:|---:|---:|
| S0 | no_feedback | 7.2173504099 | 7.2173504099 | 0.0000000000 |
| S0 | teacher_real | 6.8263555762 | 6.8263555762 | 0.0000000000 |
| S0 | teacher_shuffled | 6.9925456587 | 6.9925456587 | 0.0000000000 |
| S0 | self_B1_only | 7.3801287660 | 7.3801287660 | 0.0000000000 |
| S0 | self_real | 7.3266605202 | 7.3266605202 | 0.0000000000 |
| S0 | self_shuffled | 7.2942764722 | 7.2942764722 | 0.0000000000 |
| S1 | no_feedback | 6.9123014597 | 6.9123014597 | 0.0000000000 |
| S1 | teacher_real | 6.5328185435 | 6.5328185435 | 0.0000000000 |
| S1 | teacher_shuffled | 6.7063873779 | 6.7063873779 | 0.0000000000 |
| S1 | self_B1_only | 6.9771576937 | 6.9771576937 | 0.0000000000 |
| S1 | self_real | 6.9284622064 | 6.9284622064 | 0.0000000000 |
| S1 | self_shuffled | 6.8756979592 | 6.8756979592 | 0.0000000000 |
| S2 | no_feedback | 6.5146212910 | 6.5146212910 | 0.0000000000 |
| S2 | teacher_real | 6.1350264967 | 6.1350264967 | 0.0000000000 |
| S2 | teacher_shuffled | 6.3166498995 | 6.3166498995 | 0.0000000000 |
| S2 | self_B1_only | 6.5014924855 | 6.5014924855 | 0.0000000000 |
| S2 | self_real | 6.4535695963 | 6.4535695963 | 0.0000000000 |
| S2 | self_shuffled | 6.3836747594 | 6.3836747594 | 0.0000000000 |
| S3 | no_feedback | 6.1869757262 | 6.1869757262 | 0.0000000000 |
| S3 | teacher_real | 5.8726367776 | 5.8726367776 | 0.0000000000 |
| S3 | teacher_shuffled | 6.0502017283 | 6.0502017283 | 0.0000000000 |
| S3 | self_B1_only | 6.1760421194 | 6.1760421194 | 0.0000000000 |
| S3 | self_real | 6.1455257057 | 6.1455257057 | 0.0000000000 |
| S3 | self_shuffled | 6.0672469415 | 6.0672469415 | 0.0000000000 |

## Scientific questions

### Q1

YES: S2, S3 have positive same-window self recovery.

### Q2

Correct self sequence identity does not become beneficial for S1-S3.

### Q3

S2 has the highest raw same-window self recovery.

### Q4

S2 has the best self-recovery per low-block KV-window budget among S1-S3.

### Q5

Ordinary local-KV restoration contributes window-only gains of S1=+0.3050489502, S2=+0.7027291189, S3=+1.0303746836; these gains are not credited to recurrence.

### Q6

The extra B2-B4 readers improve B1-only self recurrence in S0, S1, S2, S3.

### Q7

Teacher feedback remains sequence-specific: teacher shuffled-real gaps and paired aligned wins are S0=+0.1661900824 (20/20), S1=+0.1735688344 (20/20), S2=+0.1816234028 (20/20), S3=+0.1775649507 (20/20).

### Q8

YES: source cosine rises and source RMS difference falls monotonically from S0 through S3.

### Q9

Reduced source-state drift is reported alongside recovery; it correlates with the window intervention but is not treated as a causal objective.

### Q10

Reduced drift does not restore correct self-sequence alignment; any positive recovery is generic recurrent utility rather than aligned memory.

### Q11

After separating local-window and recurrent gains, S2 is the best raw recurrence schedule and S2 is the best budget tradeoff; S2/S3 are not credited for their much larger local-KV gains.

### Q12

The rerun supports a material contribution from the abrupt B1-B4 receptive-field cliff only to the extent that widening windows improves same-window recurrence and representation compatibility; it does not by itself prove that geometry is the sole autonomous-loop bottleneck.

## Next-experiment decisions

### Decision A

YES, as a separately preregistered experiment; the graded diagnostic supports testing a full monotonic pyramid.

### Decision B

Use S2 as the seed geometry if a full-pyramid experiment is separately approved.

### Decision C

YES, under S2's frozen geometry, as a separate experiment.

### Decision D

YES; incremental layer-by-layer reader training is the more interpretable design if reader adaptation is approved.

### Decision E

DEFER; all-strictly-higher source expansion would confound the path-consistent geometry result and needs its own protocol.

### Decision F

YES; adapt the existing self readers before changing the source bank because recurrence helps but aligned identity remains negative.

### Decision G

NO; one-step recurrence must first be stable, aligned, and reproduced with any geometry-specific reader adaptation.

### Decision H

YES; writers remain deferred until direct-reader one-step recurrence is established.

### Decision I

YES; keep B1 at W=1 in the first full-pyramid experiment unless a separate protocol explicitly tests opening B1.

## Integrity and provenance

- 2C4 frozen tag: `experiment-2c4-graded-window-diagnostic-unstable-final`
- 2C4 parent commit: `f1f765c946e218ea3f3f54972133c7d29dceefb5`
- 2C3 frozen source tag: `experiment-2c3-cumulative-reader-scaling-100m-final`
- 2C3 frozen source commit: `8b1af7e14d1547417e799ac02fe0d513b0755f6e`
- 2C4R implementation commit: `5bb76b8cc7788718b4e9b72f9433e0eb915c9596`
- 2C4R results commit: `56d614dae37ad34bf973e9485b2d711410d67d6d`
- C4@100M checkpoint SHA-256: `fce81b995543c42821abd080f615bcb5d2f755f113345988aa24d07b265b0447`
- Base checkpoint SHA-256: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`
- Canonical validation SHA-256: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`
- Final audit: PASS; failed checks: []
- Optimizer objects, scheduler objects, GradScaler objects, backward calls, optimizer steps, parameter updates, and training targets were all exactly zero.
- No writers, inner loops, source-bank expansion, reader adaptation, auxiliary loss, BPTT, B5-B12 window changes, or HellaSwag evaluation ran.

Even where drift and recurrence co-vary, the result is mechanistic evidence about the receptive-field cliff, not proof that window geometry is the only cause.

# EXPERIMENT 2C4R COMPLETE