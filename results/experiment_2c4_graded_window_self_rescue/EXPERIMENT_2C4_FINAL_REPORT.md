# Experiment 2C4 — Zero-Optimizer Graded-KV-Window Rescue Diagnostic

## Opening result

| Schedule | B1-B4 windows | No feedback | Teacher real | Self real | Self recovery | Self shuffled-real gap | Self B1-only | Extra-reader gain | Mean source cosine | Mean source RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1/1/1/1 | 7.2173504099 | 6.8263555762 | 7.3266605202 | -0.1093101103 | -0.0323840480 | 7.3801287660 | +0.0534682458 | 0.29127653 | 0.25618213 |
| S1 | 1/2/4/8 | 6.9123014597 | 6.5328185435 | 6.9284622064 | -0.0161607467 | -0.0527642472 | 6.9771576937 | +0.0486954872 | 0.36862073 | 0.24109655 |
| S2 | 1/4/16/64 | 6.5146212910 | 6.1350264967 | 6.4535695963 | +0.0610516947 | -0.0698948368 | 6.5014924855 | +0.0479228892 | 0.47203644 | 0.21978590 |
| S3 | 1/8/64/256 | 6.1869757262 | 5.8726367776 | 6.1455257057 | +0.0414500205 | -0.0782787642 | 6.1760421194 | +0.0305164137 | 0.53343936 | 0.20588795 |

Debug-only numerical maximum by recurrent recovery: **S2**.
Debug-only numerical maximum by KV/recovery tradeoff: **S2**.
Classification: **GRADED-WINDOW DIAGNOSTIC UNSTABLE**.
Frozen rule: At least one frozen integrity/S0 regression check failed.

**Hard stop:** S0 failed its preregistered regression oracle. Consequently, S1-S3 are retained only as debugging observations and are not valid scientific results. No scientific classification or follow-on decision may be based on their apparent trends.

## S0 regression diagnosis

- The required S0 no-feedback oracle, `7.2172823668`, is the 2C3 full-sequence generic-control masked loss (`7.217282366752625`).
- The observed S0 no-feedback loss, `7.217350409878418`, exactly reproduces the separately saved 2C3 incremental self-transfer zero-feedback loss (`7.217350409878418`). Its delta from the frozen oracle is `+0.000068043126`, above the fixed `0.000005` tolerance.
- S0 self-real, self-shuffled, self B1-only, self-specific gap, and extra-reader gain reproduce the 2C3 incremental self-transfer values to numerical precision. The derived self-recovery check fails only because its no-feedback term comes from the mismatched oracle path.
- The required teacher-real and teacher-shuffled oracles likewise come from the 2C3 full-sequence generic-control path. The 2C4 incremental schedule path differs by `+0.000054287636` and `+0.000005656753`, respectively.
- Therefore the frozen S0 oracle combines full-sequence generic-control values with incremental self-transfer values. A same-code rerun would deterministically reproduce the mismatch, so the tolerance was not relaxed and no result-bearing rerun was launched.

Local-window gain, teacher-feedback gain, self-feedback gain, aligned-sequence gain, and extra B2-B4 reader gain are reported as separate quantities throughout. Absolute loss improvement from a larger window is not treated as recurrent rescue.

## Primary result table

| Schedule | Windows | No-feedback | Teacher real | Teacher shuffled | Teacher gap | Self real | Self shuffled | Self gap | Self recovery | Window-only gain | Self B1-only | Self matched gain | Self/teacher recovery | Self real wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1/1/1/1 | 7.2173504099 | 6.8263555762 | 6.9925456587 | +0.1661900824 | 7.3266605202 | 7.2942764722 | -0.0323840480 | -0.1093101103 | +0.0000000000 | 7.3801287660 | +0.0534682458 | -0.279569 | 0/20 |
| S1 | 1/2/4/8 | 6.9123014597 | 6.5328185435 | 6.7063873779 | +0.1735688344 | 6.9284622064 | 6.8756979592 | -0.0527642472 | -0.0161607467 | +0.3050489502 | 6.9771576937 | +0.0486954872 | -0.042586 | 8/20 |
| S2 | 1/4/16/64 | 6.5146212910 | 6.1350264967 | 6.3166498995 | +0.1816234028 | 6.4535695963 | 6.3836747594 | -0.0698948368 | +0.0610516947 | +0.7027291189 | 6.5014924855 | +0.0479228892 | +0.160834 | 17/20 |
| S3 | 1/8/64/256 | 6.1869757262 | 5.8726367776 | 6.0502017283 | +0.1775649507 | 6.1455257057 | 6.0672469415 | -0.0782787642 | +0.0414500205 | +1.0303746836 | 6.1760421194 | +0.0305164137 | +0.131864 | 16/20 |

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

| Schedule | Mean receiver cosine | Receiver RMS difference | Receiver norm ratio |
|---|---:|---:|---:|
| S0 | 0.61067553 | 0.10075076 | 1.15499822 |
| S1 | 0.63468921 | 0.09923056 | 1.18542377 |
| S2 | 0.65412955 | 0.09771029 | 1.20687317 |
| S3 | 0.66069519 | 0.09793428 | 1.22962815 |

## Cache budget and physical storage

| Schedule | B1 | B2 | B3 | B4 | Sum windows | Ratio vs 4×1024 | Max actual historical KV lengths B1-B4 | B5-B12 max |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| S0 | 1 | 1 | 1 | 1 | 4 | 0.000977 | 0/0/0/0 | 1023 |
| S1 | 1 | 2 | 4 | 8 | 15 | 0.003662 | 0/1/3/7 | 1023 |
| S2 | 1 | 4 | 16 | 64 | 85 | 0.020752 | 0/3/15/63 | 1023 |
| S3 | 1 | 8 | 64 | 256 | 329 | 0.080322 | 0/7/63/255 | 1023 |

These are low-block window budgets, not claims about exact end-to-end memory savings; B5-B12 and fixed model memory remain.

## Paired controls and monotonicity

- S0: self vs no-feedback 0/20 wins; real vs shuffled 2/20 wins; all readers vs B1-only 20/20 wins.
- S1: self vs no-feedback 8/20 wins; real vs shuffled 1/20 wins; all readers vs B1-only 20/20 wins.
- S2: self vs no-feedback 17/20 wins; real vs shuffled 0/20 wins; all readers vs B1-only 20/20 wins.
- S3: self vs no-feedback 16/20 wins; real vs shuffled 0/20 wins; all readers vs B1-only 20/20 wins.

- self_recovery: nondecreasing=False, nonincreasing=False.
- self_specific_gap: nondecreasing=False, nonincreasing=True.
- source_RMS_difference: nondecreasing=False, nonincreasing=True.
- source_cosine: nondecreasing=True, nonincreasing=False.

## Scientific questions

### Q1

INVALIDATED. S0 failed the hard regression oracle, so the apparent positive same-window recovery in S2/S3 is debug-only and cannot establish rescue.

### Q2

INVALIDATED. The negative aligned-vs-shuffled gaps in S1-S3 are recorded for diagnosis but cannot support a scientific conclusion.

### Q3

INVALIDATED. S2 is the numerical debug-only maximum for raw recovery and KV/recovery tradeoff, not a valid optimum.

### Q4

INVALIDATED for scientific interpretation. Window-only gain and same-window recovery remain separated in the debug tables.

### Q5

INVALIDATED. The extra-reader gains are retained as debug measurements only.

### Q6

INVALIDATED. Teacher controls participate directly in the failed S0 oracle and therefore cannot support compatibility claims.

### Q7

INVALIDATED. The source-drift trend is diagnostic output only and cannot be interpreted scientifically after the hard gate failure.

### Q8

INVALIDATED. Receiver and source drift remain separately recorded for implementation diagnosis only.

### Q9

INVALIDATED. The failed S0 oracle prevents this run from providing evidence for or against the 1→1024 cliff.

### Q10

INVALIDATED. S2 is only the debug-only numerical maximum; no optimal schedule is selected.

## Next-experiment decisions

### Decision A

DEFER. First repair and preregister a path-consistent S0 oracle.

### Decision B

DEFER. Reader retraining is not authorized by an unstable diagnostic.

### Decision C

DEFER. Source-bank expansion is not authorized by an unstable diagnostic.

### Decision D

DEFER. No pyramid-training design is selected by this run.

### Decision E

DEFER. Inner-loop testing remains unauthorized; the prerequisite evidence was not validly evaluated.

### Decision F

DEFER. Writers remain absent and unauthorized.

### Decision G

DEFER. Keep B1 unchanged until a separate protocol supplies a path-consistent S0 oracle and explicitly authorizes another destination-window test.

## Integrity and provenance

- 2C3 frozen tag: `experiment-2c3-cumulative-reader-scaling-100m-final`
- 2C3 parent commit: `8b1af7e14d1547417e799ac02fe0d513b0755f6e`
- 2C4 implementation commit: `bf29d3b1750474c227244738b19adfd939cd3af0`
- 2C4 results commit: `c2d1a73b8a6a43ad93ea2ba7dc036595cdae9459`
- C4@100M checkpoint SHA-256: `fce81b995543c42821abd080f615bcb5d2f755f113345988aa24d07b265b0447`
- Base checkpoint SHA-256: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`
- Canonical validation SHA-256: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`
- Final audit: FAIL; failed checks: ['S0_no_feedback_regression', 'S0_teacher_regression', 'S0_self_regression']
- Optimizer objects, backward calls, optimizer steps, parameter updates, and additional training targets were all exactly zero.
- No writers, inner loops, source-bank expansion, reader adaptation, auxiliary loss, BPTT, B5-B12 window changes, or HellaSwag evaluation ran.

The S1-S3 outputs may be used only to diagnose the oracle/path mismatch. Because the hard S0 gate failed, they are not mechanistic evidence about the receptive-field cliff.

# EXPERIMENT 2C4 COMPLETE