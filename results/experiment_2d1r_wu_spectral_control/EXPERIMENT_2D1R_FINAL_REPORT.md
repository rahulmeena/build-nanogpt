EXPERIMENT 2D1R CLASSIFICATION:
W_U NORM CONTROL DOES NOT FULLY STABILIZE 2D1

SECONDARY RECURRENCE CLASSIFICATION:
NO RECURRENT UTILITY

# Experiment 2D1R — Terminal Scientific Result

Spectral control successfully carried the exact C954 continuation through all
of Stage C, but the unchanged rho=1 Stage-D transition triggered the original
three-consecutive recurrent-scale hard stop on attempted update
1915. The last persisted
result update is 1914; no failed update was appended.

The recurrent-input RMS on the terminal attempt was
`0.404470711946`
(11.390345× the Stage-A
reference), above the exact hard threshold
`0.355099629611`. Every completed
W_u projection remained within the frozen C954 cap
`1.026231765747`. Therefore this is a valid negative scientific
result, not an integrity failure.

At the last completed Stage-C milestone (C1908), plain and real recurrent
validation CE were `3.0815834309` and
`3.1751997390`. Recurrent gain was
`-0.0936163081` with
0/20 paired wins. The 32-pass Stage-C probe was
OSCILLATORY and remained bounded at
`0.322392284870`.

Final-target controls and incremental validation are not scientifically
applicable because the protocol-mandated hard stop prevented reaching Stage E.
Their required artifacts explicitly record `NOT_RUN_TERMINAL_HARD_FAILURE`.

## Scientific questions

### Q1

Yes for the original Stage-C failure: training completed Stage C through update 1908, then failed at rho=1 in Stage D.

### Q2

Stage-C projected fraction was 100.000000%.

### Q3

Stage-C maximum raw sigma was 1.030534744; maximum pressure was 0.004193.

### Q4

U/ZN remained finite through completed updates; last value was 0.465710313.

### Q5

Stage-C maximum X/E was 9.186046674; Stage D crossed the recurrent hard-failure threshold.

### Q6

See failed_lineage_comparison.json; C1000/C1100 CE remained essentially matched while recurrent RMS was reduced.

### Q7

OSCILLATORY and bounded at C1908; max RMS 0.322392285.

### Q8

No. Stage D reached the three-consecutive scale hard stop on attempted update 1915.

### Q9

Not reached because the Stage-D hard stop is terminal by protocol.

### Q10

C1908 plain-triangle loss: 3.0815834309; no final Stage-E model exists.

### Q11

C1908 real recurrent loss: 3.1751997390; no final Stage-E model exists.

### Q12

C1908 recurrent gain was -0.0936163081 CE.

### Q13

C1908 sequence gap was +0.1929402302, but real recurrence had only 0/20 wins over plain.

### Q14

At C1908, zeroing recurrence increased loss by +6.6099134989, showing dependence without utility.

### Q15

Not run: no final Stage-E model exists after the terminal hard stop.

### Q16

Final-target cache audit is not applicable; preflight cache mechanics passed.

### Q17

Stage E was not reached.

### Q18

The cap was continuously binding: projection fractions were {'C': 1.0, 'D': 1.0, 'E': 0.0}.

### Q19

ADD SECONDARY RECURRENT STABILIZATION

## Integrity

Terminal scientific audit before the Git/persistence seal:
**PASS**.

Next recommendation: **ADD SECONDARY RECURRENT STABILIZATION**. It was not executed.

# EXPERIMENT 2D1R COMPLETE
