# Experiment 2D1B — C1908 Rho × KV-Window Factorial Diagnostic

## Final classifications

**Scale:** RHO=1 DOMINATES STAGE-D SCALE FAILURE

**Predictive:** RHO=1 IS PRIMARY PREDICTIVE BOTTLENECK

This was a frozen-checkpoint, zero-training diagnostic. Optimizer steps, backward calls, parameter updates, training targets, and training-loader advances were all exactly zero.

## T1 — Factorial validation

| Condition | Windows | rho | Plain | Real | Shuffled | Zero | Gain | Sequence gap | Zero dependency |
|---|---|---|---|---|---|---|---|---|---|
| A | C12 | 0.75 | 3.0815834309 | 3.1751997390 | 3.3681399692 | 9.7851132379 | -0.0936163081 | 0.1929402302 | 6.6099134989 |
| B | C12 | 1.00 | 3.0815834309 | 3.4573128504 | 4.1471915961 | 10.0388495819 | -0.3757294195 | 0.6898787457 | 6.5815367315 |
| C | D12 | 0.75 | 3.1001356740 | 3.1871706692 | 3.4008460603 | 9.7202898742 | -0.0870349951 | 0.2136753911 | 6.5331192050 |
| D | D12 | 1.00 | 3.1001356740 | 3.4835763178 | 4.2185583813 | 10.0590158791 | -0.3834406438 | 0.7349820635 | 6.5754395613 |

## T2 — Factor contrasts

| Contrast | Real CE | Recurrent gain | Max recurrent RMS |
|---|---|---|---|
| rho @ C | 0.2821131114 | -0.2821131114 | 0.1066784263 |
| rho @ D | 0.2964056487 | -0.2964056487 | 0.1077708304 |
| window @ .75 | 0.0119709302 | 0.0065813130 | 0.0032681823 |
| window @ 1 | 0.0262634674 | -0.0077112243 | 0.0043605864 |
| interaction | 0.0142925372 | -0.0142925372 | 0.0010924041 |

Plain-window damage is C plain − A plain = +0.0185522432. The additional recurrent window effect is -0.0065813130 at rho=.75 and +0.0077112243 at rho=1.

## T3 — REAL scale decomposition

| Condition | E | Z | ZN | U | G | F | X | XPOS | F/E | X/E |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.035749 | 2.345441 | 0.999511 | 0.445472 | 0.944440 | 0.427312 | 0.321075 | 0.323387 | 11.952999 | 8.981287 |
| B | 0.035749 | 2.345441 | 0.999511 | 0.445472 | 0.944440 | 0.427312 | 0.427313 | 0.429581 | 11.952999 | 11.953040 |
| C | 0.035749 | 2.344564 | 0.999511 | 0.450228 | 0.944440 | 0.431820 | 0.324454 | 0.326765 | 12.079115 | 9.075807 |
| D | 0.035749 | 2.344564 | 0.999511 | 0.450228 | 0.944440 | 0.431820 | 0.431822 | 0.434089 | 12.079115 | 12.079155 |

## T4 — Embedding-content diagnostics

| Condition | cos(F,E) | cos(X,E) | cos(F,X) | cos(ZN,E) |
|---|---|---|---|---|
| A | 0.052985 | 0.081906 | 0.999552 | 0.181900 |
| B | 0.052985 | 0.052985 | 1.000000 | 0.181900 |
| C | 0.052868 | 0.081481 | 0.999561 | 0.180914 |
| D | 0.052868 | 0.052868 | 1.000000 | 0.180914 |

Cosines are descriptive alignment measures, not direct measures of representation quality. Distribution and position-bin statistics are in the machine-readable artifacts.

## T5 — 32-pass stability

| Condition | Max RMS | Pass-32 RMS | Pass-32 CE | State-change | State cosine | Classification |
|---|---|---|---|---|---|---|
| A | 0.322392 | 0.299723 | 3.173010 | 0.082207 | 0.999391 | BOUNDED OSCILLATORY |
| B | 0.429071 | 0.415193 | 3.575699 | 0.199215 | 0.996410 | EXPANSIVE |
| C | 0.325660 | 0.302273 | 3.184309 | 0.085918 | 0.999335 | BOUNDED OSCILLATORY |
| D | 0.433431 | 0.421807 | 3.611316 | 0.229520 | 0.995179 | EXPANSIVE |

The frozen hard threshold is 0.3550996296107769 (10× the Stage-A reference 0.03550996296107769).

## Direct answers Q1–Q19

### Q1

Yes. A's maximum native-oracle absolute delta was 4.267e-11; every required oracle check passed.

### Q2

At C windows, rho=1 changed REAL CE by +0.2821131114, recurrent gain by -0.2821131114, and max repeated RMS by +0.1066784263.

### Q3

At rho=.75, D windows changed REAL CE by +0.0119709302, recurrent gain by +0.0065813130, and max RMS by +0.0032681823.

### Q4

D REAL CE is 3.4835763178, gain -0.3834406438, and max repeated RMS 0.4334312975.

### Q5

The REAL-CE interaction is +0.0142925372; the max-RMS interaction is +0.0010924041.

### Q6

RHO=1 DOMINATES STAGE-D SCALE FAILURE.

### Q7

RHO=1 IS PRIMARY PREDICTIVE BOTTLENECK.

### Q8

Yes; B max RMS is 0.4290707111 versus 0.3550996296.

### Q9

Yes; C max RMS is 0.3256604671.

### Q10

Yes, it crosses the frozen hard threshold; D is EXPANSIVE.

### Q11

A: cos(F,E)=0.052985; B: cos(F,E)=0.052985; C: cos(F,E)=0.052868; D: cos(F,E)=0.052868.

### Q12

A: cos(X,E)=0.081906; B: cos(X,E)=0.052985; C: cos(X,E)=0.081481; D: cos(X,E)=0.052868.

### Q13

Under C windows, rho=1 changes X/E from 8.981287 to 11.953040; F/E changes from 11.952999 to 11.952999.

### Q14

At rho=.75, compression changes F/E by +0.126116.

### Q15

Yes.

### Q16

No; recurrent gain is non-positive in every condition.

### Q17

At rho=1, zero removes both usable recurrence and the direct E path: zero losses are B=10.038850, D=10.059016. This supports pathway dependence but does not isolate semantic content from scale.

### Q18

The factorial should be interpreted jointly: rho=1 removes the explicit E path, while the frozen scale and predictive contrasts quantify whether that architectural restriction is sufficient, window-dependent, or neither. A future fusion change should follow the scale decision tree, not cosine alone.

### Q19

REDESIGN RECURRENT FUSION TO RETAIN AN EXPLICIT CURRENT-TOKEN CONTENT PATH.

## Exactly one next recommendation

**REDESIGN RECURRENT FUSION TO RETAIN AN EXPLICIT CURRENT-TOKEN CONTENT PATH**

No follow-on experiment was started.

## Integrity and provenance

- Frozen postmortem tag: `experiment-2d1r-postmortem-final` → `67f11c1d36e2a1e6aab2543576f0235db5c7025f`.
- C1908: `1d8c5f96cfe5eadc0cdb458d7cdc40b8c8cb78b15ef91a142d1c6044cd6d3864`, 1508096779 bytes, strict reopen passed.
- Canonical validation collection: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.
- Model, W_u, and W_g hashes were unchanged before/after.
- Exactly A/B/C/D were run with the preregistered windows and rho values.
- Fixed row derangement, causal-shift, row-isolation, and parameter-immutability audits passed.
- No HellaSwag, teacher, reconstruction, AttnRes, fusion redesign, optimizer, scheduler, or training operation was used.

## Artifacts

All JSON, this report, and P1–P8 plots are under `results/experiment_2d1b_c1908_rho_window_factorial/`.

# EXPERIMENT 2D1B COMPLETE
