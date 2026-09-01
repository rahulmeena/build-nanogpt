# Experiment 2D6 Final Report

## Result

**B6 REPRESENTATION COMPARISON UNRESOLVED**

Evaluation set: **reused sealed matched panel** (2,048 paired sequences; 2,097,152 targets per condition).

- Fixed REAL CE: `3.044323022936`
- New REAL CE: `3.044262807243`
- Fixed − New: `+0.000060215693`; paired 95% CI `[-0.000008212805, +0.000129402616]`
- New penalty (New − Fixed): `-0.000060215693`; paired 95% CI `[-0.000129402616, +0.000008212805]`
- Binding practical margin: `delta_CE = 0.0001`
- Zero-training geometry shock (New geometry − Original): `+0.000309330293`
- Fixed B6-OFF effect (OFF − REAL): `+0.000008137956`; paired 95% CI `[-0.000032899209, +0.000049577754]`
- Persistent state, New − Fixed: `+1,536` bytes physical (`+1,536` logical)
- Median latency, New − Fixed: `-0.548067` s (`-4.683%`)
- Median throughput, New − Fixed: `+4.298` token/s (`+4.913%`)
- Final checkpoint SHA-256: `6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7`
- Audit: `PASS`
- Git branch: `experiment-2d6-b6-w1024-no-b7-recurrence-matched-100m`
- Git tag: `experiment-2d6-b6-w1024-no-b7-recurrence-matched-100m-final`
- Pod status: `STOPPED`; volume retained

## Answers

1. Native B6 W1024 is classified as: **B6 REPRESENTATION COMPARISON UNRESOLVED**.
2. B7→B6 inside mature W512 changes CE by `+0.000008137956` when removed (OFF − REAL); positive means recurrence helps.
3. The approximately equal-memory native architecture is preferable only if the classification and measured speed support it; the measured physical memory delta is `+1,536` bytes.
4. Exact tradeoff: quality `-0.000060215693` CE penalty, memory `+1,536` bytes, throughput `+4.913%`.
5. Fresh-panel confirmation needed: **Yes**.
6. No further training is warranted.

The Fixed stored losses were reused only after sentinel reproduction. This was not a fresh confirmation set.

STOPPED AFTER ONE NEW ARM AT EXACTLY 191 UPDATES / 100,139,008 TARGETS
