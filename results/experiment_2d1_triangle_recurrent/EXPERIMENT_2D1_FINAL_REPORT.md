# Experiment 2D1 terminal report

EXPERIMENT 2D1 FINAL CLASSIFICATION:
EXPERIMENT 2D1 UNSTABLE

SECONDARY RECURRENCE CLASSIFICATION:
RECURRENT INPUT MAGNITUDE IS UNSTABLE

The frozen result run recorded 1,159 complete updates / 
607,649,792 targets. Update 1160 then triggered the
preregistered recurrent-state explosion hard stop: recurrent-input RMS
0.3988617659 exceeded the frozen 10x
threshold 0.3550996296 for a third consecutive update.

This was not a NaN/Inf or top-state explosion. The terminal top-state RMS was
2.3238081932, every logged loss/gradient/
parameter/optimizer check was finite, and the latest verified recovery checkpoint is update
1100 (`6cca94e75ac4802f92df8c1e18d611eb875f42d4312146bb09cb43dfe6d67ad6`).

An isolated retry from that checkpoint reproduced all substantive metrics exactly through
update 1159 and triggered the same hard stop at update 1160. The frozen protocol therefore
has deterministic recurrent-input magnitude instability and must not be continued or silently
modified.

At the last scientific milestone (Stage B end, update 954), plain validation CE was
3.0735813706 and recurrent CE was
3.1693472274, a recurrent gain of
-0.0957658568 with 0/20 paired wins.

Recommended next experiment: **REDUCE OR NORM-CAP THE RECURRENT FUSION BEFORE RETRAINING**.
