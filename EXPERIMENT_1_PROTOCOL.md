# Experiment 1 Short-Run Protocol

## Learning-rate choice

Two defensible short-run policies are:

1. Preserve the original 10B-token schedule and stop early. At 100M or 250M
   tokens, both models remain in the same 715-update warmup used by the frozen
   baseline. This most faithfully answers which architecture optimizes better at
   equal early token counts under the original recipe, but it does not optimize
   either model for the small budget.
2. Compress or rescale the schedule to the short budget. This produces a
   self-contained short training run that reaches peak and decayed learning
   rates, but it changes the recipe and can confound architecture effects with a
   schedule that was never tuned for either model.

Experiment 1 defaults to policy 1: the original 715-step warmup, 6e-4 maximum,
19,073-step cosine horizon, and 6e-5 floor are preserved, and each matched run is
stopped after 191 (~100M) or 477 (~250M) updates. Any rescaled-schedule result
must use a separately named protocol and may not replace this comparison.

## Matched controls

Both arms load the same `baseline_init_seed1337.pt`, use seed 1337, the same
FineWeb shard ordering, B=64, T=1024, eight microsteps per update, BF16 autocast,
fused AdamW, weight decay 0.1, betas (0.9, 0.95), epsilon 1e-8, and global-norm
clipping at 1.0. The primary metric is validation cross-entropy at equal
processed tokens. Validation loss at equal wall-clock time and HellaSwag are
secondary metrics.

Experiment 1A performs exactly 191 updates and evaluates after steps 0, 19, 48,
95, 143, and 190, at exact processed-token counts 524,288; 10,485,760;
25,690,112; 50,331,648; 75,497,472; and 100,139,008. Every milestone uses the
unchanged 20-batch validation computation and the complete 10,042-example
upstream HellaSwag evaluation. Only cadence differs from the upstream
every-250-update schedule. The prepared 250M configuration is not part of
Experiment 1A and is not launched without separate approval.
