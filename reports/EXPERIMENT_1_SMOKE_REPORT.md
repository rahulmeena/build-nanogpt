# Experiment 1 implementation and smoke report

## Scope and outcome

Full Attention Residuals was implemented as the only architecture change. The
standard GPT-2 path remains available and is bit-exact against the frozen
baseline implementation in the regression test. CPU correctness,
initialization equivalence, A100 B64/T1024 fit, one complete production-shape
optimizer update, and a 12-update real-data smoke all passed. No 100M, 250M, 1B,
or 10B experiment was launched.

The authoritative design and source comparison are in
`EXPERIMENT_1_DESIGN.md`. The implementation follows the Full AttnRes equations
in the Kimi Team's Attention Residuals report and was cross-checked against the
official MoonshotAI repository. No material discrepancy was found.

## Git and initialization

- Frozen baseline commit: `a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb`
- Frozen tag: `baseline-gpt2-124m-10b` (pushed before implementation)
- Experiment branch: `experiment-1-full-attnres`
- Canonical step-0 checkpoint on persistent storage:
  `/workspace/build-nanogpt/experiment_artifacts/baseline_init_seed1337.pt`
- Checkpoint SHA256:
  `39de351efe080de4e2409355c572095f17dcbaea76154a2f55e375acfdafc3b6`
- Shared tensors: 149
- Exact matches: 149
- Mismatches: 0
- Maximum absolute difference: 0.0

## Architecture and parameters

There are 24 pre-sublayer routers and one final output router. Every router has
one 768-element zero-initialized pseudo-query and one 768-element RMSNorm scale.
The embedding is `v0`; all 12 attention outputs and 12 MLP outputs are separate
raw values. Keys are destination-specific RMSNorms of those values, and softmax
is over residual depth. The final router mixes all 25 values before the
unchanged `ln_f` and tied LM head.

| Measurement | Value |
|---|---:|
| Standard GPT-2 parameters | 124,475,904 |
| Full AttnRes parameters | 124,514,304 |
| Added parameters | 38,400 |
| Increase | 0.030849% |
| Queries, including final router | 19,200 |
| RMSNorm scales, including final router | 19,200 |
| Final router share of added parameters | 1,536 |
| Other added parameters | 0 |

## Verification

The six-test CPU suite passed:

1. output shape, dtype, and device;
2. exact identity in the one-source case;
3. exactly uniform initial weights from zero queries;
4. softmax normalization over depth only;
5. finite forward/backward and gradient flow to sources, queries,
   normalization, attention, and MLP parameters;
6. standard-mode logits and loss exactly equal to the frozen baseline at
   `rtol=0`, `atol=0`.

The initialization-control report independently instantiated the frozen code at
seed 1337, saved its checkpoint, loaded all shared tensors into Full AttnRes,
and found 149/149 exact tensor matches.

## A100 fit and overhead

Hardware was one NVIDIA A100-SXM4-80GB with CUDA 12.8 and PyTorch 2.8.0. The
production shape was B=64, T=1024, eight microsteps, 524,288 tokens per update,
and BF16 autocast with FP32 parameters and the FP32 residual accumulator.

The Phase-B single-microbatch probe measured:

| Measurement | Standard | Full AttnRes | Ratio/delta |
|---|---:|---:|---:|
| Forward | 383.114 ms | 1,210.384 ms | 3.159x |
| Backward | 287.691 ms | 2,190.214 ms | 7.613x |
| Peak allocated after forward | 47,341.704 MiB | 49,962.226 MiB | +2,620.521 MiB |
| Peak allocated after backward | 59,530.546 MiB | 62,151.068 MiB | +2,620.521 MiB |

The first complete eight-microstep optimizer update measured:

| Measurement | Standard | Full AttnRes |
|---|---:|---:|
| Total update time | 4.285 s | 25.489 s |
| Eight forward passes | 2,179.570 ms | 8,043.508 ms |
| Eight backward passes | 2,011.996 ms | 17,357.410 ms |
| Optimizer kernel | 11.178 ms | 11.892 ms |
| Throughput | 122,348.6 tok/s | 20,569.2 tok/s |
| Peak allocated | 60,014.135 MiB | 62,637.928 MiB |
| Peak reserved | 72,564 MiB | 75,470 MiB |

Across updates 1--11, Full AttnRes stabilized at 24.665 s/update and 21,256.2
tok/s. Relative to the same instrumented standard smoke, this is a 5.756x
slowdown and retains 17.37% of baseline throughput. The Full run's maximum was
63,585.056 MiB allocated and 75,472 MiB reserved. Compared with the paired
standard smoke peak, that is +3,570.921 MiB allocated (+5.95%) and +2,908 MiB
reserved (+4.01%). The historical frozen 10B baseline peak was 60,963.53 MiB,
making the Full run +2,621.53 MiB (+4.30%) over that reference.

The BF16 theoretical maximum raw residual stack is 2.34375 GiB for 25 values at
B64/T1024/C768. This implementation's mixed-dtype stack is approximately 2.4375
GiB because the embedding value is FP32. The Phase-B allocated delta of 2.559
GiB is consistent with that stack plus routing logits and framework overhead.
The dominant runtime overhead is the quadratic-in-depth RMSNorm/score
recomputation in backward, not the AdamW update.

## Twelve-update real-data smoke

The real FineWeb-Edu dataset was reused and verified: one 100,000,000-token
validation shard plus 9,853,989,344 training tokens, 9,953,989,344 total tokens,
all uint16. The run processed 6,291,456 tokens in 12 optimizer updates.

| Step | Train loss | Gradient norm |
|---:|---:|---:|
| 0 | 10.944196 | 14.710123 |
| 1 | 10.894193 | 14.247821 |
| 2 | 10.799818 | 13.896689 |
| 3 | 10.663531 | 12.520129 |
| 4 | 10.524563 | 10.348259 |
| 5 | 10.385511 | 8.746472 |
| 6 | 10.267161 | 7.515247 |
| 7 | 10.157256 | 6.455466 |
| 8 | 10.046498 | 5.577319 |
| 9 | 9.971875 | 4.711303 |
| 10 | 9.886333 | 4.058722 |
| 11 | 9.838446 | 3.548226 |

All losses and gradient norms were finite; no NaN or Inf appeared. The
two-batch validation pipeline returned 9.770629. The 32-example HellaSwag smoke
returned 5/32 (0.15625); this only verifies the pipeline and is far too small to
support a model-quality comparison.

## AttnRes behavior

Initial weights were exactly uniform. Representative initial measurements were:

| Destination | Sources | Initial mean weights | Initial entropy |
|---|---:|---:|---:|
| Block 1 attention | 1 | 1.0 | 0.000000 |
| Block 1 MLP | 2 | 0.5 each | 0.693147 |
| Block 6 attention | 11 | 0.090909 each | 2.397895 |
| Block 12 MLP | 24 | 0.041667 each | 3.178054 |
| Final `ln_f` input | 25 | 0.04 each | 3.218876 |

After the smoke, the distributions had moved slightly but measurably. Block 1
MLP was `[0.499268, 0.500732]` with entropy 0.693144; Block 6 attention weights
ranged from about 0.09029 to 0.09170 with entropy 2.397883; Block 12 MLP weights
ranged from about 0.04120 to 0.04209 with entropy 3.178040; and final-router
weights ranged from about 0.03958 to 0.04036 with entropy 3.218863.

At initialization, 25 query tensors received gradients and 24 were nonzero.
The first one-source router is necessarily constant and therefore has zero query
gradient. RMSNorm gradients are mathematically zero on update 0 because all
queries are zero; from update 1 onward, 24 RMSNorm tensors have nonzero
gradients. At the end, maximum query magnitude was 6.595e-5 and maximum RMSNorm
scale displacement from one was 4.733e-5. This verifies that the learnable
depth weights and key normalizers moved away from initialization.

## Prepared next experiments

- `configs/exp1_100m.json`: 191 updates, 100,139,008 actual tokens, evaluations
  at 10M/25M/50M/100M.
- `configs/exp1_250m.json`: 477 updates, 250,085,376 actual tokens, plus
  evaluations at 150M/200M/250M.
- `scripts/run_exp1_ab.sh`: paired guarded launcher; it requires explicit
  `CONFIRM_EXP1_AB=1`, the clean experiment branch, the frozen tag, the canonical
  initialization SHA, verified A100/dataset reports, and writes reproducibility
  metadata for each arm.
- `scripts/plot_exp1.py`: generates the six requested comparison and depth
  plots.
- `scripts/attnres_ablate.py`: offline source-masking validation utility.

The preserved original 10B learning-rate schedule is the selected default; the
alternative compressed short-run policy and tradeoff are documented in
`EXPERIMENT_1_PROTOCOL.md`.

The prepared 100M and 250M matched A/B runs were not launched.

# EXPERIMENT 1 IMPLEMENTATION READY FOR A/B TESTING
