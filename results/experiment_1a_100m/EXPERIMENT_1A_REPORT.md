# Experiment 1A — Matched 100M-token A/B

## Outcome

**APPROXIMATELY NEUTRAL**

At 100,139,008 matched training tokens, Full Attention Residuals finished with validation loss 6.412367 versus 6.421217 for standard GPT-2, a favorable delta of -0.008851 (AttnRes - Standard, about -0.138%). The milestone trajectory was not consistently favorable: AttnRes was slightly worse at approximately 10M, 25M, and 50M tokens, then slightly better at 75M and 100M. This is a single-seed exploratory run, so the small late advantage is not evidence of statistical significance.

No follow-on training was launched.

## Protocol and integrity

- Experiment implementation commit: `2bdb3c4752d05174218c951d85cb3d32cec353d3`
- Exact Experiment 1A harness commit: `29cd1a74e48517da25fcaa561f6d00212c92b5d1`
- Baseline commit/tag: `a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb` / `baseline-gpt2-124m-10b`
- Canonical initialization SHA256: `39de351efe080de4e2409355c572095f17dcbaea76154a2f55e375acfdafc3b6`
- Hardware: one NVIDIA A100-SXM4-80GB; runs executed sequentially, Standard then Full AttnRes
- Each arm: 191 optimizer updates, 524,288 tokens/update, 100,139,008 processed tokens, BF16, seed 1337
- Batch geometry: B=64, T=1024, gradient accumulation=8
- Original 10B schedule preserved: max LR 6e-4, min LR 6e-5, 715 warmup steps, 19,073-step schedule horizon. All 191 updates therefore occurred during the original warmup.
- Shared initialization: 149/149 tensors matched exactly in both arms; mismatch count 0; maximum absolute difference 0.0
- AttnRes-only initialization: 25 query tensors exactly zero and 25 RMSNorm tensors exactly one
- Data order: same shard (`edu_fineweb10B/edufineweb_train_000001.npy`), shard index 0, position 0
- First-eight-microbatch combined SHA256 in both arms: `3052dfed888d246de05607fa77df2c3d08bb4c1c5ec654c9e217d27371f74d1e`
- Every arm contains 191 train records, six validation records, six complete 10,042-example HellaSwag records, and six milestone checkpoints. All recorded numerical metrics are finite.

## Final metrics

| Metric | Standard GPT-2 | Full AttnRes |
|---|---:|---:|
| Final step | 190 | 190 |
| Processed tokens | 100,139,008 | 100,139,008 |
| Train loss | 6.462361 | 6.456120 |
| Validation loss | 6.421217 | 6.412367 |
| HellaSwag normalized accuracy | 0.245967 (2470/10042) | 0.244174 (2452/10042) |
| Final gradient norm | 0.829626 | 0.646369 |
| Mean gradient norm | 1.806523 | 1.778253 |
| Total runtime | 971.334 s (16.19 min) | 9029.546 s (150.49 min) |
| Training-update time | 593.884 s | 4710.861 s |
| Mean tokens/s | 168,645.5 | 21,257.1 |
| Peak allocated VRAM | 60,962.4 MiB | 63,585.6 MiB |
| Peak reserved VRAM | 72,566 MiB | 75,492 MiB |
| Parameters | 124,475,904 | 124,514,304 |

The final training-loss delta was -0.006241. Final HellaSwag differed by -0.001792 in normalized accuracy (-0.179 percentage points, 18 examples), with both arms near chance as expected at this token budget. Gradient norms remained finite and similar in aggregate; no OOM, NaN, Inf, or CUDA failure occurred.

## Matched validation comparison

Delta is Full AttnRes minus Standard, so a negative value favors Full AttnRes.

| Actual tokens | Standard val | AttnRes val | Delta |
|---:|---:|---:|---:|
| 10,485,760 | 9.541013 | 9.549112 | +0.008098 |
| 25,690,112 | 8.604507 | 8.605434 | +0.000926 |
| 50,331,648 | 7.249641 | 7.254308 | +0.004666 |
| 75,497,472 | 6.709752 | 6.704704 | -0.005048 |
| 100,139,008 | 6.421217 | 6.412367 | -0.008851 |

For completeness, after the first optimizer update (524,288 tokens), Standard was 10.900392 and AttnRes was 10.892006 (delta -0.008385).

## Routing behavior

At initialization, all AttnRes queries were exactly zero, producing uniform routing with entropy `ln(number of available sources)`. The first recorded milestone, after one optimizer update, remained numerically almost uniform:

| Representative destination | Initial uniform entropy | First recorded entropy | Final entropy | Strongest final source depths (weight) |
|---|---:|---:|---:|---|
| Block 1 MLP (2 sources) | 0.693147 | 0.693147 | 0.677353 | 0 (0.5221), 1 (0.4779) |
| Block 6 attention (11 sources) | 2.397895 | 2.397896 | 2.396672 | 2 (0.0952), 1 (0.0944), 9 (0.0934), 4 (0.0932), 7 (0.0916) |
| Block 12 MLP (24 sources) | 3.178054 | 3.178054 | 3.120560 | 1 (0.0611), 4 (0.0587), 2 (0.0587), 9 (0.0577), 11 (0.0531) |
| Final norm input (25 sources) | 3.218876 | 3.218874 | 3.188244 | 14 (0.0524), 16 (0.0512), 18 (0.0507), 12 (0.0501), 6 (0.0489) |

Maximum/median query norm evolved as follows:

| Tokens | Maximum | Median |
|---:|---:|---:|
| Initialization | 0 | 0 |
| 524,288 | 0.000023 | 0.000023 |
| 10,485,760 | 0.003406 | 0.002730 |
| 25,690,112 | 0.015856 | 0.009914 |
| 50,331,648 | 0.048545 | 0.023803 |
| 75,497,472 | 0.071972 | 0.036162 |
| 100,139,008 | 0.077301 | 0.052838 |

At 100M tokens, maximum RMSNorm L2 displacement was 0.074283, median displacement was 0.046407, and maximum absolute displacement was 0.008673. Thus Full AttnRes was learning nonuniform depth selection, most visibly in later destinations, while retaining distributed rather than collapsed routing.

## Efficiency

- Total-runtime ratio (Full/Standard): **9.296x**
- Training-update-time ratio (Full/Standard): **7.932x**
- Throughput ratio (Full/Standard): **0.1260**; Full delivered 12.60% of Standard throughput, a **7.934x slowdown**
- Peak allocated VRAM increase: **2,623.2 MiB (4.30%)**
- Peak reserved VRAM increase: **2,926 MiB (4.03%)**
- Parameter increase: **38,400 (0.0308%)**

Validation loss versus wall-clock therefore strongly favors Standard under the current unoptimized implementation: the final evaluations occurred at 969.93 seconds for Standard and 9028.40 seconds for Full AttnRes.

## Interpretation

**APPROXIMATELY NEUTRAL.** Full AttnRes ended 0.008851 lower in validation loss and its final two milestones favored AttnRes, which is directionally encouraging. However, the absolute difference is small, three earlier requested milestones favored Standard, HellaSwag did not improve, and only one seed was run. The clean conclusion from this experiment is that Full AttnRes learned meaningful, nonuniform depth routing without destabilizing training, but did not demonstrate a clear learning-efficiency advantage at 100M tokens. Its current compute penalty is large and must not be obscured.

# EXPERIMENT 1A 100M A/B COMPLETE
