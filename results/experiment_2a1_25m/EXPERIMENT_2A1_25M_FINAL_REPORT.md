# Experiment 2A1 — 25M Continuation Final Report

## Outcome

Experiment 2A1 continued the verified Experiment 2A0 feedback state from completed update 10 to completed update 48, adding exactly 38 optimizer updates and 19,922,944 student tokens. Training stopped at 25,165,824 cumulative Experiment-2 tokens. No follow-on experiment was launched.

The aligned-history advantage over shuffled feedback increased monotonically at every predeclared milestone:

| Cumulative tokens | Real validation | Shuffled validation | Mean real − shuffled | Total recovery / damage | Specific recovery / damage |
|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 5.9533051014 | 5.9617962360 | -0.0084911346 | 1.0749% | 0.4481% |
| 10,485,760 | 5.9229358435 | 5.9391546011 | -0.0162187576 | 2.6775% | 0.8559% |
| 15,204,352 | 5.8944202185 | 5.9179884195 | -0.0235682011 | 4.1822% | 1.2437% |
| 25,165,824 | 5.8353391409 | 5.8765912533 | -0.0412521124 | 7.2999% | 2.1769% |

Negative `real − shuffled` means the correctly aligned higher-layer history performed better. Equivalently, the positive sequence-specific recovery `shuffled − real` grew from 0.008491 at 5M to 0.041252 at 25M. The fixed trajectory rule therefore classifies the result as:

**MEMORY SIGNAL STRENGTHENING**

## Protocol

- Hardware: 1× NVIDIA A100-SXM4-80GB.
- Source checkpoint: Experiment 2A0 completed update 10, SHA-256 `cf68b9765072e2403c16e935ba02e92f826d48600953f904e11f2bd4d266638e`.
- Immutable Experiment 1B parent SHA-256: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`.
- Code: branch `experiment-2a1-25m-continuation`, commit `529ca93b0a77d91e6aba76f9863f6d7108c6605f`.
- Geometry: B=64, T=1024, four restored replay ranks, gradient accumulation 2, eight serialized one-GPU microbatches per update.
- Global batch: 524,288 student tokens/update.
- Updates: cumulative Experiment-2 updates 11–48; metrics use zero-based indices 10–47.
- Original LR schedule: continued at global schedule steps 964–1001 without reset or rescaling.
- Trainable state: exactly 1,537 top-down parameters (query 768, RMSNorm scale 768, scalar gate 1).
- Frozen state: teacher and all existing Full-AttnRes/GPT-2 student parameters.
- Dataset manifest SHA-256: `be14a17c21682a018aef68ce02847cced77e921374c01f806deccfba72870f54`; live shards verified.
- Canonical validation-prefix hash: `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

Before training, the optimizer-free preflight restored the source model, feedback-only AdamW state, Python/NumPy/Torch CPU/Torch CUDA RNG, and all four replay-loader states exactly. Its first new global-batch hash was `01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a`, and it reproduced the verified update-10 real and shuffled losses exactly. The preflight executed zero optimizer steps.

## Final control matrix

| Control | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked Layer-1 history, no feedback | 5.9736744881 |
| Real aligned teacher feedback | 5.8353391409 |
| Shuffled teacher feedback | 5.8765912533 |
| Trained router, gate forced to zero | 5.9736744881 |

Derived quantities:

- Damage: `masked − full = 1.8950200796`.
- Total recovery: `masked − real = 0.1383353472`, or 7.2999% of the damage.
- Sequence-specific recovery: `shuffled − real = 0.0412521124`, or 2.1769% of the damage.
- Sequence-specific recovery accounts for 29.8204% of the observed total recovery.
- Shuffled feedback itself recovers 5.1231% of the damage, so most observed recovery remains generic compensation rather than sequence alignment.
- Gate-zero equals masked/no-feedback exactly on every validation batch.

The real-feedback path beats shuffled feedback on all 20 canonical validation batches at every milestone. The requested paired summaries are:

| Tokens | Mean real − shuffled | Median real − shuffled | Real wins | Shuffled wins | Ties |
|---:|---:|---:|---:|---:|---:|
| 5,242,880 | -0.0084911346 | -0.0084269047 | 20 | 0 | 0 |
| 10,485,760 | -0.0162187576 | -0.0161867142 | 20 | 0 | 0 |
| 15,204,352 | -0.0235682011 | -0.0236744881 | 20 | 0 | 0 |
| 25,165,824 | -0.0412521124 | -0.0415372849 | 20 | 0 | 0 |

The raw per-batch losses and descriptive paired intervals are preserved in the milestone evaluation JSON files. They are descriptive fixed-prefix measurements, not a claim of IID statistical significance.

## Router evolution

| Tokens | Gate | tanh(gate) | Query norm | RMSNorm displacement | Tokenwise entropy | v16 | v17 | v20 | v24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 0.00599764 | 0.00599757 | 0.124895 | 0.090100 | 1.289204 | 0.271206 | 0.251649 | 0.219422 | 0.257723 |
| 10,485,760 | 0.01199502 | 0.01199445 | 0.269662 | 0.226218 | 1.037263 | 0.354613 | 0.159561 | 0.198184 | 0.287643 |
| 15,204,352 | 0.01739252 | 0.01739077 | 0.391837 | 0.351041 | 0.850197 | 0.411335 | 0.118092 | 0.179551 | 0.291022 |
| 25,165,824 | 0.02878677 | 0.02877882 | 0.605271 | 0.581372 | 0.737608 | 0.406141 | 0.165030 | 0.169009 | 0.259819 |

The feedback coefficient, query norm, and RMSNorm displacement all increased, while mean tokenwise routing entropy fell from 1.2892 to 0.7376. Routing therefore became more selective, primarily toward v16, without becoming one-hot. Mean routing weight is not by itself causal importance.

## Final leave-one-source-out controls

Each row removes one source and renormalizes the remaining routing mixture. The delta is `ablated loss − real-feedback loss`; a positive value means removal hurt this fixed evaluation.

| Removed source | Ablated validation loss | Delta vs real | Descriptive paired 95% interval | Batches with positive delta |
|---|---:|---:|---:|---:|
| v16 — Block 8 MLP | 5.8756693840 | +0.0403302431 | [+0.0391984939, +0.0414619923] | 20/20 |
| v17 — Block 9 Attention | 5.8449754715 | +0.0096363306 | [+0.0085929073, +0.0106797540] | 20/20 |
| v20 — Block 10 MLP | 5.8494897842 | +0.0141506433 | [+0.0130700035, +0.0152312832] | 20/20 |
| v24 — Block 12 MLP | 5.8499561548 | +0.0146170139 | [+0.0127801818, +0.0164538460] | 20/20 |

All four sources contribute positively in the final renormalized-mixture ablation. v16 has the largest measured effect. These are leave-one-out mixture effects, not isolated-source effects.

## Checkpoint, replay, and safety audit

| Completed update | Cumulative tokens | Checkpoint SHA-256 | Verified next global batch hash |
|---:|---:|---|---|
| 20 | 10,485,760 | `028a3a4126ba71090900c2d5ec2a709e631e9d1d8cf60e91cd6a83fed9fd24de` | `921abc217182d1f7596f26ac421e0ba317b0c9b8b510a3baa03bf26c604d4471` |
| 29 | 15,204,352 | `f47b23cb8b2da1a6a9b2db7eabdb550379731d22759b55694be29c63ff6c0117` | `51c1a47728a9293c62481fdd1e5b4f8fe92a5eb5a98494e3a886de29dfa86674` |
| 48 | 25,165,824 | `d821b48a796b12bb489f5bc9bc1791c475c09a50de7d5b47c4a36cf766643ec2` | `1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5` |

- All 38 training global-batch hashes match the independently prepared replay oracle.
- Checkpoint payload bytes match their SHA-256 sidecars.
- Model, feedback-only optimizer, loader, and RNG reload checks pass.
- Fresh-object restart checks at updates 20 and 29 pass exactly.
- The 199 frozen parent tensors remain bit-identical; frozen-base SHA-256 is `1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd`.
- Every training row records finite loss, gradients, optimizer moments, routing values, and trainable parameters.
- Teacher stayed in `eval()` under `no_grad()`; no teacher or frozen-student parameter received a gradient.
- Teacher-memory and end-to-end future-token leakage tests pass bit-exactly at updates 10, 20, 29, and 48, with maximum prefix-logit difference 0.0.
- No optimizer update beyond completed update 48 exists.

## Resources

- Full result-bearing invocation wall time: 2,113.15 seconds (35 minutes 13 seconds).
- Summed optimizer-update wall time: 1,221.59 seconds.
- Summed evaluation wall time: 389.72 seconds.
- Peak allocated VRAM: 50,629.34 MiB.
- Peak reserved VRAM: 74,778 MiB.

## Interpretation and scope

The result supports the experiment's narrow causal claim: with Layer-1 historical attention removed, correctly aligned detached higher-layer past state increasingly outperforms both no feedback and sequence-shuffled feedback as the small top-down module trains. The growing aligned-versus-shuffled gap and exact zero-gate control show that the learned pathway uses sequence-specific historical information in addition to generic compensation. At the final milestone, sequence alignment accounts for 29.8204% of the observed recovery; the larger remainder is not sequence-specific under this shuffle control.

This remains a teacher-assisted diagnostic using detached full-context higher-layer states. It does not yet demonstrate a deployable self-recurrent model, and the repeatedly measured fixed validation prefix should not be treated as an IID significance sample.

**MEMORY SIGNAL STRENGTHENING**

**EXPERIMENT 2A1 25M COMPLETE**
