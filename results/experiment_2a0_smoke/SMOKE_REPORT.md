# EXPERIMENT 2A0 SMOKE PASSED

The approved smoke stopped after exactly 10 optimizer updates and 20,480
student tokens on one A100-SXM4-80GB. No 5M-token run was launched.

## Frozen and teacher contracts

- Exactly 1,537 parameters were trainable: query 768, shared RMSNorm scale 768,
  and scalar gate 1.
- No pretrained student parameter required gradients or acquired a gradient.
- The pretrained student state hash was bit-exact before and after the smoke:
  `1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd`.
- The teacher remained in `eval()`, all teacher parameters had
  `requires_grad=False`, memory generation ran under `no_grad`, every shifted
  memory bank was detached, no teacher parameter acquired a gradient, and its
  state hash was bit-exact before and after.
- Every loss, new-parameter tensor, gradient, gradient norm, routing weight,
  and optimizer moment checked was finite.

## Gradient staging

- Update 1: finite nonzero gate gradient (`2.75`); query and RMSNorm gradients
  were present, finite, and exactly zero.
- Update 2: gate and query gradients were nonzero; RMSNorm remained exactly
  zero.
- Update 3 onward: RMSNorm gradients were nonzero.

| Update | Loss | Gate | Query norm | Gate grad L2 | Query grad L2 | RMSNorm grad L2 | Peak allocated MiB | Update seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.179938 | 0.000599774 | 0.000000 | 2.750000 | 0.000000 | 0 | 2773.06 | 0.749906 |
| 2 | 6.399370 | 0.001199547 | 0.012203 | 2.281249 | 0.006586 | 0 | 2799.38 | 0.748257 |
| 3 | 5.980854 | 0.001799317 | 0.022230 | 2.343747 | 0.013578 | 5.981e-6 | 2799.38 | 0.605681 |
| 4 | 6.180397 | 0.002399085 | 0.032978 | 2.609367 | 0.019980 | 1.753e-5 | 2799.38 | 0.607916 |
| 5 | 5.854656 | 0.002998850 | 0.043149 | 2.406236 | 0.036384 | 4.770e-5 | 2799.38 | 0.613653 |
| 6 | 6.204764 | 0.003598609 | 0.053076 | 1.554674 | 0.039884 | 6.903e-5 | 2772.57 | 1.108616 |
| 7 | 6.025514 | 0.004198364 | 0.063563 | 1.968724 | 0.048663 | 1.055e-4 | 2802.57 | 0.724461 |
| 8 | 5.905416 | 0.004798116 | 0.074697 | 2.656203 | 0.061469 | 1.612e-4 | 2802.57 | 0.626332 |
| 9 | 6.228633 | 0.005397867 | 0.086168 | 2.843685 | 0.051500 | 1.588e-4 | 2802.57 | 0.613522 |
| 10 | 5.997731 | 0.005997615 | 0.097415 | 2.546801 | 0.060514 | 2.149e-4 | 2802.57 | 0.597931 |

Maximum reserved GPU memory was 3,512 MiB. The ten measured update bodies took
6.996276 seconds in total; forward time was 3.815540 seconds and backward time
was 3.044469 seconds.

## Real top-down routing

Weights are mean tokenwise depth-softmax probabilities for the real shifted
teacher memory used by each update.

| Update | v16 | v17 | v20 | v24 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 2 | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 3 | 0.251463 | 0.251193 | 0.248643 | 0.248701 |
| 4 | 0.256669 | 0.244483 | 0.247070 | 0.251778 |
| 5 | 0.255866 | 0.248427 | 0.239794 | 0.255914 |
| 6 | 0.260601 | 0.251074 | 0.228486 | 0.259839 |
| 7 | 0.261097 | 0.253987 | 0.227148 | 0.257769 |
| 8 | 0.265589 | 0.260872 | 0.220481 | 0.253058 |
| 9 | 0.267542 | 0.256226 | 0.219197 | 0.257036 |
| 10 | 0.275561 | 0.237310 | 0.235937 | 0.251191 |

## Causality

The production CUDA/BF16 leakage probe passed at `B=2`, `T=1024`. Inputs were
changed from position 512 onward; shifted memory at position 512 remained
bit-exact independently for `v16`, `v17`, `v20`, and `v24`. Position-zero
memory remained exactly zero.

## Real process restart

- Phase 1 process PID 8466 ran updates 1–5 and exited successfully.
- Its update-5 checkpoint records 5 completed updates and 10,240 student
  tokens. SHA-256:
  `2ed168c19c3e531f3595350fa3a69b233637adfb0d183805c0ff50aedf47f980`.
- Phase 2 process PID 8520 was launched only after PID 8466 had exited.
- Strict model load, exact optimizer reload, exact RNG reload, exact loader/data
  position reload, checksum sidecar validation, and serialized loader replay
  all passed.
- The restored next-batch hash was
  `83c92aa0dbdce81f6976cf9bc7198244619c43804488695a7418030482eefcb5`;
  update 6 consumed exactly that batch and resumed at global schedule step 959.
- Metrics before resume contained exactly updates 1–5 and required no
  reconciliation truncation.

An uninterrupted 10-update reference was not run because it would have added
10 optimizer updates beyond the explicitly approved budget. The stronger
state-at-boundary checks above were performed instead.

## Final artifact

The verified update-10 checkpoint remains on the pod at:

`/workspace/build-nanogpt-exp2a0/runs/experiment_2a0_smoke_10update/checkpoints/checkpoint_updates_000010.pt`

SHA-256:
`e307005252e3fe7548f4a9881b8a2eb2a107dac974547ff48f1b33f02ecda97b`.
It records exactly 10 completed updates and 20,480 student tokens. Its strict
model, optimizer, RNG, loader, next-batch, finiteness, and sidecar checks pass.
Both training processes have exited and the GPU was idle after the final audit.
