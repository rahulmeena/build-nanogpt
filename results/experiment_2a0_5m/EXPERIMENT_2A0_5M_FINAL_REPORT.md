# Experiment 2A0 5M final report

## Result

# HIGHER-STATE FEEDBACK SIGNAL CONFIRMED

The 5M-token result-bearing run completed exactly 10 global optimizer updates
from the immutable Experiment 1B checkpoint. Real shifted higher-layer feedback
reduced masked-L1 validation loss by `0.0203693867`. Shuffling the feedback
removed `0.0084911346` (41.69%) of that recovery, and forcing the gate to zero
removed all of it.

The recovered fraction is small—`1.074890%` of the Block-1-history damage—but
the control ordering satisfies the experiment's preregistered causal criterion.

## Provenance and protocol

- Parent code commit:
  `abecd3e91e89e1259f7198d72d15664943ad48bf`.
- Experiment code commit used for the run:
  `2eaa26f3a3c1d32c5172a522a2fa96bed4a3b70f`.
- Branch: `experiment-2a0-topdown-l1`; launch worktree clean.
- Parent checkpoint SHA-256:
  `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`.
- Hardware: one NVIDIA A100-SXM4-80GB; PyTorch 2.8.0+cu128; CUDA 12.8.
- Ten updates at 524,288 student tokens/update: 5,242,880 total.
- Each global update replayed four legacy ranks at `B=64`, `T=1024`, gradient
  accumulation 2 as eight ordered one-GPU microbatches.
- The exact original data stream resumed from the parent loader states. All ten
  global-batch hashes match the independently predicted sequence; the first is
  `8f1848a7f86750145743c77e58cb766a0bb5eddd1137aeb6ade62897df112000`.
- All dataset shards passed the canonical manifest check
  (`be14a17c21682a018aef68ce02847cced77e921374c01f806deccfba72870f54`).
- LR schedule positions were exactly 954 through 963 of the original 10B-token
  schedule.

The run loaded the Experiment 1B parent directly and freshly initialized the
feedback query and scalar gate to zero and RMSNorm scale to one. It did not load
the 10-update smoke checkpoint.

## Frozen boundary

Exactly 1,537 feedback parameters were trainable: query 768, shared RMSNorm
scale 768, and scalar gate 1. No pretrained student or teacher tensor acquired
a gradient. Teacher memory was produced in evaluation mode under `no_grad` and
detached before use. Every tracked loss, gradient, trainable tensor, optimizer
moment, routing statistic, and checkpoint tensor was finite.

The frozen student state was bit-exact before and after:

`1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd`.

Gradient staging passed at production geometry: update 1 had a nonzero gate
gradient with zero query/RMSNorm gradients; update 2 had nonzero gate/query
gradients and zero RMSNorm gradient; RMSNorm became nonzero from update 3.

## Canonical validation controls

All modes used the same SHA-pinned 20-batch validation prefix with `B=64`,
`T=1024` (1,310,720 target tokens), BF16 autocast, and hash
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

| Control | Validation loss |
| --- | ---: |
| Full context | 4.0786544085 |
| Masked L1, no feedback | 5.9736744881 |
| Masked L1, real shifted teacher feedback | 5.9533051014 |
| Masked L1, shuffled feedback | 5.9617962360 |
| Masked L1, trained router with gate forced to zero | 5.9736744881 |

```text
damage            = 1.8950200796
recovery          = 0.0203693867
recovery_fraction = 0.0107489028  (1.074890%)
```

The zero-gate loss was bit-identical to masked/no-feedback on every validation
batch. Shuffled feedback retained only `0.0118782520` recovery and was
`0.0084911346` worse than real feedback. Thus content-aligned feedback matters;
the gain is not explained solely by the trained gate or an arbitrary added
activation.

The final masked damage is `0.0002027273` above the preliminary frozen value
quoted before the final self-only implementation was committed. The final
result uses the internally matched current implementation for all five controls
and reproduces the full-context Experiment 1B loss exactly.

## Learned top-down state

```text
gate                    = 0.0059976405
tanh(gate)              = 0.0059975688
query norm              = 0.1248954162
RMSNorm displacement    = 0.0901003852
mean tokenwise entropy  = 1.2892043769 nats
normalized entropy      = 0.9299643806
```

Canonical-validation mean routing weights:

| Source | Mean weight |
| --- | ---: |
| v16 — Block 8 MLP | 0.2712061688 |
| v17 — Block 9 Attention | 0.2516491823 |
| v20 — Block 10 MLP | 0.2194218785 |
| v24 — Block 12 MLP | 0.2577227719 |

## Source ablation

The delta is `loss with source removed - real-feedback loss`; positive means
the source helped in the learned mixture.

| Removed source | Ablated loss | Delta vs real feedback |
| --- | ---: | ---: |
| v16 | 5.9571524143 | +0.0038473129 |
| v17 | 5.9524984360 | -0.0008066654 |
| v20 | 5.9542355061 | +0.0009304047 |
| v24 | 5.9531856775 | -0.0001194239 |

At this short horizon, v16 is the clearest helpful source and v20 is mildly
helpful. Removing v17 or v24 very slightly improves loss; no significance claim
is made for these small single-evaluation deltas.

## Checkpoint and restart verification

At update 5, the checkpoint contained 2,621,440 student tokens and was published
at SHA-256
`07915d3f8bac5d4d300f417d8cfd9266276779d96c00c3c0cabc844023fdb4f4`.
A fresh student, optimizer, and four loaders were instantiated. Strict model,
optimizer, loader, RNG, data-position, and next-batch restoration all passed;
the next-batch hash
`16ae84d368806ac92fcbc0fc7ca2a4b1a47124a4e1f30938d993452a435059a2`
was exactly the batch consumed by update 6 at schedule step 959.

The verified final update-10 checkpoint remains on the pod at:

`/workspace/build-nanogpt-exp2a0/runs/experiment_2a0_5m/checkpoints/checkpoint_updates_000010.pt`

SHA-256:
`cf68b9765072e2403c16e935ba02e92f826d48600953f904e11f2bd4d266638e`.
It contains exactly 10 completed updates, 5,242,880 student tokens, four loader
states, exact RNG state, three finite optimizer states, and next-batch hash
`01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a`.
Strict model/optimizer reload and serialized loader replay pass.

## Resources and stop condition

- Peak allocated VRAM: 50,632.703 MiB (49.446 GiB).
- Peak reserved VRAM: 70,410 MiB (68.760 GiB).
- Ten measured update bodies: 316.755 seconds.
- Complete command, including startup, shard verification, both checkpoints,
  training, and canonical controls/ablations: 612 seconds (10m12s).
- After the final audit: no compute process, 0 MiB GPU memory, 0% utilization.

No longer run, extra destination, source, distillation objective, recurrence, or
unfreezing was launched.

# EXPERIMENT 2A0 5M COMPLETE
