# Experiment 2A0 preflight report

## Status

The frozen, no-optimizer preparation phase is complete and passes. No
Experiment 2A0 optimizer step has been run. Both optimizer-bearing commands are
fail-closed behind `--allow-optimizer-steps`, and the 10-update smoke remains an
explicit approval gate.

## Parent artifact and replay provenance

- Parent model: Experiment 1B Full AttnRes at 500,170,752 processed tokens and
  954 completed updates.
- Parent checkpoint SHA-256:
  `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`.
- Parent code commit:
  `abecd3e91e89e1259f7198d72d15664943ad48bf`.
- Actual Experiment 1B geometry: four ranks, `B=64`, `T=1024`, and gradient
  accumulation 2, for 524,288 tokens per optimizer update. This corrects the
  `B=32`, accumulation-4 geometry stated in the chat handoff; the global batch
  size itself was correct.
- All 100 FineWeb shard hashes match the canonical manifest. The replayed next
  global batch hash exactly matches the hash stored in the parent checkpoint.
- The canonical 20-batch validation prefix is pinned by SHA-256
  `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

## Frozen results

All losses use the Experiment 1B validation protocol: 20 serial batches,
`B=64`, `T=1024`, BF16 autocast, and 1,310,720 target tokens.

| Mode | Validation loss |
| --- | ---: |
| Full-context Experiment 1B | 4.0786544085 |
| Block-1 self-only, no feedback | 5.9734717607 |
| Block-1 self-only, zero-gate teacher feedback | 5.9734717607 |

Removing Block-1 history causes `+1.8948173523` validation loss. The zero-gate
feedback path equals the no-feedback masked path bit-for-bit on every batch.
The unmodified full-context path is also bit-exact against the parent code at
the real checkpoint: maximum logit difference 0 and identical loss.

The production-shape causality probe changed the input suffix beginning at
position 512 and verified that shifted `v16`, `v17`, `v20`, and `v24` memory at
position 512 remained bit-exact. Position-zero memory is exactly zero.

## Implemented contract

- A frozen full-context teacher captures raw detached `v16`, `v17`, `v20`, and
  `v24`, then shifts each source by exactly one token without wraparound.
- Only Block-1 history is removed. Its masked attention computes the equivalent
  current-token V projection; Blocks 2–12 retain normal causal attention.
- The independent depth router has one zero-initialized query, shared RMSNorm
  keys, raw values, depth-only softmax, and a zero-initialized scalar `tanh`
  gate: 1,537 trainable parameters total.
- The Experiment 1B base and teacher are frozen. The smoke asserts the required
  gate-only, then query, then RMSNorm gradient staging and rejects any base or
  teacher gradient.
- Checkpoints include model, fresh feedback-only optimizer, replay loaders, RNG,
  next-batch hash, parent lineage, and configuration. Smoke phase 1 stops after
  publishing update 5; a separate OS process reopens it and runs updates 6–10.
- The 5M-token runner serializes the original four-rank `B=64`, `T=1024`,
  accumulation-2 batches into eight one-GPU microbatches per update, preserving
  data order and the original 10B-token LR schedule position.

FineWeb is packed exactly as in the parent experiment. Memory resets at row
position zero, not at in-row end-of-text tokens, because the parent causal
attention also crosses those packed end-of-text boundaries. There is no BPTT
through teacher memory and no new inference KV-cache implementation.

## Verification completed

- 15/15 Experiment 2A0 architecture, causality, gradient, and router tests pass.
- 6/6 existing Full AttnRes regression tests pass.
- A real-checkpoint CUDA/BF16 `B=64`, `T=1024` teacher/student backward preflight
  completed without an optimizer step, with finite gate-only gradient, no base
  or teacher gradients, and 56,886.6 MiB peak allocated on the A100-80GB.
- Independent reviews found no remaining functional blocker for the gated
  10-update smoke.

## Next approval gate

The next command is the disposable 10-update smoke with `B=2`, `T=1024`, one
microbatch per update (20,480 student tokens total), including a forced restart
after update 5. It must not be launched without explicit approval.
