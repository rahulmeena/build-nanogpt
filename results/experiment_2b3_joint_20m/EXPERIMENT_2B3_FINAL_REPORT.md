# Experiment 2B3 — Joint Writer + Reader Co-Adaptation with One-Step Temporal Credit

## Outcome

Joint optimization produced a large recurrent real-loss improvement but did not
preserve sequence-specific dependence on the actual recurrent memory. Final real
loss improved from 5.0959878206 to 4.8141904593, a gain of 0.2817973614, and was
0.0055164814 better than the matched writer-only update-38 result. The shuffled
loss improved almost equally, however: the shuffled-minus-real gap fell from
0.0290133953 at the 15M start to 0.0035031557, below even the matched writer-only
gap of 0.0088150740. Real memory won only 16/20 paired batches.

The cross-swap makes the mechanism clear. Final writers with the starting reader
reached 4.8203086138, accounting for almost all of the gain. The final reader
with starting writers reached only 5.0937164545, and adding the final reader to
the final writers improved 4.8203086138 to 4.8141904593. Reader adaptation was
therefore useful but small, while the learned system continued moving toward a
generic recurrent compensation that did not depend strongly on the correct
sequence-specific memory.

All scientific and systems-integrity checks passed. The run stopped after the
authorized ninth joint update. The expensive conditional reset/ablation suite
was skipped by the frozen rule, and HellaSwag was not run.

## Git

| Field | Value |
|---|---|
| 2B2A frozen tag | `experiment-2b2a-writers-15m-canonical` |
| 2B2A commit | `c47e4c7619d4f7507f1c05bba8557d6c4712ab73` |
| 2B3 branch | `experiment-2b3-joint-writer-reader-1step` |
| Implementation commit | `7a2b8ba7de4294438db4ba63f236e7f4e46452fc` |
| Results commit | `16f74f4296188cfc85b883882054e1961b06f026` |
| Final-report commit | The commit containing this report; recorded in the handoff |

## Source

| Field | Value |
|---|---|
| Canonical 15M checkpoint | `/workspace/runs/experiment_2b2a/checkpoints/checkpoint_updates_000029.pt` |
| Local source backup | `runs/experiment_2b2a/checkpoints/checkpoint_updates_000029.pt` |
| SHA-256 | `86c66343141e24d0beffcf8bc98a558f25c82e1dc05582feade2300d30b2ba84` |
| Starting next-batch SHA-256 | `8b9fe2fa1c2a10ce930caff4d527c48e4f14ab0e1a6f5e4b352e42f61b8b360d` |
| Writer Adam starting step | 29 for all eight writer tensors |
| Reader Adam starting step | 0; fresh state for exactly 1,537 parameters |

The reader weights were preserved exactly; only the reader optimizer state was
fresh. The noncanonical writer-only update-38 checkpoint was not used as the
training source.

## Pre-result hard stops

Two no-result preflight attempts failed and were preserved for audit. The first
used a conventional flattened NCCL sum. Its no-step gradients passed, but tiny
floating-point differences around near-zero fresh-reader gradients were
amplified by the first Adam step: reader update cosine was 0.9999927346 and
relative update L2 was 0.0038127841. A rank-slotted fixed-order sum reduced this
to cosine 0.9999982419 and relative L2 0.0018755254, but still failed because the
new entry point had not enabled the deterministic CUDA settings used by the
frozen lineage.

Both attempts hard-stopped before any result update, and all temporary optimizer
states were discarded. After deterministic PyTorch/cuDNN/cuBLAS execution was
restored, the same rank-slotted protocol passed exactly. The failed artifacts
remain under `failed_preflight_naive_sum/` and
`failed_preflight_missing_determinism/`.

## Distributed equivalence

| Metric | 1 GPU | 4 GPU | Acceptance | Result |
|---|---:|---:|---:|---|
| Global loss | 5.077572417940246 | 5.077572417940246 | absolute delta ≤ 1e-5 | PASS |
| Writer gradient cosine | — | 0.9999999999999925 | ≥ 0.999999 | PASS |
| Reader gradient cosine | — | 1.0000000000000007 | ≥ 0.999999 | PASS |
| Writer gradient relative L2 | — | 0.0 | ≤ 1e-4 | PASS |
| Reader gradient relative L2 | — | 0.0 | ≤ 1e-4 | PASS |
| Writer update cosine / relative L2 | — | 1.0000000000000024 / 0.0 | ≥ 0.999999 / ≤ 1e-4 | PASS |
| Reader update cosine / relative L2 | — | 0.9999999999999997 / 0.0 | ≥ 0.999999 / ≤ 1e-4 | PASS |

The result path used four disjoint rank slots in one 202,756-element FP32
communication buffer, one `all_reduce(SUM)`, then a fixed local rank-order sum
to form the logical 50,689-element joint gradient. Writer and reader gradients
were clipped separately to 1.0 after synchronization. Automatic DDP backward
synchronization was not used.

## Training

| Field | Value |
|---|---:|
| Joint updates | 9 |
| Additional joint-training targets | 4,718,592 |
| Final writer-lineage targets | 19,922,944 |
| Final writer Adam step | 38 for all eight writer tensors |
| Final reader Adam step | 9 for all three reader tensors |
| Update compute wall time | 7,825.43 s (2 h 10 m 25 s) |
| End-to-end two-stage wall time | 7,848.94 s (2 h 10 m 49 s) |
| Mean update time | 869.49 s |
| Effective targets/second | 602.98 |
| Peak allocated VRAM per rank | 40,904.06 MiB (39.95 GiB) |
| Peak reserved VRAM per rank | 79,968 MiB (78.09 GiB) |

Training loss moved from 5.0775724179 on joint update 1 to 4.6903304540 on
joint update 9. Writer pre-clip gradient norms ranged from 1.092647 to 1.155225;
reader norms ranged from 0.968460 to 1.127591. Every writer and reader tensor had
a present, finite, nonzero gradient on every update.

The first four-worker process group exited after update 5. Its checkpoint
strict-reloaded at writer step 34 and reader step 5 with SHA-256
`c2fb2cac39fdbdbbbdd277d9a7db19ada1451ac2790aea59e4a1170714db33ec`.
Four fresh workers then ran updates 6–9.

Final checkpoint:

```text
/workspace/runs/experiment_2b3/checkpoints/checkpoint_joint_updates_000009.pt
SHA-256: 7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b
next batch: 7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7
```

The same checkpoint was copied locally to
`runs/experiment_2b3/checkpoints/checkpoint_joint_updates_000009.pt` and matched
the pod SHA-256.

## Validation

| Condition | Loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked Block-1 / no feedback | 5.9736744881 |
| 15M starting real | 5.0959878206 |
| Final joint real | **4.8141904593** |
| Final joint shuffled | 4.8176936150 |
| Final joint gate zero | 5.9736480713 |
| Teacher sources + final joint writers/reader | 4.9358792543 |

The canonical 20-batch validation SHA-256 was
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.
Full-context, masked, gate-zero, and starting-state controls reproduced their
frozen values exactly. Self-recurrent final memory beat teacher sources passed
through the same final writers and reader by 0.1216887951.

## Matched writer-only counterfactual

| Metric | Writer-only ~20M | Joint ~20M |
|---|---:|---:|
| Real | 4.8197069407 | **4.8141904593** |
| Shuffled | 4.8285220146 | **4.8176936150** |
| Specific gap | **0.0088150740** | 0.0035031557 |

| Derived comparison | Value |
|---|---:|
| Specific-gap gain over writer-only | -0.0053119183 |
| Joint-vs-writer-only real-loss delta | -0.0055164814 |

The joint run consumed exactly the same nine FineWeb global batches as the
writer-only update-29→38 counterfactual. Joint optimization obtained a very small
real-loss advantage but lost even more of the real-vs-shuffled distinction.

## Cross-swap

| Writer | Reader | Loss |
|---|---|---:|
| Start | Start | 5.0959878206 |
| Final | Start | 4.8203086138 |
| Start | Final | 5.0937164545 |
| Final | Final | **4.8141904593** |

Final writers with the starting reader supplied a 0.2756792068 gain. The final
reader alone supplied only 0.0022713661 with starting writers and another
0.0061181545 when paired with final writers. Co-adaptation helped the raw loss,
but writer learning remained the dominant source of improvement.

## Reader evolution

| Metric | Start | Final |
|---|---:|---:|
| Gate | 0.1594133824 | 0.1603136957 |
| Query norm | 1.9570868015 | 1.9564809799 |
| RMSNorm displacement | 1.9478669167 | 1.9470244646 |
| v16 routing | 0.5535031855 | 0.4487045601 |
| v17 routing | 0.0241434742 | 0.0931129873 |
| v20 routing | 0.2180521749 | 0.2776227742 |
| v24 routing | 0.2043011412 | 0.1805595115 |
| Routing entropy | 0.4656425059 | 0.4962184191 |

The reader redistributed mass away from dominant v16 toward v17 and v20 and
became modestly more entropic. Its scalar gate changed only slightly.

## Writer evolution

| Source | Start delta/source | Final delta/source | Final source/adapted cosine |
|---|---:|---:|---:|
| v16 | 0.049116402 | 0.072993432 | 0.996568045 |
| v17 | 0.097499142 | 0.147160412 | 0.988887689 |
| v20 | 0.052848685 | 0.085313903 | 0.995217171 |
| v24 | 0.039606757 | 0.059306742 | 0.997668669 |

All four learned writer residuals grew. v17 remained the largest relative
change, while v16 continued to receive the largest routing weight.

## Sequence specificity

| Metric | Value |
|---|---:|
| 15M starting gap | 0.0290133953 |
| Writer-only matched-token gap | 0.0088150740 |
| Joint gap | **0.0035031557** |
| Real wins | 16/20 |
| Shuffled wins | 4/20 |
| Ties | 0/20 |
| Restores memory specificity vs writer-only | **NO** |

The joint gap fell by 0.0255102396 from the 15M start and by 0.0053119183
relative to the writer-only matched-token result. Joint reader learning did not
arrest the sequence-specific collapse.

## Integrity

| Hard invariant | Result |
|---|---|
| Canonical update-29 source, SHA, model subsets, lineage, and cursor exact | PASS |
| Writer optimizer restored at step 29; reader optimizer fresh at step 0 | PASS |
| Exactly 49,152 writer + 1,537 reader trainable parameters; base frozen | PASS |
| Loss(t+1)→writer(t) present, finite, nonzero | PASS |
| Loss(t+2)→writer(t) absent | PASS |
| Reader and all eight writer gradients present, finite, nonzero | PASS |
| Block-1 historical cache absent; Blocks 2–12 historical K/V detached | PASS |
| Future causality and row isolation bit-exact | PASS |
| Four GPUs, loader-to-rank mapping, replay hashes, and loss scaling exact | PASS |
| Final 1-GPU/4-GPU loss, gradients, Adam moments, and temporary updates within bounds | PASS |
| One joint FP32 all-reduce and separate writer/reader clipping per update | PASS |
| Every per-update model and optimizer state identical across ranks | PASS |
| Atomic update-5 checkpoint, strict reopen, process exit, and fresh-process resume | PASS |
| Exactly nine result updates; writer step 38; reader step 9; no tenth update | PASS |
| Final checkpoint atomic, strict-reopened, and SHA-verified | PASS |
| Final next-batch hash equals matched writer-only cursor | PASS |
| Four loader states and four rank RNG states preserved | PASS |
| Canonical validation hash and all frozen controls exact | PASS |
| Cross-swap component state restored; all losses and cache checks finite/healthy | PASS |
| Teacher training forward calls | 0 — PASS |
| Conditional diagnostics | SKIPPED AS REQUIRED — PASS |
| HellaSwag | NOT RUN — PASS |

## Final classification

**JOINT CO-ADAPTATION IMPROVES GENERIC COMPENSATION ONLY**

## Decisions

### A. Did co-adaptation prevent the sequence-specific collapse?

**No.** The joint gap was 0.0035031557, below both the 15M starting gap and the
matched writer-only gap, with only 16/20 real-memory wins.

### B. Should joint writer+reader training continue for another approximately 5M tokens?

**No.** The small real-loss advantage over writer-only training came with worse
sequence specificity. Another continuation would optimize the wrong signal more
strongly. No continuation was launched.

### C. Should temporal credit remain exactly one token?

**Yes.** Keep the one-token boundary for the next controlled experiment. This
result does not provide evidence that longer credit would restore specificity,
and changing it now would confound the failure mechanism.

### D. Is there enough evidence to unfreeze source-producing Transformer layers?

**No.** The learned writer/reader pair still converged toward generic
compensation. Unfreezing source layers would add substantial capacity before the
objective demonstrates that it rewards correct sequence-specific memory.

### E. Should future recurrent experiments use the validated four-GPU implementation?

**Yes.** With deterministic execution and the rank-slotted fixed-order reduction,
the implementation achieved exact gradient and temporary-update equivalence and
passed every replay, restart, checkpoint, and cross-rank integrity check.

# EXPERIMENT 2B3 COMPLETE
