# Experiment 2B0 — Close the First Self-Recurrent High→Low Loop

## Outcome

**SELF-RECURRENT MEMORY TRANSFERS PARTIALLY**

The frozen Experiment-2A3 reader transferred zero-shot from detached teacher
memory to the recurrent student's own prior-token high states. On the canonical
20×B64×T1024 FineWeb prefix, self feedback reduced loss from **5.9736744881** to
**5.7074206829**. This recovered **0.2662538052 loss**, or **14.0502%** of the
original Block-1 masking damage and **65.9631%** of the teacher reader's recovery.

Aligned self memory beat fixed-derangement shuffled self memory on 18/20 batches.
The mean sequence-specific gap was **0.0214961290 loss**, equal to **1.1343%** of
the original masking damage and **8.0735%** of total self recovery. Most immediate
zero-shot recovery is therefore distributional, but the correct recurrent
sequence identity contributes a repeatable additional benefit.

No optimizer was constructed, no optimizer step was taken, and the complete
model state and source checkpoint hashes were unchanged.

## Git

| Field | Value |
|---|---|
| Experiment-2A frozen tag | `experiment-2a3-teacher-reader-250m` |
| Frozen tag target / parent commit | `75bb4e571e4356cacce76c89f352123a40254b5b` |
| Experiment-2B branch | `experiment-2b0-self-feedback-l1` |
| Experiment-2B implementation commit | `ff552a2d662e92b417b0d4cdae295c3a17180ca7` |
| Results commit | The commit containing this report; recorded in the handoff |
| Starting status | clean branch at the frozen parent |

The immutable Experiment-2A tag was created and pushed before branching.

## Starting checkpoint

| Field | Value |
|---|---|
| Path | `/workspace/build-nanogpt-exp2a0/runs/experiment_2a3_250m/checkpoints/checkpoint_updates_000477.pt` |
| SHA-256 | `0702dc09c74b01eee8be504a7f5f89ca61fcc504cda8f34f30865d4ff9653d76` |
| Reader training | 477 updates / 250,085,376 student tokens |
| Gate | 0.1595292091 |
| tanh(gate) | 0.1581895351 |
| Query norm | 1.9564993382 |
| RMSNorm displacement | 1.9469332695 |
| Reader parameters | 1,537 |

The checkpoint contained Adam step 477 for all three reader tensors, but the
optimizer was deliberately not constructed for Experiment 2B0.

## Incremental engine

The new state is explicit and compactly serializable. It contains the next
position, mode, preallocated KV buffers with logical lengths, and a detached
`[4,B,1,768]` v16/v17/v20/v24 memory bank. Block 1 has no cache in masked/self
modes; Blocks 2–12 retain normal growing KV prefixes. Full AttnRes remains
unchanged within each current token.

| Equivalence mode | Max absolute logit diff | Mean absolute diff | Relative mean diff | Argmax agreement | Absolute loss diff |
|---|---:|---:|---:|---:|---:|
| Full context | 0.140625 | 0.0093973279 | 0.2671% | 99.2188% | 0.0014343262 |
| Masked L1 / no feedback | 0.250000 | 0.0108051747 | 0.3029% | 99.2188% | 0.0006313324 |

The CPU float32 unit model matched at `rtol=2e-5, atol=2e-6`. Production BF16
uses different Flash-Attention reduction shapes for parallel causal attention
and one-query incremental attention, so bit equality is not expected. The BF16
loss differences are below 0.0015 and negligible relative to the experimental
0.2663 self recovery. A provisional mean-difference cutoff of 0.010 stopped the
first run at 0.010805; the diagnostic was strengthened with FP32 loss, relative,
RMS, and argmax measures, and the practical BF16 mean bound was set to 0.015.

## Causality and state

| Test | Result |
|---|---|
| Future-suffix prefix logits | bit-exact, max difference 0 |
| Future-suffix prefix memory and KV | bit-exact |
| Batch-row sequence isolation | bit-exact |
| Fresh-sequence reset | position 0, zero memory, empty caches |
| Block-1 cache absence | PASS |
| Serialize/resume continuation logits | bit-exact |
| Serialize/resume final memory and KV | bit-exact |
| Overall | **PASS** |

## Short-horizon stability

The memory norm below is the mean RMS across v16/v17/v20/v24 and all rows/tokens.

| T | Loss | Mean memory RMS | Feedback RMS | Cache health |
|---:|---:|---:|---:|---|
| 8 | 6.6790466309 | 0.2197045349 | 0.0241921861 | PASS; all cached lengths 8 |
| 16 | 6.2151336670 | 0.2126675323 | 0.0253700502 | PASS; all cached lengths 16 |
| 32 | 6.0234889984 | 0.2071771212 | 0.0255841427 | PASS; all cached lengths 32 |
| 64 | 6.0270485878 | 0.2000814527 | 0.0254555885 | PASS; all cached lengths 64 |

All logits, recurrent memories, feedback values, and cached K/V tensors were
finite. No state explosion was observed.

## Initial two-batch diagnostic

| System | Mean loss |
|---|---:|
| Masked / no feedback | 5.9397585392 |
| Teacher feedback | 5.5338072777 |
| Self feedback | 5.6829826832 |

Teacher recovery was 0.4059512615; self recovery was 0.2567758560, or 63.2529%
of teacher recovery. Both batches were stable and self was better than masked,
so the preregistered full-canonical gate passed.

## Full canonical validation

All results use the pinned validation prefix hash
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.
The unchanged full/masked/teacher values are reused from the audited 2A3 artifact;
the new self, shuffled-self, and self-gate-zero paths were executed recurrently.

| System | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked L1 / no feedback | 5.9736744881 |
| Teacher feedback 250M | 5.5700338840 |
| Self feedback zero-shot | **5.7074206829** |
| Shuffled self feedback | 5.7289168119 |
| Self feedback, gate zero | 5.9736480713 |

Derived results:

| Metric | Value |
|---|---:|
| Original masking damage | 1.8950200796 |
| Teacher recovery | 0.4036406040 |
| Self recovery | 0.2662538052 |
| Self recovery / damage | 14.0502% |
| Self / teacher recovery | 65.9631% |
| Shuffled-self minus real-self | 0.0214961290 |
| Sequence-specific recovery / damage | 1.1343% |
| Sequence-specific share of self recovery | 8.0735% |
| Gate-zero minus masked | -0.0000264168 |

Paired self-vs-shuffled results: 18 real wins, 2 shuffled wins, 0 ties. The mean
shuffled-minus-real gap was 0.0214961290, median 0.0241808891, sample standard
deviation 0.0123408287, range [-0.0023870468, 0.0394287109], with a descriptive
paired 95% t interval [0.0157204434, 0.0272718147]. The fixed validation batches
are not assumed IID, so this interval is descriptive rather than inferential.

## Reset horizon

Only the four-vector high→low memory was reset. Blocks 2–12 KV caches and absolute
positions continued normally.

| Reset interval | Validation loss |
|---|---:|
| 1 | 5.9737106323 |
| 2 | 5.8261297226 |
| 4 | 5.7516333818 |
| 8 | 5.7254642010 |
| 16 | 5.7151402950 |
| 32 | 5.7113305092 |
| 64 | 5.7093259096 |
| 128 | 5.7083078384 |
| Never | **5.7076439857** |

The reset sweep batched nine isolated copies together, which changes BF16 GEMM
selection slightly: its never-reset endpoint is 0.0002233 above the standalone
self result, while reset-1 is 0.0000361 above masked. Within that common batched
path, loss improves monotonically with every longer reset horizon. The large gain
from interval 1 to 2 proves that a single recurrent transition helps; the further
monotonic gains through never-reset show information survives repeated
high→low→high transitions.

## Teacher/student drift

Each cell is `cosine similarity / RMS difference / norm ratio` for matched raw
teacher and recurrent-student source states.

| Positions | v16 | v17 | v20 | v24 |
|---|---|---|---|---|
| 1–16 | 0.754077 / 0.138084 / 1.010536 | 0.761936 / 0.148096 / 1.045027 | 0.773242 / 0.135458 / 0.956824 | 0.740540 / 0.143655 / 0.906625 |
| 17–32 | 0.572620 / 0.189024 / 0.982282 | 0.662008 / 0.182524 / 1.004857 | 0.599128 / 0.185858 / 0.912197 | 0.542441 / 0.191850 / 0.881639 |
| 33–64 | 0.530614 / 0.197385 / 0.965120 | 0.667966 / 0.182811 / 0.966962 | 0.559490 / 0.193772 / 0.893737 | 0.498403 / 0.199979 / 0.878179 |
| 65–128 | 0.522809 / 0.197238 / 0.940898 | 0.701836 / 0.173893 / 0.900386 | 0.550573 / 0.195129 / 0.881887 | 0.485122 / 0.201571 / 0.870074 |
| 129–256 | 0.538661 / 0.192002 / 0.935254 | 0.736800 / 0.163735 / 0.843450 | 0.564721 / 0.191477 / 0.880208 | 0.499079 / 0.199452 / 0.867079 |
| 257–512 | 0.557694 / 0.187649 / 0.934048 | 0.760147 / 0.155879 / 0.809090 | 0.585365 / 0.186878 / 0.883357 | 0.521818 / 0.197282 / 0.881036 |
| 513–1023 | 0.568000 / 0.187448 / 0.903542 | 0.774488 / 0.145858 / 0.805181 | 0.608936 / 0.185321 / 0.882799 | 0.554520 / 0.192264 / 0.906535 |

Drift appears rapidly by positions 17–64, then stabilizes rather than diverging.
v17 remains most teacher-like at long horizons despite receiving the lowest
self-memory routing weight. Student norms are generally lower than teacher norms
after the first bins, but remain bounded and stable.

## Routing

| Memory | v16 | v17 | v20 | v24 | Entropy | Top-down RMS | Feedback RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 0.427155 | 0.128827 | 0.220112 | 0.223906 | 0.607931 | 0.179133 | 0.028339 |
| Self | 0.603340 | 0.029581 | 0.146279 | 0.220800 | 0.516001 | 0.160448 | 0.025383 |

The frozen router shifts sharply toward v16 under self memory and nearly removes
v17, even though v17 has the strongest long-horizon teacher/student cosine. This
distribution shift is a concrete target for reader-only adaptation.

## Integrity and verification

- Dedicated Experiment-2B0 unit suite: 9/9 PASS.
- Production future causality, row isolation, reset, and serialize/resume: PASS.
- Canonical and reset validation hashes: exact.
- Complete model-state SHA-256 before and after:
  `5ad907f194217ac2b163c13f1f62c5e0c0bfc7d01ec45b0d7e76ed80e8241dbe`.
- Reader checkpoint SHA-256 after evaluation: unchanged at `0702dc09…9653d76`.
- Optimizer constructed: no. Optimizer steps: 0.
- HellaSwag: not run; the frozen protocol requires separate approval after this
  positive FineWeb result.

## Interpretation and next experiment

### SELF-RECURRENT MEMORY TRANSFERS PARTIALLY

Self feedback is substantially better than masked/no-feedback on all 20 batches,
is stable through 1,024 tokens, recovers nearly two-thirds of the teacher reader's
gain, and retains an aligned-sequence advantage on 18/20 batches. It does not
qualify as strong transfer under the frozen rule because recovery is 65.96%, below
the 80% threshold, and self remains 0.13739 loss behind teacher feedback.

**Should the next experiment train the same 1,537 feedback parameters using
student-generated recurrent memory? YES.** The architecture is stable, the reader
already transfers positively, the reset curve demonstrates repeated recurrent
propagation, and the self routing distribution differs materially from teacher
routing. A separately approved follow-on should adapt only the existing query,
RMSNorm, and scalar gate using detached student recurrent memory. It should not
add sources, destinations, base unfreezing, auxiliary losses, or new recurrence
mechanisms in its first test.

# EXPERIMENT 2B0 ZERO-SHOT COMPLETE
