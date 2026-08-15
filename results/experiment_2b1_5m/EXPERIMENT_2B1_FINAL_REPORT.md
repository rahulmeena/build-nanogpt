# Experiment 2B1 — Train the Reader on Detached Self-Recurrent Memory

## Outcome

**SELF-ADAPTATION IS NEUTRAL**

Training the existing 1,537-parameter reader for exactly 10 updates / 5,242,880
tokens produced a small favorable change on the canonical recurrent validation:
self-memory loss fell from **5.7074206829** to **5.7013087273**, an adaptation
gain of **0.0061119556**. The aligned-sequence signal also strengthened:
shuffled-minus-real increased from **0.0214961290** to **0.0297215700**, and real
memory beat shuffled memory on 20/20 batches rather than 18/20.

These changes are directionally positive, but the preregistered equality band is
0.01 loss. The total adaptation gain remains inside that band, so the result is
classified as neutral rather than as improved sequence memory. The exact same
checkpoint and optimizer should **not** continue to approximately 25M tokens.

All safety, replay, checkpoint, frozen-base, restart, validation, and cache
integrity checks passed. No teacher forward was used in training, no temporal
gradient path was present, no optimizer update beyond 10 was run, and HellaSwag
was not launched.

## Git

| Field | Value |
|---|---|
| Experiment-2B0 frozen tag | `experiment-2b0-zero-shot` |
| Experiment-2B0 full commit | `a8b271ee71ae1af77da8ddad022ce549be390682` |
| Experiment-2B1 branch | `experiment-2b1-self-reader-adaptation` |
| Experiment-2B1 implementation commit | `c2f8082a7e0deff797a59e1f8b78949e7da3d246` |
| Experiment-2B1 results commit | The commit containing this report; recorded in the handoff |

The 2B0 tag was created and pushed before the 2B1 branch. The final audit ran
against the same implementation commit recorded in both result checkpoints.

## Initialization

| Field | Value |
|---|---|
| Source checkpoint | `/workspace/build-nanogpt-exp2a0/runs/experiment_2a3_250m/checkpoints/checkpoint_updates_000477.pt` |
| Source SHA-256 | `0702dc09c74b01eee8be504a7f5f89ca61fcc504cda8f34f30865d4ff9653d76` |
| Source reader updates / tokens | 477 / 250,085,376 |
| Zero-shot gate | 0.1595292091 |
| Zero-shot tanh(gate) | 0.1581895351 |
| Zero-shot query norm | 1.9564993382 |
| Zero-shot RMSNorm displacement | 1.9469332695 |
| Reader parameters | 1,537 |
| First 2B1 global-batch SHA-256 | `95081c5f68b7d05d6e39b68043f2714657c21ca05cc317549063ba9a4f9f6986` |

The source checkpoint hash and exact next FineWeb batch were verified before any
result-bearing optimizer step. The reader weights were not reinitialized. A new
AdamW optimizer was constructed without restoring the 2A3 teacher-memory moments.

## Preflight and disposable smoke

All gates passed before the large run:

- Full-context BF16 incremental/parallel loss difference: 0.0014343262;
  mean absolute logit difference: 0.0093973279.
- Masked-L1 BF16 incremental/parallel loss difference: 0.0006313324;
  mean absolute logit difference: 0.0108051747.
- FP32 chunked-backward maximum absolute gradient difference:
  `8.8817841970e-16`; maximum relative difference: `6.0563625692e-08`.
- Future-suffix causality, row isolation, fresh reset, Block-1 cache absence,
  and serialized continuation were bit-exact.
- Prior-token recurrent-memory and historical-KV gradients were absent;
  stored state had no `grad_fn`; current reader gradients were finite/nonzero.
- T=8/16/32/64 recurrent state and cache stability checks passed.
- Pre-training canonical regression reproduced masked **5.9736744881**,
  zero-shot self **5.7074206829**, and gate-zero **5.9736480713** exactly.
- The disposable B2×T64 smoke completed three updates, saved after update 2,
  resumed update 3 in a fresh process, matched the next batch, and was discarded.

## Training

| Field | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0 |
| Learning rate | constant 1.0e-4 |
| Gradient clip | 1.0 |
| Backward chunk | 16 tokens |
| Updates | exactly 10 |
| Targets per update | 524,288 |
| 2B1 tokens | 5,242,880 |
| Training wall time | 41,716.56 s = 11 h 35 m 17 s |
| Peak allocated VRAM | 40,865.98 MiB |
| Peak reserved VRAM | 80,574 MiB |
| Teacher training forwards | 0 |
| Final checkpoint SHA-256 | `5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a` |

Every update consumed its preregistered replay hash, included exactly 524,288
targets, produced finite/nonzero query/RMSNorm/gate gradients, left every frozen
gradient absent, and kept all recurrent memories and historical K/V finite and
detached.

| Update | Loss | Grad norm | Gate | Entropy | v16 | v17 | v20 | v24 | Wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.592898 | 0.190651 | 0.159629 | 0.510740 | 0.617002 | 0.033056 | 0.146680 | 0.203263 | 4173.9 |
| 2 | 5.627865 | 0.176981 | 0.159579 | 0.506779 | 0.620535 | 0.031157 | 0.145080 | 0.203229 | 4171.0 |
| 3 | 5.608401 | 0.178392 | 0.159525 | 0.502357 | 0.626516 | 0.029218 | 0.141675 | 0.202590 | 4172.3 |
| 4 | 5.562062 | 0.150182 | 0.159475 | 0.501499 | 0.623939 | 0.027895 | 0.143406 | 0.204761 | 4180.9 |
| 5 | 5.585043 | 0.137895 | 0.159433 | 0.491689 | 0.633005 | 0.025142 | 0.139335 | 0.202517 | 4180.1 |
| 6 | 5.638240 | 0.142903 | 0.159375 | 0.500005 | 0.631184 | 0.030385 | 0.135870 | 0.202561 | 4177.7 |
| 7 | 5.686364 | 0.131265 | 0.159363 | 0.495053 | 0.628849 | 0.025397 | 0.136909 | 0.208845 | 4164.1 |
| 8 | 5.624338 | 0.115038 | 0.159369 | 0.490614 | 0.632123 | 0.023866 | 0.137893 | 0.206119 | 4164.8 |
| 9 | 5.626139 | 0.115854 | 0.159397 | 0.481374 | 0.639458 | 0.021999 | 0.133293 | 0.205251 | 4164.7 |
| 10 | 5.706830 | 0.102065 | 0.159413 | 0.490044 | 0.630181 | 0.022974 | 0.129629 | 0.217216 | 4167.0 |

## Resume audit

| Field | Value |
|---|---|
| Update-5 checkpoint SHA-256 | `aca49ac08c607a71320a7cfa922df2db5e6936320de0f6a45f49a085516835db` |
| Restored next-batch SHA-256 | `129c40a5211020d3fe13cfb32e356651a97d6bbcc0a3da4e86ee933d89cc9132` |
| Save process / resume process | PID 26211 / PID 27932 |
| Model strict reload | PASS |
| Optimizer strict reload | PASS; Adam steps 5/5/5 |
| Four-loader strict reload | PASS |
| Python/NumPy/Torch CPU/Torch CUDA RNG reload | PASS |
| Fresh-process update-6 batch verification | PASS; exact restored hash |

The final checkpoint strict-reloaded at local update 10 with Adam steps 10/10/10,
four replay-loader states, all RNG fields, and next-batch SHA-256
`3c4d0cd7905f16bfcfdd283cbc0799ff9f85c91ac3c521e886f0e403fc11ae57`.

## Canonical validation

All controls used 20×B64×T1024 BF16 batches with pinned aggregate SHA-256
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

| System | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked L1 / no feedback | 5.9736744881 |
| Zero-shot self | 5.7074206829 |
| Trained self | **5.7013087273** |
| Trained shuffled self | 5.7310302973 |
| Trained gate-zero | 5.9736480713 |
| Trained reader + teacher memory | 5.5720659256 |
| Teacher reader before adaptation | 5.5700338840 |

The adapted reader improved self-memory slightly while teacher-memory
compatibility worsened by 0.0020320415. Gate-zero remained unchanged, confirming
that the measured behavior still enters through the feedback path.

## Derived metrics

| Metric | Value |
|---|---:|
| Zero-shot self recovery | 0.2662538052 |
| Trained self recovery | 0.2723657608 |
| Adaptation gain | **0.0061119556** |
| Zero-shot specific gap | 0.0214961290 |
| Trained specific gap | **0.0297215700** |
| Specific-gap gain | **0.0082254410** |
| Trained recovery / masking damage | 14.3727% |
| Trained recovery / original teacher recovery | 67.4773% |

## Paired trained-self vs shuffled-self

| Metric | Value |
|---|---:|
| Real wins | 20 |
| Shuffled wins | 0 |
| Ties | 0 |
| Mean shuffled-minus-real | 0.0297215700 |
| Median | 0.0317671299 |
| Sample standard deviation | 0.0118614709 |
| Minimum | 0.0069518089 |
| Maximum | 0.0462307930 |

The sequence-specific signal became more consistent than in 2B0 (18/20 wins),
but the total adaptation gain remained below the 0.01 classification threshold.

## Reader evolution

| Quantity | Before | After |
|---|---:|---:|
| Gate | 0.1595292091 | 0.1594133824 |
| tanh(gate) | 0.1581895351 | 0.1580765992 |
| Query norm | 1.9564993382 | 1.9570868015 |
| RMSNorm displacement | 1.9469332695 | 1.9478669167 |
| v16 routing weight | 0.6033400536 | 0.6340336651 |
| v17 routing weight | 0.0295812954 | 0.0206797247 |
| v20 routing weight | 0.1462786213 | 0.1305237375 |
| v24 routing weight | 0.2207999326 | 0.2147629216 |
| Routing entropy | 0.5160012662 | 0.4845177144 |
| Top-down RMS | 0.1604477145 | 0.1623093784 |
| Feedback RMS | 0.0253828241 | 0.0256773381 |

Adaptation sharpened the self router further toward v16 and reduced v17 rather
than restoring it. With teacher memory, the trained reader routed
v16/v17/v20/v24 as 0.458106/0.102152/0.213272/0.226470 with entropy 0.591857.

## Reset horizon

Only the four-vector high→low memory was reset; Blocks 2–12 K/V and absolute
positions continued. The trained sweep remained monotonic toward uninterrupted
recurrence and showed increasing improvement over zero-shot at longer horizons.

| Interval | Zero-shot loss | Trained loss | Trained − zero-shot |
|---:|---:|---:|---:|
| 1 | 5.9737106323 | 5.9737106323 | +0.0000000000 |
| 2 | 5.8261297226 | 5.8261914968 | +0.0000617743 |
| 4 | 5.7516333818 | 5.7499492407 | -0.0016841412 |
| 8 | 5.7254642010 | 5.7218983650 | -0.0035658360 |
| 16 | 5.7151402950 | 5.7104103088 | -0.0047299862 |
| 32 | 5.7113305092 | 5.7059802055 | -0.0053503036 |
| 64 | 5.7093259096 | 5.7035196781 | -0.0058062315 |
| 128 | 5.7083078384 | 5.7021847963 | -0.0061230421 |
| Never | 5.7076439857 | **5.7013566494** | -0.0062873363 |

The batched nine-horizon endpoint differs slightly from standalone canonical
loss because its larger BF16 GEMM shape selects different reductions.

## Renormalized source ablation

| Removed source | Ablated loss | Delta vs trained real self | Positive batches | Negative batches |
|---|---:|---:|---:|---:|
| v16 | 6.3152048111 | +0.6138960838 | 20 | 0 |
| v17 | 5.6999582291 | -0.0013504982 | 5 | 15 |
| v20 | 5.7507675886 | +0.0494588614 | 20 | 0 |
| v24 | 5.8199162960 | +0.1186075687 | 20 | 0 |

These leave-one-out effects are renormalized and are not additive. v16 became
decisively dominant; v24 and v20 were consistently useful; v17 was effectively
neutral and slightly adverse under the adapted self router.

## Teacher/student drift

Mean teacher/student RMS difference changed from 0.1800561583 to 0.1797343626
(-0.1787%), while mean cosine changed from 0.6140368721 to 0.6152218420
(+0.001185). Under the frozen ±1% RMS rule, drift **stayed similar**.

Each table cell is `zero-shot cosine/RMS/norm → trained cosine/RMS/norm`.

| Positions | v16 | v17 | v20 | v24 |
|---|---|---|---|---|
| 1-16 | 0.754077/0.138084/1.010536 → 0.755651/0.137642/1.010164 | 0.761936/0.148096/1.045027 → 0.763843/0.147133/1.041506 | 0.773242/0.135458/0.956824 → 0.775097/0.134971/0.957047 | 0.740540/0.143655/0.906625 → 0.742524/0.143211/0.907823 |
| 17-32 | 0.572620/0.189024/0.982282 → 0.574554/0.188572/0.982075 | 0.662008/0.182524/1.004857 → 0.664505/0.181200/0.998442 | 0.599128/0.185858/0.912197 → 0.601516/0.185311/0.912317 | 0.542441/0.191850/0.881639 → 0.544797/0.191516/0.883566 |
| 33-64 | 0.530614/0.197385/0.965120 → 0.532318/0.196999/0.964985 | 0.667966/0.182811/0.966962 → 0.670360/0.181614/0.960969 | 0.559490/0.193772/0.893737 → 0.561660/0.193285/0.893785 | 0.498403/0.199979/0.878179 → 0.500394/0.199758/0.880415 |
| 65-128 | 0.522809/0.197238/0.940898 → 0.523729/0.197048/0.940986 | 0.701836/0.173893/0.900386 → 0.703201/0.173193/0.895895 | 0.550573/0.195129/0.881887 → 0.551755/0.194869/0.881996 | 0.485122/0.201571/0.870074 → 0.486099/0.201546/0.872399 |
| 129-256 | 0.538661/0.192002/0.935254 → 0.538984/0.191959/0.935618 | 0.736800/0.163735/0.843450 → 0.737119/0.163517/0.840744 | 0.564721/0.191477/0.880208 → 0.565167/0.191385/0.880430 | 0.499079/0.199452/0.867079 → 0.499205/0.199587/0.869509 |
| 257-512 | 0.557694/0.187649/0.934048 → 0.558266/0.187572/0.934608 | 0.760147/0.155879/0.809090 → 0.760260/0.155795/0.807362 | 0.585365/0.186878/0.883357 → 0.585757/0.186812/0.883769 | 0.521818/0.197282/0.881036 → 0.521807/0.197431/0.883301 |
| 513-1023 | 0.568000/0.187448/0.903542 → 0.568789/0.187306/0.903856 | 0.774488/0.145858/0.805181 → 0.774806/0.145734/0.804088 | 0.608936/0.185321/0.882799 → 0.609441/0.185221/0.883090 | 0.554520/0.192264/0.906535 → 0.554606/0.192376/0.908536 |

The raw unrounded tables are preserved in `FINAL_AUDIT.json`.

## Integrity and scope audit

The final audit passed every required check:

- source 2B0 tag/commit and source checkpoint SHA were exact;
- exactly 10 updates and 5,242,880 targets were consumed;
- all ten replay hashes and all per-update target counts were exact;
- every reader gradient was present, finite, and nonzero;
- every frozen gradient was absent and every stored recurrent state was valid;
- update 5 and update 10 checkpoint sidecars strict-reloaded and rehashed;
- the forced update-5 process exit and fresh update-6 process were distinct;
- the frozen GPT-2/Full-AttnRes base hash remained unchanged;
- canonical and diagnostic validation hashes were exact;
- authorized optimizer updates were exhausted at 10; beyond-10 updates: 0;
- HellaSwag artifacts were absent.

HellaSwag was not run. The audit's planning estimate for five recurrent controls
is 80.0–160.1 A100 GPU-hours, with strict candidate isolation; it is an estimate,
not a benchmark.

## Final classification and continuation decision

### SELF-ADAPTATION IS NEUTRAL

The run is stable and directionally favorable: real self loss improved by
0.006112, the sequence-specific gap improved by 0.008225, real memory won all 20
paired batches, and longer recurrent persistence benefited more after training.
However, the primary adaptation gain remains inside the preregistered ±0.01
neutral band. Reader movement was tiny, representation drift stayed similar,
and the router concentrated still more heavily on v16 rather than learning a
materially broader interpretation of self memory.

**Should the exact same checkpoint, optimizer, detached-state semantics, and
architecture continue from 5.24M to approximately 25M tokens? NO.** The 5M gate
did not establish a sufficiently large self-loss improvement to justify the
continuation. Do not launch additional optimizer updates from this checkpoint.

# EXPERIMENT 2B1 5M COMPLETE
