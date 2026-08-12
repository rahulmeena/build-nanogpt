# Experiment 1B — Final 500M Analysis

This report is derived only from already-produced Experiment 1B artifacts. No new training, evaluation, or ablation was run.

## Executive result

Full AttnRes was stable through **954 updates / 500,170,752 tokens**. Its final validation loss was **4.078654**, versus **4.201933** for Standard, an AttnRes−Standard delta of **-0.123278**. Final HellaSwag was slightly lower: **25.214%** versus **25.573%**.

The validation advantage did **not** grow monotonically all the way to 500M. It widened from **-0.005572** at 100M to **-0.096170** at 250M and **-0.190818** at warmup end, then narrowed to **-0.123278** at 500M. It never changed sign after 100M.

## 1. Matched learning trajectory

`optimizer step` below is completed optimizer updates; the corresponding last zero-based update is one less. HellaSwag was intentionally measured only at 0, ~100M, and ~500M.

| Tokens | Optimizer step | Standard train | Standard val | AttnRes train | AttnRes val | AttnRes−Standard val | HellaSwag Standard / AttnRes | LR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,139,008 | 191 | 6.425617 | 6.422073 | 6.430205 | 6.416501 | -0.005572 | 24.627% / 24.587% | 0.000160280 |
| 250,085,376 | 477 | 5.343612 | 5.374175 | 5.254767 | 5.278005 | -0.096170 | — / — | 0.000400280 |
| 374,865,920 | 715 | 4.710934 | 4.710258 | 4.539973 | 4.519439 | -0.190818 | — / — | 0.000600000 |
| 500,170,752 | 954 | 4.177829 | 4.201933 | 4.056465 | 4.078654 | -0.123278 | 25.573% / 25.214% | 0.000599776 |

At initialization, validation was **10.951632 Standard** and **10.941046 AttnRes**; this small difference reflects the distinct architecture before learning and is not a trained advantage.

## 2. Full AttnRes routing maturation

The first router has only v0 available and therefore has a structurally zero query norm and zero entropy. The learned-router minimum excludes that first router.

| Tokens | Step | Query norm min (all) | Min learned | Median | Max | Median entropy (nats) | Median normalized entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,139,008 | 191 | 0.000000 | 0.020171 | 0.052090 | 0.076953 | 2.557185 | 0.988685 |
| 250,085,376 | 477 | 0.000000 | 0.060146 | 0.106050 | 0.156701 | 2.313240 | 0.931270 |
| 374,865,920 | 715 | 0.000000 | 0.084592 | 0.144612 | 0.232482 | 2.227719 | 0.892830 |
| 500,170,752 | 954 | 0.000000 | 0.102557 | 0.166904 | 0.266855 | 2.220910 | 0.885310 |


Median normalized entropy fell from **0.989** to **0.885**, while median query norm rose from **0.0521** to **0.1669**. Routers therefore became materially more selective, although many late, wide-choice routers remain distributed rather than one-hot.

### Entropy by destination

| Destination | ~100M | ~250M | ~375M | ~500M |
|---|---:|---:|---:|---:|
| Block 1 Attention input | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Block 1 Mlp input | 0.6763 | 0.6208 | 0.3530 | 0.2012 |
| Block 2 Attention input | 1.0907 | 1.0388 | 1.0215 | 1.0028 |
| Block 2 Mlp input | 1.3503 | 1.2036 | 1.0857 | 1.0667 |
| Block 3 Attention input | 1.6075 | 1.5873 | 1.5750 | 1.5734 |
| Block 3 Mlp input | 1.7542 | 1.6686 | 1.5815 | 1.6481 |
| Block 4 Attention input | 1.9422 | 1.9254 | 1.8636 | 1.8517 |
| Block 4 Mlp input | 2.0285 | 1.8170 | 1.5987 | 1.5920 |
| Block 5 Attention input | 2.1927 | 2.1389 | 2.0349 | 1.9491 |
| Block 5 Mlp input | 2.2635 | 1.9226 | 1.6434 | 1.6869 |
| Block 6 Attention input | 2.3969 | 2.3132 | 2.2303 | 2.1795 |
| Block 6 Mlp input | 2.4522 | 2.0962 | 1.8856 | 1.8673 |
| Block 7 Attention input | 2.5572 | 2.3200 | 2.2143 | 2.2209 |
| Block 7 Mlp input | 2.6091 | 2.2914 | 2.2277 | 2.2264 |
| Block 8 Attention input | 2.7018 | 2.5783 | 2.4787 | 2.4651 |
| Block 8 Mlp input | 2.7412 | 2.5200 | 2.4755 | 2.4766 |
| Block 9 Attention input | 2.8260 | 2.7117 | 2.5895 | 2.5616 |
| Block 9 Mlp input | 2.8387 | 2.6352 | 2.5188 | 2.5334 |
| Block 10 Attention input | 2.9387 | 2.8244 | 2.7537 | 2.7311 |
| Block 10 Mlp input | 2.9372 | 2.7635 | 2.6815 | 2.6522 |
| Block 11 Attention input | 3.0381 | 2.9215 | 2.8443 | 2.8230 |
| Block 11 Mlp input | 3.0133 | 2.8786 | 2.6681 | 2.5845 |
| Block 12 Attention input | 3.1274 | 3.0713 | 2.9810 | 2.8606 |
| Block 12 Mlp input | 3.1224 | 2.9779 | 2.8266 | 2.6794 |
| Final LN input | 3.1873 | 3.0194 | 2.8773 | 2.7127 |

### Strongest source by destination

| Destination | ~100M | ~250M | ~375M | ~500M |
|---|---|---|---|---|
| Block 1 Attention input | v0 — Embedding (1.000) | v0 — Embedding (1.000) | v0 — Embedding (1.000) | v0 — Embedding (1.000) |
| Block 1 Mlp input | v0 — Embedding (0.523) | v0 — Embedding (0.644) | v0 — Embedding (0.878) | v0 — Embedding (0.944) |
| Block 2 Attention input | v0 — Embedding (0.363) | v1 — Block 1 Attention (0.435) | v1 — Block 1 Attention (0.477) | v1 — Block 1 Attention (0.501) |
| Block 2 Mlp input | v1 — Block 1 Attention (0.300) | v1 — Block 1 Attention (0.493) | v1 — Block 1 Attention (0.566) | v1 — Block 1 Attention (0.566) |
| Block 3 Attention input | v1 — Block 1 Attention (0.203) | v1 — Block 1 Attention (0.239) | v1 — Block 1 Attention (0.268) | v2 — Block 1 MLP (0.248) |
| Block 3 Mlp input | v2 — Block 1 MLP (0.201) | v2 — Block 1 MLP (0.277) | v1 — Block 1 Attention (0.297) | v1 — Block 1 Attention (0.278) |
| Block 4 Attention input | v1 — Block 1 Attention (0.150) | v1 — Block 1 Attention (0.195) | v1 — Block 1 Attention (0.253) | v1 — Block 1 Attention (0.229) |
| Block 4 Mlp input | v2 — Block 1 MLP (0.162) | v2 — Block 1 MLP (0.290) | v2 — Block 1 MLP (0.323) | v1 — Block 1 Attention (0.317) |
| Block 5 Attention input | v7 — Block 4 Attention (0.119) | v7 — Block 4 Attention (0.208) | v7 — Block 4 Attention (0.289) | v7 — Block 4 Attention (0.347) |
| Block 5 Mlp input | v2 — Block 1 MLP (0.130) | v2 — Block 1 MLP (0.280) | v1 — Block 1 Attention (0.376) | v1 — Block 1 Attention (0.352) |
| Block 6 Attention input | v2 — Block 1 MLP (0.095) | v9 — Block 5 Attention (0.180) | v9 — Block 5 Attention (0.192) | v9 — Block 5 Attention (0.222) |
| Block 6 Mlp input | v2 — Block 1 MLP (0.108) | v11 — Block 6 Attention (0.303) | v11 — Block 6 Attention (0.402) | v11 — Block 6 Attention (0.413) |
| Block 7 Attention input | v1 — Block 1 Attention (0.088) | v11 — Block 6 Attention (0.282) | v11 — Block 6 Attention (0.321) | v11 — Block 6 Attention (0.292) |
| Block 7 Mlp input | v1 — Block 1 Attention (0.094) | v11 — Block 6 Attention (0.271) | v11 — Block 6 Attention (0.241) | v13 — Block 7 Attention (0.253) |
| Block 8 Attention input | v1 — Block 1 Attention (0.075) | v11 — Block 6 Attention (0.167) | v2 — Block 1 MLP (0.161) | v14 — Block 7 MLP (0.152) |
| Block 8 Mlp input | v1 — Block 1 Attention (0.084) | v11 — Block 6 Attention (0.206) | v11 — Block 6 Attention (0.184) | v13 — Block 7 Attention (0.133) |
| Block 9 Attention input | v1 — Block 1 Attention (0.068) | v11 — Block 6 Attention (0.149) | v11 — Block 6 Attention (0.177) | v13 — Block 7 Attention (0.142) |
| Block 9 Mlp input | v1 — Block 1 Attention (0.082) | v11 — Block 6 Attention (0.166) | v1 — Block 1 Attention (0.195) | v1 — Block 1 Attention (0.156) |
| Block 10 Attention input | v1 — Block 1 Attention (0.060) | v11 — Block 6 Attention (0.128) | v17 — Block 9 Attention (0.103) | v17 — Block 9 Attention (0.102) |
| Block 10 Mlp input | v1 — Block 1 Attention (0.075) | v11 — Block 6 Attention (0.159) | v1 — Block 1 Attention (0.150) | v1 — Block 1 Attention (0.132) |
| Block 11 Attention input | v4 — Block 2 MLP (0.054) | v11 — Block 6 Attention (0.121) | v11 — Block 6 Attention (0.124) | v17 — Block 9 Attention (0.100) |
| Block 11 Mlp input | v1 — Block 1 Attention (0.071) | v11 — Block 6 Attention (0.106) | v1 — Block 1 Attention (0.202) | v1 — Block 1 Attention (0.212) |
| Block 12 Attention input | v1 — Block 1 Attention (0.050) | v11 — Block 6 Attention (0.079) | v17 — Block 9 Attention (0.081) | v17 — Block 9 Attention (0.098) |
| Block 12 Mlp input | v1 — Block 1 Attention (0.061) | v15 — Block 8 Attention (0.092) | v1 — Block 1 Attention (0.133) | v1 — Block 1 Attention (0.177) |
| Final LN input | v14 — Block 7 MLP (0.053) | v15 — Block 8 Attention (0.091) | v1 — Block 1 Attention (0.142) | v1 — Block 1 Attention (0.192) |

### Clear routing patterns

- **Very early processing:** Block 1 Attention must use the embedding. Block 1 MLP increasingly concentrates on v0, reaching weight **0.944**. Blocks 2–4 mostly retrieve v1/v2, the Block 1 Attention/MLP states.
- **Middle routing hubs:** Block 5 Attention strongly prefers v7 (Block 4 Attention, **0.347**). Block 6 MLP strongly prefers v11 (Block 6 Attention, **0.413**). Blocks 7–9 organize around v11/v13/v14, the Block 6–7 states.
- **Late retrieval:** Blocks 10–12 Attention prefer v17 (Block 9 Attention) at 500M. Several late MLP destinations return all the way to v1, showing genuine long-depth retrieval rather than only local skips.
- **Final router:** Final LN input prefers v1 (**0.192**), then v17 (**0.101**), v15 (**0.091**), v22 (**0.069**), v20 (**0.063**), v18 (**0.061**), and v24 (**0.057**). It combines early and intermediate/late representations.
- **Uniformity:** Routers are not uniformly collapsed. Some are highly concentrated, while late routers retain high entropy because they mix many eligible sources. The complete matrix is in `final_routing_matrix.csv`; `final_routing_heatmap.svg` is its visual rendering.

## 3. Causal ablation

The saved ablation tested exactly **v0, v4, v8, v12, v16, v20, and v24**: embedding plus every fourth residual source, corresponding to the MLP output after Blocks 2, 4, 6, 8, 10, and 12. This representative subset followed the protocol's instruction to avoid an exhaustive combinatorial ablation.

| Source | Human-readable source | Normal val | Masked val | Delta loss | Final mean routing weight when eligible |
|---|---|---:|---:|---:|---:|
| v0 | Embedding | 4.031627 | 7.583286 | +3.551659 | 0.1770 |
| v20 | Block 10 MLP | 4.031627 | 4.160790 | +0.129162 | 0.0701 |
| v16 | Block 8 MLP | 4.031627 | 4.136947 | +0.105320 | 0.0645 |
| v24 | Block 12 MLP | 4.031627 | 4.127351 | +0.095724 | 0.0575 |
| v12 | Block 6 MLP | 4.031627 | 4.111422 | +0.079794 | 0.0472 |
| v4 | Block 2 MLP | 4.031627 | 4.094478 | +0.062850 | 0.0594 |
| v8 | Block 4 MLP | 4.031627 | 4.067571 | +0.035944 | 0.0282 |

v0 has a special limitation: it remains the sole input to the first attention sublayer, so the utility cannot remove it there; it masks v0 only from later routers. Its very large delta should not be compared naively with the other sources.

Among non-embedding tested states, **Block 10 MLP (v20)** was most causally important, followed by **Block 8 MLP (v16)**, **Block 12 MLP (v24)**, and **Block 6 MLP (v12)**. Across the six tested non-v0 sources, causal delta and mean routing weight when eligible have Pearson **r≈0.871**. This is suggestive, not definitive (n=6 and the sources were not exhaustively sampled). Total routing mass is confounded by how many later destinations can access a source.

## 4. Destination-specific behavior

No saved experiment performs destination×source causal interventions, so destination-specific statements are **routing associations, not destination-specific causal proof**. Existing evidence supports:

- Block 5 Attention → v7 / Block 4 Attention (**0.347**).
- Block 6 MLP → v11 / Block 6 Attention (**0.413**).
- Block 7 Attention → v11 / Block 6 Attention (**0.292**).
- Block 7 MLP → v13 / Block 7 Attention (**0.253**).
- Blocks 10–12 Attention → v17 / Block 9 Attention (~**0.098–0.102** each).
- Final LN input mixes v1, v17, v15, v22, v20, v18, and v24 rather than relying on a single local predecessor.

The exact top-three routes for every destination are in `final_routing_top3.csv`.

## 5. Four-GPU performance

| Metric | Standard | Full AttnRes |
|---|---:|---:|
| Main-run throughput | 596,617.9 tok/s | 76,235.6 tok/s |
| Main-run runtime | 994.9s (16.58 min) | 7545.6s (125.76 min) |
| Controlled 4-GPU benchmark | 606,460.8 tok/s | 76,425.0 tok/s |
| Previous 1-GPU reference | 168,645 tok/s | 21,257 tok/s |
| Controlled 4-GPU speedup | 3.596× | 3.595× |
| Controlled scaling efficiency | 89.90% | 89.88% |
| Peak allocated VRAM/GPU | 61,483.5 MiB | 64,108.0 MiB |
| Peak reserved VRAM/GPU | 78,396.0 MiB | 76,992.0 MiB |

Standard delivered **7.826×** Full AttnRes throughput; Full AttnRes took **7.584×** as long end-to-end.

## 6. Resume/checkpoint validation

Both final checkpoints were successfully written and read-only reload-verified by the training harness. Each verification covered model state, AdamW moments, completed step/tokens, four per-rank DataLoader states, four per-rank RNG states, world-size metadata, and the next-global-batch hash.

- **Standard final recorded path:** `/workspace/build-nanogpt/runs/exp1b_500m/standard/checkpoints/checkpoint_tokens_000500170752.pt`
  **SHA-256:** `3343ddb9b780c47e94da20f304517f2fa4591abf1d9bc3d7e4dd7f4af5b6690c`
  **Optimizer:** 148 state/moment entries, all finite and nonzero.
- **Full AttnRes final path:** `/workspace/build-nanogpt/runs/exp1b_500m/full_attnres/checkpoints/checkpoint_tokens_000500170752.pt`
  **SHA-256:** `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`
  **Optimizer:** 198 state/moment entries, 196 nonzero (the two zero moments are the structurally inactive first one-source router query/norm), all finite.

The Full AttnRes final payload was additionally reopened CPU-only during final analysis: 199 model tensors, 198 optimizer states, step 954, 500,170,752 tokens, four DataLoader states, and four RNG dictionaries containing `python_random`, `numpy_random`, `torch_cpu`, and per-rank `torch_cuda`. Metadata records `world_size=4`.

## 7. Artifact preservation audit

The currently mounted persistent US-WA volume contains:

- complete Standard and Full AttnRes metrics and summaries;
- all routing snapshots and the causal-ablation JSON;
- all four Full AttnRes resumable checkpoint payloads plus SHA/completion/verification sidecars;
- resume and host-migration audits;
- this final report, CSV tables, JSON data, and routing heatmap after Git synchronization.

**Preservation gap:** the four large Standard checkpoint payloads are not on the currently mounted US-WA volume. They were successfully reload-verified and their paths/SHAs are preserved in Standard's summary and `checkpoint_manifest.csv`, but the payloads were written on the earlier US-MD network volume. That volume must be mounted again if the Standard payloads need to be co-located or copied. No new GPU compute is required; a CPU pod is sufficient for that storage transfer.

## 8. Experiment-2 recommendation

**Yes—with a staged design.** Lower→higher AttnRes is sufficiently mature and stable to serve as the substrate for a high→low cross-token feedback experiment: it trained stably to 500M, beat Standard validation at every trained milestone, specialized progressively, and every tested source had positive causal utility. The caveat is that final HellaSwag was **0.358 percentage points lower**, so Experiment 2 should use a conservative gated feedback path and preserve a no-feedback control.

Most useful higher-layer states, based on direct causal evidence, are **Block 10 MLP (v20)**, **Block 8 MLP (v16)**, and **Block 12 MLP (v24)**. Based on routing association (not ablation), **Block 9 Attention (v17)** is also a strong candidate because it is the preferred source of Blocks 10–12 Attention and the second-largest final-LN route.

The most logical first recipients of top-down feedback are middle destinations that already act as routing integration hubs: **Block 5 Attention input**, **Block 6 Attention/MLP inputs**, and **Block 7 Attention input**. Begin there before injecting into Blocks 1–3, whose strong early-state specialization appears foundational and may be more destabilizing.

## Machine-readable artifacts

- `matched_learning_trajectory.csv`
- `routing_maturation_summary.csv`
- `routing_entropy_by_destination.csv`
- `routing_strongest_source_by_destination.csv`
- `final_routing_matrix.csv`
- `final_routing_top3.csv`
- `final_routing_heatmap.svg`
- `causal_ablation.csv`
- `checkpoint_manifest.csv`
- `report_data.json`

EXPERIMENT 1B ANALYSIS FINALIZED
