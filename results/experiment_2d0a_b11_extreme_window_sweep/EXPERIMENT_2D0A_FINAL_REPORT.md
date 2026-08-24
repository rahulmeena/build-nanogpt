# Experiment 2D0A — B11 Extreme KV-Window Sensitivity Sweep

## Outcome

Classification: **B11 EXPLICIT HISTORY MODERATELY REDUNDANT**.

This was an evaluation-only sweep. Optimizers, backward calls, parameter updates, and training targets were all zero. No recurrence or completion module was active.

## Complete B11 sensitivity curve

| Window | Val loss | Damage | Historical KV retained | KV fraction | B11 cosine | B11 RMS | B12 cosine | B12 RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 3.0750437753 | +0.0000000000 | 1023 | 1.000000 | 0.9999999917 | 0.0000000000 | 0.9999999916 | 0.0000000000 |
| 896 | 3.0753463744 | +0.0003025990 | 895 | 0.875000 | 0.9998283632 | 0.0125425534 | 0.9998353501 | 0.0132700680 |
| 768 | 3.0757388638 | +0.0006950885 | 767 | 0.750000 | 0.9996055236 | 0.0249192500 | 0.9996199206 | 0.0247916507 |
| 512 | 3.0773528399 | +0.0023090646 | 511 | 0.500000 | 0.9988526994 | 0.0607526377 | 0.9988820830 | 0.0589212351 |
| 384 | 3.0792040860 | +0.0041603106 | 383 | 0.375000 | 0.9981447150 | 0.0896239768 | 0.9982015159 | 0.0996532427 |
| 256 | 3.0827297657 | +0.0076859903 | 255 | 0.250000 | 0.9968389391 | 0.1354983650 | 0.9969480877 | 0.1502632061 |
| 128 | 3.0934758906 | +0.0184321153 | 127 | 0.125000 | 0.9933011459 | 0.2295864673 | 0.9935888583 | 0.2538782506 |
| 1 | 3.3540805459 | +0.2790367706 | 0 | 0.000977 | 0.9155252578 | 0.9726895127 | 0.9160033359 | 1.0861344004 |

The KV fraction is W/1024 for B11 only; it is not a total-model KV saving. Historical 2D0 B12 diagnostics used the final normalized top state, while new 2D0A diagnostics use the B12 post-block H12 state.

## Quality-retention Pareto profile

| Allowed damage | Smallest B11 window | KV fraction |
|---:|---:|---:|
| 0.0010 | 768 | 0.750000 |
| 0.0025 | 512 | 0.500000 |
| 0.0050 | 384 | 0.375000 |
| 0.0100 | 256 | 0.250000 |
| 0.0200 | 128 | 0.125000 |
| 0.0500 | 128 | 0.125000 |
| 0.1000 | 128 | 0.125000 |

Windows incidentally in the old 0.01–0.10 recurrence-reference band: [128]. This does not authorize recurrence training.

## Position-dependent loss

### W384

| Position bin | Full loss | Short loss | Delta | Targets |
|---|---:|---:|---:|---:|
| 1-64 | 3.5649234841 | 3.5649136591 | -0.0000098250 | 81920 |
| 65-128 | 3.1827517307 | 3.1827433139 | -0.0000084169 | 81920 |
| 129-256 | 3.0944029219 | 3.0944142516 | +0.0000113297 | 163840 |
| 257-384 | 3.0448328851 | 3.0448309642 | -0.0000019209 | 163840 |
| 385-512 | 3.0298848223 | 3.0346281455 | +0.0047433232 | 163840 |
| 513-768 | 3.0125585693 | 3.0192739213 | +0.0067153520 | 327680 |
| 769-896 | 3.0069411320 | 3.0136387277 | +0.0066975957 | 163840 |
| 897-1023 | 3.0016298198 | 3.0101066191 | +0.0084767993 | 162560 |

### W256

| Position bin | Full loss | Short loss | Delta | Targets |
|---|---:|---:|---:|---:|
| 1-64 | 3.5649234841 | 3.5649136591 | -0.0000098250 | 81920 |
| 65-128 | 3.1827517307 | 3.1827433139 | -0.0000084169 | 81920 |
| 129-256 | 3.0944029219 | 3.0944308903 | +0.0000279683 | 163840 |
| 257-384 | 3.0448328851 | 3.0517739277 | +0.0069410425 | 163840 |
| 385-512 | 3.0298848223 | 3.0383522532 | +0.0084674309 | 163840 |
| 513-768 | 3.0125585693 | 3.0235968244 | +0.0110382551 | 327680 |
| 769-896 | 3.0069411320 | 3.0179441446 | +0.0110030126 | 163840 |
| 897-1023 | 3.0016298198 | 3.0147131121 | +0.0130832923 | 162560 |

### W128

| Position bin | Full loss | Short loss | Delta | Targets |
|---|---:|---:|---:|---:|
| 1-64 | 3.5649234841 | 3.5649136591 | -0.0000098250 | 81920 |
| 65-128 | 3.1827517307 | 3.1828938915 | +0.0001421608 | 81920 |
| 129-256 | 3.0944029219 | 3.1082970170 | +0.0138940950 | 163840 |
| 257-384 | 3.0448328851 | 3.0629379976 | +0.0181051125 | 163840 |
| 385-512 | 3.0298848223 | 3.0509488715 | +0.0210640492 | 163840 |
| 513-768 | 3.0125585693 | 3.0357550377 | +0.0231964684 | 327680 |
| 769-896 | 3.0069411320 | 3.0296157185 | +0.0226745865 | 163840 |
| 897-1023 | 3.0016298198 | 3.0270886916 | +0.0254588718 | 162560 |

### W1

| Position bin | Full loss | Short loss | Delta | Targets |
|---|---:|---:|---:|---:|
| 1-64 | 3.5649234841 | 3.8396351058 | +0.2747116217 | 81920 |
| 65-128 | 3.1827517307 | 3.4648186674 | +0.2820669366 | 81920 |
| 129-256 | 3.0944029219 | 3.3762413882 | +0.2818384663 | 163840 |
| 257-384 | 3.0448328851 | 3.3236423697 | +0.2788094845 | 163840 |
| 385-512 | 3.0298848223 | 3.3133067766 | +0.2834219543 | 163840 |
| 513-768 | 3.0125585693 | 3.2927913390 | +0.2802327697 | 327680 |
| 769-896 | 3.0069411320 | 3.2830811095 | +0.2761399775 | 163840 |
| 897-1023 | 3.0016298198 | 3.2770106962 | +0.2753808764 | 162560 |

Before a window's history-removal boundary, any tiny nonzero delta reflects numerical differences between the full causal and explicit sliding-mask kernels, not removed history.

## Representation and logit diagnostics

| Window | B11 norm ratio | B12 norm ratio | Logit MAE | Logit RMS | Argmax agreement |
|---:|---:|---:|---:|---:|---:|
| 384 | 1.0024511744 | 1.0042076427 | 0.0724749385 | 0.1447749616 | 0.9761505127 |
| 256 | 1.0044327668 | 1.0062618513 | 0.1066246159 | 0.1868649148 | 0.9642127991 |
| 128 | 1.0088404547 | 1.0100644321 | 0.1759374994 | 0.2684362947 | 0.9375885010 |
| 1 | 1.0436734035 | 1.0273983236 | 0.7899765479 | 1.0326323863 | 0.7563888550 |

Incoming h10 was bit-identical in every batch. Position-binned B11/B12, attention-output, MLP-output, and per-position loss diagnostics are retained in the machine-readable artifacts.

## Monotonicity

Damage was nondecreasing across 1024→896→768→512→384→256→128→1.

## Scientific questions

1. Damage versus full B11: W384 +0.0041603106, W256 +0.0076859903, W128 +0.0184321153, W1 +0.2790367706.
2. The curve is monotonic at measured resolution.
3. Smallest windows within the requested damage limits — +0.001: 768; +0.005: 384; +0.01: 256.
4. Damage is evaluated by absolute position. It begins at each removal boundary apart from very small kernel-path numerical deltas before that boundary; the bin tables quantify its later concentration.
5. B11 divergence rises as the window shrinks; the exact cosine, RMS, norm-ratio, and position-binned curves are reported above and in b11_state_drift.json.
6. B12 residual error — W384: B11 0.089624 → B12 0.099653; W256: B11 0.135498 → B12 0.150263; W128: B11 0.229586 → B12 0.253878; W1: B11 0.972690 → B12 1.086134. Lower B12 RMS is evidence of partial absorption; higher RMS is persistence/amplification.
7. At W1, validation loss is 3.3540805459, damage is +0.2790367706, and argmax agreement is 0.7563888550; this is the surviving end-to-end performance with zero B11 historical KV.
8. The result is evidence that a late layer can rely heavily on already-contextualized bottom-up residual state, but it is not proof: B1-B10 and B12 remained full-context.
9. It challenges a blanket assumption that KV width must monotonically increase toward the top, but it does not determine the final joint layerwise shape.
10. The measured endpoint and intermediate curve are sufficiently informative for the recommendation below, subject to the integrity audit.

## Causal interpretation

This experiment does not show that B11 needs no long-range information. It measures only the end-to-end damage caused by reducing B11's own direct attention history while B1-B10 and B12 remain full-context. Distant information may already be encoded in h10(t). Likewise, W1 is not automatically an optimal final setting: lower layers in a jointly KV-reduced model may deliver less-contextualized residual states.

## Integrity audit

- PASS — source ~10B checkpoint exact
- PASS — Standard GPT-2 exact
- PASS — Full AttnRes absent
- PASS — canonical validation data exact
- PASS — Phase-A W1024 regression sentinel
- PASS — Phase-A W512 sentinel if executed
- PASS — B1-B10 exactly W1024
- PASS — B12 exactly W1024
- PASS — only B11 window modified
- PASS — windows exactly 384/256/128/1
- PASS — absolute positions unchanged
- PASS — same evaluation precision
- PASS — same loss denominator
- PASS — all losses finite
- PASS — all activations finite
- PASS — incoming B11 h10 identical to full
- PASS — optimizer objects zero
- PASS — backward calls zero
- PASS — parameter updates zero
- PASS — training targets zero
- PASS — model tensors unchanged before/after
- PASS — no recurrence
- PASS — no completion module active
- PASS — no HellaSwag
- PASS — required machine-readable artifacts present
- PASS — one independent process per GPU; no DDP/NCCL

## Performance

| Window/GPU | Wall s | Targets | Targets/s | Peak allocated MB | Peak reserved MB |
|---|---:|---:|---:|---:|---:|
| W384/GPU0 | 46.782 | 1310720 | 28017.4 | 33869.0 | 41750.0 |
| W256/GPU1 | 48.969 | 1310720 | 26766.1 | 33869.0 | 41750.0 |
| W128/GPU2 | 49.161 | 1310720 | 26661.8 | 33869.0 | 41750.0 |
| W1/GPU3 | 46.831 | 1310720 | 27988.3 | 33869.0 | 41750.0 |

Total four-GPU elapsed wall time: 49.266 seconds.

## Next experiment

**PROCEED TO FULL LAYER×WINDOW SENSITIVITY MAP**

The recommendation was not executed.

# EXPERIMENT 2D0A COMPLETE
