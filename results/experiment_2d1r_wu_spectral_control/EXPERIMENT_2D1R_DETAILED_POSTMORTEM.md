# Experiment 2D1R — Detailed Postmortem

## Executive conclusion

**Final classification: W_U NORM CONTROL DOES NOT FULLY STABILIZE 2D1.** Spectral projection did exactly what it was intended to do locally: every one of the 954 completed Stage-C updates was projected, post-projection `W_u` stayed at the frozen C954 cap, Stage C completed through update 1908, and every saved R1000/R1100/R1200/R1908 32-pass trajectory remained bounded. It did not create recurrent utility. At C1908, real recurrence was `0.093616308` CE worse than the same checkpoint run plain and lost all 20 paired batches.

The recurrent pathway was not newly damaged by Stage-C training. It was already harmful at C954: gain was `-0.095765857` under native Stage-B geometry and `-1.386107447` under the prospective Stage-C geometry before update 955. Stage-C adaptation then repaired most of the prospective transition damage: by C1908 gain improved to `-0.093616308`, but never crossed zero.

Stage D remains causally unresolved. Update 1909 simultaneously changed `rho .75 → 1.0` and C12 → D12 windows. The first completed Stage-D row exceeded 10× Stage-A RMS; the terminal three-update streak was 1913, 1914, and attempted 1915. Because no saved C1908 counterfactual holds one factor fixed, attribution is **NOT IDENTIFIABLE FROM CURRENT RUN**.

Primary interpretation: **F. multifactor failure**. The cap fixed one global-scale mechanism; Stage C still learned a sequence-specific but predictively harmful recurrent pathway; and the confounded Stage-D transition then caused a separate hard-scale failure.

Recommended next experiment: **Experiment 2D1B — C1908 rho × window factorial diagnostic**, zero training. It was not started.

## Analysis-only boundary

This report was generated from already-produced JSON/JSONL metrics, validations, audits, and plots. It did not load a model checkpoint or execute model code.

```text
optimizer steps = 0
backward calls = 0
parameter updates = 0
training targets = 0
new training batches consumed = 0
new HellaSwag evaluations = 0
new result-bearing experiments = 0
GPU pods started = 0
```

The machine-readable companion is [`DETAILED_POSTMORTEM.json`](DETAILED_POSTMORTEM.json). Existing plots used as visual cross-checks were not regenerated: [`P1_recurrent_scale.png`](plots/P1_recurrent_scale.png), [`P2_wu_spectral_norm.png`](plots/P2_wu_spectral_norm.png), [`P3_training_ce.png`](plots/P3_training_ce.png), [`P4_projection_scale.png`](plots/P4_projection_scale.png), [`P5_self_composition.png`](plots/P5_self_composition.png), and [`P6_validation_trajectory.png`](plots/P6_validation_trajectory.png).

## 1. Stage-C training trajectory

`processed adaptation tokens/targets` is the additional count since C954; “tokens” and next-token training targets are the same count in this protocol. Each completed update contributed 524,288. `total targets` is the cumulative 2D1 count. C954 is the unchanged source boundary and therefore has zero 2D1R adaptation tokens. C12 is `[256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024]`; B12 is `[512, 545, 581, 618, 658, 702, 747, 796, 848, 903, 962, 1024]`. None of the specifically requested training rows was a scheduled three-pass update, so pass-3 CE is `NOT AVAILABLE` there. The full per-update trajectory (955–1914) remains in `training_metrics.jsonl`.

| update | processed adaptation tokens | total targets | stage/settings | rho | pass-1 CE | pass-2 CE | pass-3 CE | weighted CE |
|---|---|---|---|---|---|---|---|---|
| 954 | 0 | 500170752 | C boundary (native B settings) | 0.50 | 3.095486 | 3.119351 | NOT AVAILABLE | 3.113385 |
| 955 | 524288 | 500695040 | C | 0.75 | 3.150455 | 3.356542 | NOT AVAILABLE | 3.305020 |
| 956 | 1048576 | 501219328 | C | 0.75 | 3.101701 | 3.418199 | NOT AVAILABLE | 3.339075 |
| 1000 | 24117248 | 524288000 | C | 0.75 | 3.066848 | 3.110331 | NOT AVAILABLE | 3.099460 |
| 1100 | 76546048 | 576716800 | C | 0.75 | 2.916671 | 2.952523 | NOT AVAILABLE | 2.943560 |
| 1200 | 128974848 | 629145600 | C | 0.75 | 2.952358 | 2.983710 | NOT AVAILABLE | 2.975872 |
| 1500 | 286261248 | 786432000 | C | 0.75 | 3.174508 | 3.208255 | NOT AVAILABLE | 3.199818 |
| 1908 | 500170752 | 1000341504 | C | 0.75 | 3.106007 | 3.122042 | NOT AVAILABLE | 3.118033 |

Canonical validation used the same 20 batches at each saved milestone. Loss definitions are `gain = plain - real` and `sequence gap = shuffled - real`; positive is favorable to real recurrence. Zero/shuffled controls were saved only at C1908.

| point | rho | windows | plain | real | zero | shuffled | gain | seq. gap | real>plain wins | real>zero wins | real>shuffle wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C954 native B | .50 | B12 | 3.073581371 | 3.169347227 | NOT AVAILABLE | NOT AVAILABLE | -0.095765857 | NOT AVAILABLE | 0/20 | NOT AVAILABLE | NOT AVAILABLE |
| C954 prospective C (before 955) | .75 | C12 | 3.132144560 | 4.518252006 | NOT AVAILABLE | NOT AVAILABLE | -1.386107447 | NOT AVAILABLE | 0/20 | NOT AVAILABLE | NOT AVAILABLE |
| 955 | .75 | C12 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 956 | .75 | C12 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 1000 | .75 | C12 | 3.086890560 | 3.221241591 | NOT AVAILABLE | NOT AVAILABLE | -0.134351031 | NOT AVAILABLE | 0/20 | NOT AVAILABLE | NOT AVAILABLE |
| 1100 | .75 | C12 | 3.082405883 | 3.187101397 | NOT AVAILABLE | NOT AVAILABLE | -0.104695514 | NOT AVAILABLE | 0/20 | NOT AVAILABLE | NOT AVAILABLE |
| 1200 | .75 | C12 | 3.082470906 | 3.183410328 | NOT AVAILABLE | NOT AVAILABLE | -0.100939422 | NOT AVAILABLE | 0/20 | NOT AVAILABLE | NOT AVAILABLE |
| 1500 | .75 | C12 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 1908 | .75 | C12 | 3.081583431 | 3.175199739 | 9.785113238 | 3.368139969 | -0.093616308 | 0.192940230 | 0/20 | 20/20 | 20/20 |

### When did recurrent gain become negative?

It was already negative before Stage C. At C954 under the native Stage-B geometry, plain/real losses were `3.073581371` / `3.169347227`, for gain `-0.095765857` and 0/20 wins. Merely evaluating the unchanged C954 weights under the upcoming Stage-C windows and `rho=.75`, before update 955, changed plain/real to `3.132144560` / `4.518252006`, for gain `-1.386107447` and 0/20 wins.

The saved Stage-C gains then moved monotonically toward zero: `-0.134351031` at R1000, `-0.104695514` at R1100, `-0.100939422` at R1200, and `-0.093616308` at C1908. Thus Stage C did **not** progressively reverse a positive recurrent gain. It progressively repaired a large pre-existing negative gain, without achieving positive utility.

## 2. Stage-C adaptation effectiveness

### A. Did the underlying triangle Transformer improve?

Yes. Under prospective Stage-C geometry, plain-triangle validation improved from `3.132144560` at C954 to `3.081583431` at C1908, a change of `-0.050561129` CE. From the first saved rescued validation at R1000, it improved from `3.086890560` to `3.081583431` (`-0.005307129`).

### B. Did recurrence improve?

It improved but remained harmful. Real recurrent loss fell from `4.518252006` at the zero-update C954 Stage-C transition to `3.175199739` (`-1.343052267` CE). Gain improved by `1.292491138` CE over the same interval. From R1000 to C1908, real loss improved by `-0.046041852` and gain improved by `0.040734723`.

Therefore the specific hypothesis “plain improves while recurrent worsens” is not supported. The supported statement is: **plain improves; recurrence improves even more from the transition shock; recurrence nevertheless remains harmful relative to plain at every saved Stage-C validation.**

## 3. C1908 full control table

| model/control | absolute loss | delta vs parent | delta vs C1908 plain |
|---|---|---|---|
| parent_full | 3.075043775 | 0.000000000 | -0.006539656 |
| plain | 3.081583431 | 0.006539656 | 0.000000000 |
| real | 3.175199739 | 0.100155964 | 0.093616308 |
| zero | 9.785113238 | 6.710069463 | 6.703529807 |
| shuffled | 3.368139969 | 0.293096194 | 0.286556538 |

The reported `-0.093616308` recurrent gain means exactly:

```text
plain - real = 3.081583430867340 - 3.175199738963056
             = -0.093616308095716
```

Equivalently, real recurrence increased validation CE by `+0.093616308` relative to C1908 plain. It did so on all 20 paired batches.

The saved 20 paired differences below are `real - comparator`; negative favors real recurrence.

```json
{
  "real_minus_plain": {
    "definition": "C1908 real loss - C1908 plain loss; negative favors real",
    "losses": 20,
    "mean": 0.09361625909805298,
    "ties": 0,
    "values": [
      0.09228849411010742,
      0.09276294708251953,
      0.09061503410339355,
      0.09727358818054199,
      0.09608054161071777,
      0.08654284477233887,
      0.09554791450500488,
      0.08939123153686523,
      0.09367060661315918,
      0.0888359546661377,
      0.0860738754272461,
      0.10704612731933594,
      0.09296751022338867,
      0.09233736991882324,
      0.09070873260498047,
      0.09719514846801758,
      0.09370684623718262,
      0.10063862800598145,
      0.09733891487121582,
      0.09130287170410156
    ],
    "wins": 0
  },
  "real_minus_shuffled": {
    "definition": "C1908 real loss - C1908 shuffled loss; negative favors real",
    "losses": 0,
    "mean": -0.19294029474258423,
    "ties": 0,
    "values": [
      -0.2100369930267334,
      -0.21050524711608887,
      -0.17107677459716797,
      -0.20654010772705078,
      -0.204392671585083,
      -0.20296335220336914,
      -0.19637799263000488,
      -0.18278026580810547,
      -0.20970678329467773,
      -0.2001652717590332,
      -0.1727290153503418,
      -0.2028360366821289,
      -0.1920773983001709,
      -0.18347644805908203,
      -0.17190098762512207,
      -0.19379329681396484,
      -0.18337273597717285,
      -0.20880579948425293,
      -0.18708014488220215,
      -0.16818857192993164
    ],
    "wins": 20
  },
  "real_minus_zero": {
    "definition": "C1908 real loss - C1908 zero loss; negative favors real",
    "losses": 0,
    "mean": -6.609913444519043,
    "ties": 0,
    "values": [
      -6.599548816680908,
      -6.61736536026001,
      -6.648298501968384,
      -6.569318532943726,
      -6.665431261062622,
      -6.636035203933716,
      -6.495680570602417,
      -6.703655242919922,
      -6.640645503997803,
      -6.610563039779663,
      -6.518123626708984,
      -6.708176612854004,
      -6.629615783691406,
      -6.718468189239502,
      -6.540465831756592,
      -6.643067836761475,
      -6.640421152114868,
      -6.521752595901489,
      -6.483053684234619,
      -6.60858154296875
    ],
    "wins": 20
  }
}
```

Wins/losses/ties were real-vs-plain `0/20/0`, real-vs-zero `20/0/0`, and real-vs-shuffled `20/0/0`. Argmax agreement was **NOT AVAILABLE**; no saved 2D1R artifact contains it.

The control pattern is important: real recurrence is much better than zero (`6.609913499` CE) and shuffled (`0.192940230` CE), so the model depends on sequence-specific recurrent information; yet the same real state is worse than omitting recurrence through the plain path.

## 4. `W_u` spectral-control behavior

| update | cap | raw sigma | post sigma | proj. scale | pressure | projected | W_u Fro. | W_g Fro. |
|---|---|---|---|---|---|---|---|---|
| 954 | 1.026231766 | 1.026231766 | 1.026231766 | 1.000000000 | 0.000000000 | false | 1.391305089 | 43.670898438 |
| 955 | 1.026231766 | 1.028000236 | 1.026224017 | 0.998279699 | 0.001723266 | true | 1.391412258 | 43.684154510 |
| 956 | 1.026231766 | 1.027472734 | 1.026230097 | 0.998792213 | 0.001209248 | true | 1.392020464 | 43.698310852 |
| 1000 | 1.026231766 | 1.029070020 | 1.026222229 | 0.997241923 | 0.002765705 | true | 1.297036171 | 44.526649475 |
| 1100 | 1.026231766 | 1.028940320 | 1.026238203 | 0.997367627 | 0.002639320 | true | 1.213232756 | 45.742412567 |
| 1200 | 1.026231766 | 1.029337645 | 1.026167274 | 0.996920635 | 0.003026489 | true | 1.180373430 | 46.899715424 |
| 1500 | 1.026231766 | 1.029240251 | 1.026160121 | 0.997008601 | 0.002931584 | true | 1.156894565 | 50.260494232 |
| 1908 | 1.026231766 | 1.029759884 | 1.026196957 | 0.996504221 | 0.003437935 | true | 1.163712859 | 54.688358307 |

Stage-C aggregate:

- cap: `1.0262317657470703`
- projected updates: `954/954` (**100%**, far above 90%)
- mean projection scale: `0.997118330769`
- minimum projection scale: `0.995824518737`
- mean projection pressure: `0.002880474203`
- maximum projection pressure: `0.004192989010`
- maximum raw sigma: `1.030534744263`
- maximum post sigma: `1.026242017746`

**Did Adam continuously push `W_u` against the cap? Yes. Every Stage-C optimizer update required projection.** This conclusion is about the raw optimizer proposal, not about the final matrix norm.

The cap did not freeze the rest of the spectrum:

| update | W_u sigma | W_u Fro. | mean singular | 2nd singular | W_g sigma | W_g Fro. |
|---|---|---|---|---|---|---|
| 954 | 1.026231766 | 1.391305089 | 0.024894420 | 0.473175079 | 26.357809067 | 43.670898438 |
| 1000 | 1.026222229 | 1.297036171 | 0.020268612 | 0.446014971 | 26.639717102 | 44.526649475 |
| 1100 | 1.026238203 | 1.213232756 | 0.017053332 | 0.360294104 | 26.990230560 | 45.742412567 |
| 1200 | 1.026167274 | 1.180373430 | 0.015880831 | 0.303574950 | 27.306610107 | 46.899715424 |
| 1908 | 1.026196957 | 1.163712859 | 0.015771255 | 0.162873626 | 29.516586304 | 54.688358307 |

The leading `W_u` singular value stayed fixed, but Frobenius norm changed from `1.391305089` to `1.163712859` (`-16.36%`), mean singular value from `0.024894420` to `0.015771255` (`-36.65%`), and the second singular value from `0.473175079` to `0.162873626` (`-65.58%`). The saved leading-ten trajectory is:

| update | largest ten W_u singular values |
|---|---|
| 954 | 1.026232, 0.473175, 0.174363, 0.158310, 0.119138, 0.098448, 0.094721, 0.083866, 0.081448, 0.076446 |
| 1000 | 1.026222, 0.446015, 0.154149, 0.130522, 0.100985, 0.081673, 0.076108, 0.070312, 0.066243, 0.065028 |
| 1100 | 1.026238, 0.360294, 0.115016, 0.104444, 0.082442, 0.061865, 0.058412, 0.056262, 0.053766, 0.052937 |
| 1200 | 1.026167, 0.303575, 0.099111, 0.095865, 0.076020, 0.056527, 0.051019, 0.048592, 0.047235, 0.045876 |
| 1908 | 1.026197, 0.162874, 0.151539, 0.076902, 0.064856, 0.060364, 0.053867, 0.053814, 0.050615, 0.049608 |

This is material redistribution/compression despite a fixed top singular value.

## 5. What learned instead of `W_u` scale?

Canonical validation-state trajectory (C954 is the prospective Stage-C evaluation). Ratios are arithmetic ratios of saved RMS values, not new activation measurements.

| update | E RMS | F RMS | F/E | X RMS | X/E | top RMS | gate mean | gate var. | gate sat. |
|---|---|---|---|---|---|---|---|---|---|
| 954 | 0.035594912 | 0.302927877 | 8.510426291 | 0.227742977 | 6.398189041 | 2.272286642 | 0.950556812 | 0.005613089 | 0.000000000 |
| 1000 | 0.035617718 | 0.359936108 | 10.105535198 | 0.270431732 | 7.592618035 | 2.309320772 | 0.949607870 | 0.005625899 | 0.000000000 |
| 1100 | 0.035633487 | 0.375052768 | 10.525289630 | 0.281764962 | 7.907308207 | 2.319584048 | 0.948511484 | 0.005587745 | 0.000000000 |
| 1200 | 0.035643132 | 0.391011670 | 10.970182696 | 0.293727410 | 8.240785619 | 2.320459199 | 0.947559983 | 0.005600389 | 0.000000000 |
| 1908 | 0.035749214 | 0.427279490 | 11.952136585 | 0.321050894 | 8.980642007 | 2.346248817 | 0.941137642 | 0.006225559 | 0.000000000 |

From C954 to C1908, `F` RMS rose `+41.05%`, recurrent-input `X` RMS rose `+40.97%`, and top-state RMS rose only `+3.25%`. In parallel, unconstrained `W_g` Frobenius norm rose `+25.23%` and spectral norm rose `+11.98%`. Gate mean did not increase—it fell from `0.950556812` to `0.941137642`—and saturation remained exactly zero. Gate variance increased modestly.

Saved batch-conditioned decomposition metrics provide `ZN`, `U`, and `X`; they are not directly comparable as a smooth trajectory because prefix coverage varies by training batch (notably C1908’s saved training row has recurrent fraction zero and therefore `X/E=1`).

| update | ZN | U | U/ZN | G RMS | F | X | X/E | source/qualification |
|---|---|---|---|---|---|---|---|---|
| 954 | 0.999510601 | 0.343487352 | 0.343655537 | 0.953586072 | 0.333139889 | 0.250367194 | 7.033352692 | 2D1A COMMON-C pass 3 mean across four saved batches |
| 1000 | 0.999510646 | 0.379488945 | 0.379674740 | NOT AVAILABLE | 0.367409885 | 0.265770316 | 7.476206969 | 2D1R saved training-batch scale diagnostic |
| 1100 | 0.999510646 | 0.392605960 | 0.392798178 | NOT AVAILABLE | 0.379028052 | 0.281553119 | 7.920393272 | 2D1R saved training-batch scale diagnostic |
| 1200 | 0.999510705 | 0.390414715 | 0.390605836 | NOT AVAILABLE | 0.375586599 | 0.168204352 | 4.700506445 | 2D1R saved training-batch scale diagnostic |
| 1908 | 0.999510646 | 0.456837684 | 0.457061349 | NOT AVAILABLE | 0.437998325 | 0.035688069 | 1.000000000 | 2D1R saved training-batch scale diagnostic |

The defensible conclusion is that amplification migrated into the *effective fused path* and coincided with substantial `W_g` growth and continued base/fusion representation learning. The saved data do **not** isolate one replacement amplifier: mean gating weakened slightly, `W_u` non-leading spectrum contracted, and no 2D1R layerwise activation trajectory was saved. So “another single component took over” is not established.

Only the C954 layerwise forensic baseline was saved (2D1A COMMON-C pass 2, mean across four batches):

| location | RMS |
|---|---|
| input_B1 | 0.228930183 |
| B1_post_attention | 0.216378994 |
| B1_post_mlp | 0.679117605 |
| B2_post_attention | 0.780799344 |
| B2_post_mlp | 0.797967136 |
| B3_post_attention | 0.784290120 |
| B3_post_mlp | 0.816762641 |
| B4_post_attention | 0.835354701 |
| B4_post_mlp | 1.011939049 |
| B5_post_attention | 1.042348295 |
| B5_post_mlp | 1.088422418 |
| B6_post_attention | 1.128632724 |
| B6_post_mlp | 1.158002108 |
| B7_post_attention | 1.233976036 |
| B7_post_mlp | 1.215992719 |
| B8_post_attention | 1.286383867 |
| B8_post_mlp | 1.312665313 |
| B9_post_attention | 1.479305744 |
| B9_post_mlp | 1.569393516 |
| B10_post_attention | 1.670041382 |
| B10_post_mlp | 1.762765199 |
| B11_post_attention | 2.065519631 |
| B11_post_mlp | 2.087594688 |
| B12_post_attention | 2.207368076 |
| B12_post_mlp | 2.392451704 |
| final_ln_f | 2.275035501 |

The requested R1100/R1200/R1908 B1 input, post-attention, post-MLP, and layerwise residual values are **NOT AVAILABLE**. Original-lineage 2D1A values at 1100 exist but are not substituted for rescued 2D1R values.

## 6. Recurrent representation content

At C1908, saved real-state diagnostics were:

- `cosine(z_t, z_(t-1))` / top temporal cosine: `0.571766027808`
- top-state RMS: `2.346248817444`
- top-state norm mean: `65.009086990356`
- top-state norm standard deviation: `1.251796269417`
- recurrent-state variance across rows: **NOT AVAILABLE**
- recurrent-state variance across token positions: **NOT AVAILABLE**
- representation-level real/shuffled/zero cosine, norm, and variance: **NOT AVAILABLE**

Loss controls provide indirect but strong content evidence: real beats shuffled by `0.192940230` CE with 20/20 wins and beats zero by `6.609913499` CE with 20/20 wins. That rules against treating the pathway as wholly generic or functionally constant. At the same time real loses to plain by `0.093616308` on 20/20 batches. The data-consistent description is **sequence-informative but predictively harmful in the way it is consumed**. “Incorrectly consumed” is an interpretation of the loss contrast, not a direct representation probe.

## 7. Repeated self-composition

All requested checkpoints have saved 32-pass diagnostics. Each table is the passwise mean of two saved batches; the machine-readable file retains all 32 values. The source classified all four `OSCILLATORY` and `native_scale_stable`; mapped to the requested vocabulary, all four are **bounded oscillatory**, not expansive.

### R1000

Classification: **BOUNDED OSCILLATORY**. Maximum saved recurrent-input RMS: `0.272383272648`; native-scale-stable: `true`. Values below are arithmetic means of the two saved batches at each pass.

| pass | CE | recurrent-input RMS | top-state RMS | state-change RMS | state cosine |
|---|---|---|---|---|---|
| 1 | 3.031209230 | 0.035541125 | 2.306681275 | NOT AVAILABLE | NOT AVAILABLE |
| 2 | 3.162939072 | 0.271616876 | 2.311536312 | 0.712403357 | 0.952739626 |
| 3 | 3.202793479 | 0.260158777 | 2.311380863 | 0.289049372 | 0.992261767 |
| 4 | 3.213588715 | 0.260191873 | 2.311110258 | 0.153790839 | 0.997821510 |
| 5 | 3.217054486 | 0.260417610 | 2.310961604 | 0.105440736 | 0.998977840 |
| 6 | 3.217609406 | 0.260533005 | 2.310900331 | 0.087653391 | 0.999292254 |
| 7 | 3.217916727 | 0.260571361 | 2.310876012 | 0.080024809 | 0.999408543 |
| 8 | 3.217778206 | 0.260573894 | 2.310872674 | 0.076457102 | 0.999459356 |
| 9 | 3.218201756 | 0.260595605 | 2.310870171 | 0.074914735 | 0.999480546 |
| 10 | 3.217680335 | 0.260580987 | 2.310881138 | 0.074266657 | 0.999489278 |
| 11 | 3.218306303 | 0.260578662 | 2.310867906 | 0.073744938 | 0.999496251 |
| 12 | 3.217608690 | 0.260600418 | 2.310877681 | 0.073075358 | 0.999505252 |
| 13 | 3.217971802 | 0.260598987 | 2.310864687 | 0.073160838 | 0.999504298 |
| 14 | 3.218006372 | 0.260616496 | 2.310878158 | 0.073205426 | 0.999503553 |
| 15 | 3.217919111 | 0.260588601 | 2.310886025 | 0.073483329 | 0.999499798 |
| 16 | 3.218223333 | 0.260584772 | 2.310872197 | 0.073534537 | 0.999498874 |
| 17 | 3.218140006 | 0.260596514 | 2.310875654 | 0.073555991 | 0.999498695 |
| 18 | 3.217945814 | 0.260595441 | 2.310884595 | 0.073224258 | 0.999503195 |
| 19 | 3.217925072 | 0.260589436 | 2.310906768 | 0.072932262 | 0.999507040 |
| 20 | 3.218237162 | 0.260563314 | 2.310867667 | 0.072898675 | 0.999507606 |
| 21 | 3.218050599 | 0.260603249 | 2.310877919 | 0.072701212 | 0.999509990 |
| 22 | 3.218063116 | 0.260593936 | 2.310890436 | 0.072793748 | 0.999508977 |
| 23 | 3.218078971 | 0.260575414 | 2.310866117 | 0.072905440 | 0.999507427 |
| 24 | 3.218266010 | 0.260589406 | 2.310894728 | 0.072813742 | 0.999508798 |
| 25 | 3.217745185 | 0.260597825 | 2.310887933 | 0.072582964 | 0.999511957 |
| 26 | 3.218476772 | 0.260588169 | 2.310877919 | 0.072594441 | 0.999511838 |
| 27 | 3.218008995 | 0.260591060 | 2.310859680 | 0.072715864 | 0.999510080 |
| 28 | 3.218001842 | 0.260602742 | 2.310880065 | 0.072597783 | 0.999511719 |
| 29 | 3.218044996 | 0.260584190 | 2.310888171 | 0.072654199 | 0.999510884 |
| 30 | 3.218352795 | 0.260586411 | 2.310874701 | 0.072459389 | 0.999513507 |
| 31 | 3.218129396 | 0.260604516 | 2.310873389 | 0.072155781 | 0.999517441 |
| 32 | 3.218136311 | 0.260591462 | 2.310892105 | 0.071922719 | 0.999520659 |

### R1100

Classification: **BOUNDED OSCILLATORY**. Maximum saved recurrent-input RMS: `0.283161908388`; native-scale-stable: `true`. Values below are arithmetic means of the two saved batches at each pass.

| pass | CE | recurrent-input RMS | top-state RMS | state-change RMS | state cosine |
|---|---|---|---|---|---|
| 1 | 3.027335882 | 0.035556940 | 2.317562342 | NOT AVAILABLE | NOT AVAILABLE |
| 2 | 3.127326965 | 0.282407284 | 2.322370410 | 0.650154471 | 0.960958749 |
| 3 | 3.154407263 | 0.270426124 | 2.322494388 | 0.234233275 | 0.994952828 |
| 4 | 3.159341931 | 0.270252541 | 2.322366595 | 0.117001291 | 0.998742640 |
| 5 | 3.160570741 | 0.270362854 | 2.322322011 | 0.085286029 | 0.999331772 |
| 6 | 3.161613464 | 0.270383164 | 2.322302222 | 0.078580301 | 0.999432981 |
| 7 | 3.161888719 | 0.270406350 | 2.322316051 | 0.076449357 | 0.999463767 |
| 8 | 3.161895633 | 0.270396769 | 2.322315097 | 0.075547736 | 0.999476373 |
| 9 | 3.162172675 | 0.270422012 | 2.322314382 | 0.074131224 | 0.999495536 |
| 10 | 3.161813498 | 0.270413920 | 2.322289467 | 0.073231976 | 0.999507368 |
| 11 | 3.162399411 | 0.270437360 | 2.322286487 | 0.072651073 | 0.999514937 |
| 12 | 3.162227273 | 0.270438164 | 2.322287440 | 0.072574116 | 0.999516040 |
| 13 | 3.162277937 | 0.270443454 | 2.322270751 | 0.072349038 | 0.999518871 |
| 14 | 3.162231922 | 0.270444974 | 2.322290421 | 0.072423954 | 0.999517918 |
| 15 | 3.162226200 | 0.270434976 | 2.322282791 | 0.072872084 | 0.999512017 |
| 16 | 3.162385821 | 0.270417675 | 2.322283030 | 0.072822932 | 0.999512613 |
| 17 | 3.162287951 | 0.270433813 | 2.322294235 | 0.071848132 | 0.999525547 |
| 18 | 3.162296057 | 0.270411775 | 2.322284579 | 0.071552165 | 0.999529421 |
| 19 | 3.162512302 | 0.270427018 | 2.322287679 | 0.071791351 | 0.999526352 |
| 20 | 3.162220478 | 0.270423710 | 2.322275639 | 0.072004069 | 0.999523640 |
| 21 | 3.162745118 | 0.270445168 | 2.322298169 | 0.072039600 | 0.999522984 |
| 22 | 3.162127733 | 0.270420015 | 2.322284698 | 0.072147284 | 0.999521673 |
| 23 | 3.162319422 | 0.270438120 | 2.322293162 | 0.072381638 | 0.999518603 |
| 24 | 3.162291050 | 0.270415947 | 2.322291970 | 0.072090834 | 0.999522537 |
| 25 | 3.162019730 | 0.270428404 | 2.322293758 | 0.071447182 | 0.999531120 |
| 26 | 3.162369490 | 0.270419478 | 2.322305202 | 0.071656439 | 0.999528348 |
| 27 | 3.162308097 | 0.270409524 | 2.322271705 | 0.072209176 | 0.999521017 |
| 28 | 3.162238359 | 0.270442396 | 2.322287917 | 0.072205715 | 0.999521106 |
| 29 | 3.162386298 | 0.270419613 | 2.322293639 | 0.072163302 | 0.999521643 |
| 30 | 3.162296176 | 0.270439640 | 2.322283506 | 0.072299212 | 0.999519736 |
| 31 | 3.162368894 | 0.270442501 | 2.322293282 | 0.071865931 | 0.999525666 |
| 32 | 3.162050724 | 0.270435125 | 2.322289109 | 0.072056275 | 0.999523103 |

### R1200

Classification: **BOUNDED OSCILLATORY**. Maximum saved recurrent-input RMS: `0.294723838568`; native-scale-stable: `true`. Values below are arithmetic means of the two saved batches at each pass.

| pass | CE | recurrent-input RMS | top-state RMS | state-change RMS | state cosine |
|---|---|---|---|---|---|
| 1 | 3.027159452 | 0.035566546 | 2.321474314 | NOT AVAILABLE | NOT AVAILABLE |
| 2 | 3.125386477 | 0.294006243 | 2.322947741 | 0.636736184 | 0.962607414 |
| 3 | 3.150613308 | 0.281629458 | 2.322687149 | 0.220556997 | 0.995524347 |
| 4 | 3.156959891 | 0.281091481 | 2.322453499 | 0.108317837 | 0.998921335 |
| 5 | 3.158307552 | 0.281169593 | 2.322362542 | 0.080860130 | 0.999399036 |
| 6 | 3.158921599 | 0.281213179 | 2.322328568 | 0.073796831 | 0.999499381 |
| 7 | 3.159212232 | 0.281212822 | 2.322307229 | 0.072303023 | 0.999519527 |
| 8 | 3.159203410 | 0.281214833 | 2.322327018 | 0.072071880 | 0.999522597 |
| 9 | 3.159428477 | 0.281186894 | 2.322304487 | 0.071510755 | 0.999530196 |
| 10 | 3.159501433 | 0.281204402 | 2.322311282 | 0.070843272 | 0.999538809 |
| 11 | 3.159239292 | 0.281205684 | 2.322300434 | 0.070984393 | 0.999536961 |
| 12 | 3.159481287 | 0.281221554 | 2.322297573 | 0.071409401 | 0.999531448 |
| 13 | 3.159191966 | 0.281224385 | 2.322291136 | 0.071073949 | 0.999535799 |
| 14 | 3.159497857 | 0.281234458 | 2.322301507 | 0.070510425 | 0.999542981 |
| 15 | 3.159272432 | 0.281228870 | 2.322300911 | 0.070727140 | 0.999540269 |
| 16 | 3.159451246 | 0.281209603 | 2.322314143 | 0.071213774 | 0.999534130 |
| 17 | 3.159224510 | 0.281210706 | 2.322317839 | 0.071144812 | 0.999535084 |
| 18 | 3.159343362 | 0.281208143 | 2.322294950 | 0.070637286 | 0.999541432 |
| 19 | 3.159336925 | 0.281221211 | 2.322302818 | 0.071408246 | 0.999531358 |
| 20 | 3.159450293 | 0.281215400 | 2.322320104 | 0.071523495 | 0.999530077 |
| 21 | 3.159237146 | 0.281182438 | 2.322312832 | 0.070488818 | 0.999543548 |
| 22 | 3.159559965 | 0.281205043 | 2.322298408 | 0.070083108 | 0.999548733 |
| 23 | 3.159425855 | 0.281239927 | 2.322296500 | 0.070424762 | 0.999544263 |
| 24 | 3.159465432 | 0.281230733 | 2.322306156 | 0.070643049 | 0.999541491 |
| 25 | 3.159525275 | 0.281210944 | 2.322323442 | 0.070324752 | 0.999545664 |
| 26 | 3.159392834 | 0.281196177 | 2.322294831 | 0.070539638 | 0.999542713 |
| 27 | 3.159624934 | 0.281219274 | 2.322299242 | 0.070701305 | 0.999540746 |
| 28 | 3.159358859 | 0.281220838 | 2.322294354 | 0.070958346 | 0.999537498 |
| 29 | 3.159387708 | 0.281218842 | 2.322306395 | 0.070536200 | 0.999542892 |
| 30 | 3.159327745 | 0.281220004 | 2.322292805 | 0.070790324 | 0.999539673 |
| 31 | 3.159615040 | 0.281230822 | 2.322309136 | 0.070003536 | 0.999549687 |
| 32 | 3.159314632 | 0.281230360 | 2.322301865 | 0.070328712 | 0.999545455 |

### R1908

Classification: **BOUNDED OSCILLATORY**. Maximum saved recurrent-input RMS: `0.322392284870`; native-scale-stable: `true`. Values below are arithmetic means of the two saved batches at each pass.

| pass | CE | recurrent-input RMS | top-state RMS | state-change RMS | state cosine |
|---|---|---|---|---|---|
| 1 | 3.026932478 | 0.035673233 | 2.348804355 | NOT AVAILABLE | NOT AVAILABLE |
| 2 | 3.119458199 | 0.321329713 | 2.348145723 | 0.626508564 | 0.964581400 |
| 3 | 3.163099170 | 0.299911246 | 2.347257614 | 0.273633644 | 0.993258893 |
| 4 | 3.170535088 | 0.299557105 | 2.347089529 | 0.142511792 | 0.998174369 |
| 5 | 3.172030926 | 0.299636692 | 2.347035766 | 0.103282787 | 0.999041319 |
| 6 | 3.172350049 | 0.299701810 | 2.347062349 | 0.091589745 | 0.999246061 |
| 7 | 3.173248410 | 0.299683869 | 2.347032070 | 0.086094186 | 0.999333411 |
| 8 | 3.173464894 | 0.299732640 | 2.347031713 | 0.086269245 | 0.999330491 |
| 9 | 3.173210859 | 0.299717352 | 2.347020507 | 0.085853297 | 0.999336421 |
| 10 | 3.172781229 | 0.299714565 | 2.347073913 | 0.083343435 | 0.999373823 |
| 11 | 3.173014402 | 0.299671873 | 2.347026467 | 0.083367925 | 0.999373645 |
| 12 | 3.173305511 | 0.299738646 | 2.347038031 | 0.083271757 | 0.999375522 |
| 13 | 3.173182011 | 0.299710929 | 2.347066760 | 0.082700208 | 0.999384403 |
| 14 | 3.172662735 | 0.299691424 | 2.347078681 | 0.082438551 | 0.999387592 |
| 15 | 3.172783732 | 0.299656257 | 2.347032785 | 0.083302993 | 0.999374539 |
| 16 | 3.172711611 | 0.299717948 | 2.347038984 | 0.083536480 | 0.999371320 |
| 17 | 3.173153520 | 0.299708799 | 2.347047806 | 0.083783399 | 0.999368310 |
| 18 | 3.173297882 | 0.299697995 | 2.347026706 | 0.085425757 | 0.999342799 |
| 19 | 3.172773600 | 0.299726009 | 2.347065330 | 0.084933154 | 0.999350160 |
| 20 | 3.172919750 | 0.299688011 | 2.347025871 | 0.083902035 | 0.999366432 |
| 21 | 3.173095226 | 0.299726278 | 2.347033978 | 0.084054191 | 0.999363124 |
| 22 | 3.172851324 | 0.299711287 | 2.347046852 | 0.082261737 | 0.999390721 |
| 23 | 3.172611594 | 0.299707174 | 2.347033978 | 0.083479822 | 0.999372184 |
| 24 | 3.173338413 | 0.299716905 | 2.347018480 | 0.082962647 | 0.999380082 |
| 25 | 3.172698736 | 0.299738556 | 2.347067714 | 0.081442114 | 0.999402225 |
| 26 | 3.172904253 | 0.299681202 | 2.347070098 | 0.080676701 | 0.999413848 |
| 27 | 3.173175216 | 0.299673334 | 2.347036839 | 0.082441557 | 0.999387115 |
| 28 | 3.173321128 | 0.299735367 | 2.347021818 | 0.083600294 | 0.999369949 |
| 29 | 3.173296452 | 0.299748018 | 2.347015142 | 0.082196184 | 0.999391764 |
| 30 | 3.172975302 | 0.299741358 | 2.347019196 | 0.082481157 | 0.999387085 |
| 31 | 3.172873735 | 0.299740940 | 2.347046137 | 0.083649982 | 0.999370426 |
| 32 | 3.173009992 | 0.299723223 | 2.347057939 | 0.082207195 | 0.999391437 |

Yes: recurrence became dynamically stable in Stage C while remaining predictively harmful. Maximum 32-pass RMS rose from `0.272383273` at R1000 to `0.322392285` at C1908, but remained below the hard threshold `0.355099630`; meanwhile every canonical saved gain remained negative and every real-vs-plain paired result was 0/20.

## 8. Original 2D1 vs rescued 2D1R

| update | lineage | training CE | plain val. | real val. | gain | W_u sigma | U/ZN | X/E | recurrent-input RMS |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | original_2D1 | 3.098132789 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | 1.146052003 | 0.393084517 | 8.022902321 | 0.294510692 |
| 1000 | 2D1R | 3.099460483 | 3.086890560 | 3.221241591 | -0.134351031 | 1.026222229 | 0.379674740 | 7.476206969 | 0.265738279 |
| 1100 | original_2D1 | 2.941644281 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | 1.372432709 | 0.495587969 | 10.067053811 | 0.373017728 |
| 1100 | 2D1R | 2.943560064 | 3.082405883 | 3.187101397 | -0.104695514 | 1.026238203 | 0.392798178 | 7.920393272 | 0.281516135 |

Original 2D1 canonical validation at 1000/1100 was not saved, so those cells are **NOT AVAILABLE**. Spectral control fixed the runaway top singular value: at 1000 it reduced `1.146052003 → 1.026222229`, and at 1100 `1.372432709 → 1.026238203`. At 1100 it also reduced `U/ZN` by `0.102789791`, `X/E` by `2.146660539`, and recurrent-input RMS by `0.091501594`, while matching training CE within `+0.001915783`.

It did not fix recurrent utility (2D1R gain remained `-0.104695514` at 1100 and `-0.093616308` at C1908), did not prevent continuous optimizer pressure against the cap, and did not survive the joint Stage-D rho/window transition.

## 9. Stage-D transition forensic table

D12 is `[128, 154, 187, 225, 272, 330, 398, 481, 581, 702, 848, 1024]`. The hard threshold is `0.355099629611`. Update 1915 was stopped before an optimizer step, so no training loss, projection row, gradient, or detailed fusion decomposition exists for it.

| update | status | weighted loss | pass 1 | pass 2 | rho | windows | recurrent RMS | Stage-A ratio | consecutive |
|---|---|---|---|---|---|---|---|---|---|
| 1909 | completed and persisted | 3.254374713 | 3.053598881 | 3.321300000 | 1.00 | D12 | 0.409732372 | 11.538518697 | 1 |
| 1910 | completed and persisted | 3.230479896 | 3.147728056 | 3.258063853 | 1.00 | D12 | 0.257316679 | 7.246323491 | 0 |
| 1911 | completed and persisted | 3.192355961 | 3.121944815 | 3.215826303 | 1.00 | D12 | 0.378353298 | 10.654849123 | 1 |
| 1912 | completed and persisted | 3.223985672 | 3.166736841 | 3.243068665 | 1.00 | D12 | 0.282339573 | 7.950995984 | 0 |
| 1913 | completed and persisted | 3.189263612 | 3.126955241 | 3.210033178 | 1.00 | D12 | 0.429369092 | 12.091510556 | 1 |
| 1914 | completed and persisted | 3.157887220 | 3.106555462 | 3.174997747 | 1.00 | D12 | 0.397121400 | 11.183379720 | 2 |
| 1915 | attempted; hard-stopped before optimizer step; not persisted as a completed update | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | 1.00 | D12 | 0.404470712 | 11.390344518 | 3 |

| update | top RMS | ZN | U | G | F | X | X/E |
|---|---|---|---|---|---|---|---|
| 1909 | 2.338106871 | 0.999510646 | 0.450123727 | NOT AVAILABLE | 0.431824684 | 0.409771025 | 11.456163127 |
| 1910 | 2.341558933 | 0.999510646 | 0.447585523 | NOT AVAILABLE | 0.428746492 | 0.257342070 | 7.147374923 |
| 1911 | 2.341004610 | 0.999510646 | 0.460016757 | NOT AVAILABLE | 0.441212565 | 0.378392339 | 10.579824282 |
| 1912 | 2.339758158 | 0.999510646 | 0.455744445 | NOT AVAILABLE | 0.437404692 | 0.282365352 | 7.887594435 |
| 1913 | 2.342925787 | 0.999510646 | 0.465621918 | NOT AVAILABLE | 0.446012944 | 0.429414481 | 11.953510858 |
| 1914 | 2.341138840 | 0.999510705 | 0.465482444 | NOT AVAILABLE | 0.446785867 | 0.397165358 | 11.057681244 |
| 1915 | 2.332999229 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |

| update | raw sigma | post sigma | proj. scale | W_g Fro. | W_g sigma | gate mean | gate std | gate var. | gate sat. | grad norm | LR base/fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1909 | 1.030160427 | 1.026157022 | 0.996116648 | 54.700771332 | NOT AVAILABLE | 0.941353977 | 0.078539275 | 0.006168418 | 0.000000000 | 15.726380348 | 3.0e-05/3.0e-04 |
| 1910 | 1.030196548 | 1.026155829 | 0.996084384 | 54.714454651 | 29.522735596 | 0.939534545 | 0.079148032 | 0.006264411 | 0.000000000 | 4.604392052 | 3.0e-05/3.0e-04 |
| 1911 | 1.030430436 | 1.026226878 | 0.995925324 | 54.728366852 | NOT AVAILABLE | 0.940993786 | 0.078797542 | 0.006209053 | 0.000000000 | 4.100257874 | 3.0e-05/3.0e-04 |
| 1912 | 1.030626893 | 1.026223063 | 0.995735482 | 54.742546082 | NOT AVAILABLE | 0.941777706 | 0.078776233 | 0.006205695 | 0.000000000 | 2.994020939 | 3.0e-05/3.0e-04 |
| 1913 | 1.030565262 | 1.026223063 | 0.995795030 | 54.757568359 | NOT AVAILABLE | 0.940010548 | 0.079387456 | 0.006302368 | 0.000000000 | 3.537743568 | 3.0e-05/3.0e-04 |
| 1914 | 1.031472921 | 1.026221275 | 0.994918766 | 54.773536682 | NOT AVAILABLE | 0.942585230 | 0.079587914 | 0.006334236 | 0.000000000 | 2.814162493 | 3.0e-05/3.0e-04 |
| 1915 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |

`G` RMS was not logged in 2D1R; gate mean/std/saturation are the available gate statistics. The first 10× crossing occurred at completed update 1909 (`11.538518697×`), but 1910 reset the consecutive counter; 1911 crossed and 1912 reset it again. The terminal consecutive streak was exactly:

```text
1913: 0.429369091988 RMS = 12.091510556×  (consecutive 1)
1914: 0.397121399641 RMS = 11.183379720×  (consecutive 2)
1915: 0.404470711946 RMS = 11.390344518×  (consecutive 3; hard stop)
```

No failed update was persisted. The last completed model state is update 1914.

## 10. Separating the two Stage-D changes

**NOT IDENTIFIABLE FROM CURRENT RUN.** At update 1909, C12 windows/`rho=.75` changed together to D12 windows/`rho=1.0`; there is no saved C1908 evaluation for C12/1.0 or D12/.75.

There is circumstantial earlier evidence against `rho=1`: at unchanged C954 weights, the saved `real_final_rho` loss was `8.696376695` under B12 windows and `8.473870945` under C12 windows. This shows final-rho behavior was already poor across two earlier window sets, but it is not a C1908 causal separation and cannot attribute the update-1909 scale failure.

The minimal later diagnostic is the zero-training C1908 2×2 factorial in section 12. It was not run here.

## 11. Direct-embedding pathway issue

At C1908/`rho=.75`, canonical saved RMS gives `F/E = 11.952136585` and `X/E = 8.980642007`. Gate mean/std/saturation were `0.941137642` / `0.078902209` / `0.000000000`. `F-vs-E` cosine and `X-vs-E` cosine are **NOT AVAILABLE**.

The existing evidence supports the *possibility* that the multiplicative GLU had not learned a sufficient token-content carrier by `rho=1`, because Stage D removes the additive `.25E` term, C954 final-rho controls were catastrophic, and the Stage-D scale stop occurred immediately after the joint transition. It does not prove that claim: token identity still conditions `G = 2*sigmoid(W_g(E))`, real recurrence is sequence-specific, and the missing cosine/content probes prevent measuring how much token identity resides in `F`.

## 12. Window-vs-rho next diagnostic recommendation

Yes. The next diagnostic should be the zero-training C1908 factorial:

| condition | windows | rho | purpose |
|---|---|---:|---|
| A | Stage C | .75 | native baseline |
| B | Stage C | 1.0 | isolated rho effect |
| C | Stage D | .75 | isolated window effect |
| D | Stage D | 1.0 | observed joint setting |

This is the smallest clean test of rho main effect, window main effect, and interaction. It should reuse identical batches and report plain/real/zero/shuffled loss, scale decomposition, and 32-pass stability. It was explicitly **not run** for this report.

## 13. Should the fusion equation change?

The Full-Bandwidth GLU successfully avoids a narrow state bottleneck: `U` and `G` are width 768, and token embeddings modulate every fused dimension. But at `rho=1`, token identity is only multiplicative context for the recurrent value; it is no longer an additive content path. The observed model had gate mean near 0.94 with zero saturation, a large sequence-specific gap, and still worse-than-plain predictions. This says the GLU carried and used information, not that it preserved the right current-token content.

Current evidence therefore justifies testing—after the factorial diagnosis—an architecture where current-token embedding information remains mandatory, conceptually `X = controlled recurrent contribution + E` or a gate in which neither `E` nor recurrence can be eliminated. It does **not** yet justify selecting or implementing one equation: the confounded Stage-D transition must first establish whether `rho=1` itself is the problem.

## 14. Was the spectral cap itself too restrictive?

Classification: **INCONCLUSIVE**.

- Projection bound continuously: 954/954 Stage-C updates and 6/6 completed Stage-D updates.
- The optimizer repeatedly proposed a larger `W_u` top singular value; maximum raw Stage-C sigma was `1.030534744`.
- Recurrent gain did **not** become more negative as saved milestone projection pressure drifted upward. It improved from `-0.134351031` at R1000 to `-0.093616308` at C1908.
- Plain and real validation both improved; real improved more, but never overtook plain.
- `W_u` continued changing substantially below the top singular direction, while `W_g` grew.
- The original uncapped matched lineage had similar training CE but larger scale, so removing the cap is not supported as a safe remedy.

Continuous binding is evidence that Adam wanted more top-direction scale, but there is no controlled counterfactual showing that the cap prevented useful information transmission. Conversely, the Stage-D hard stop despite the cap shows that top-norm control alone is insufficient. Hence neither “cap likely too restrictive” nor “cap definitively appropriate for utility” follows.

## 15. Most important scientific interpretation

**F. multifactor failure.** Quantitatively:

1. `W_u` instability was controlled through 954 Stage-C updates; all saved 32-pass probes were bounded.
2. Recurrent utility was already negative at C954 and remained negative at C1908 (`-0.093616308`, 0/20), despite real state beating shuffled by `0.192940230` and zero by `6.609913499`.
3. Effective recurrent/fused scale continued rising while `W_g` grew and `W_u` non-leading spectrum reorganized.
4. The simultaneous rho/window transition produced the separate Stage-D scale stop, and its causal split is not identified.

This rules out a single-factor story from current evidence. Interpretation A captures the Stage-C utility failure; E captures the unresolved Stage-D trigger; neither alone covers both outcomes.

## 16. Exactly one next experiment

**Experiment 2D1B — C1908 rho × window factorial diagnostic.** Run the four zero-training conditions A–D from section 12 on identical batches, with plain/real/zero/shuffled controls, scale decomposition, and 32-pass stability. Do not update parameters. This experiment directly resolves the only causal ambiguity blocking a principled decision about secondary stabilization or a fusion-equation change.

No other experiment is recommended before 2D1B, and 2D1B was not started.

## 17. No-new-training audit

The postmortem process read and aggregated saved files only. No checkpoint was loaded into model code, no optimizer was constructed, no training dataloader was advanced, and no GPU pod was started. The required counters are all exactly zero, as recorded near the top and in `DETAILED_POSTMORTEM.json`.

## 18. Provenance and availability audit

### Source artifacts

- `results/experiment_2d1r_wu_spectral_control/training_metrics.jsonl` — SHA-256 `4184578700a640dc8faec3bb5706aa5c487c2fdfaa05f8b741bd1bca47e1a79c`, 2623679 bytes
- `results/experiment_2d1r_wu_spectral_control/projection_metrics.jsonl` — SHA-256 `c9c4dcac9a51a718642b9e00fb57bd7df39c8bcbf3d4abe44aa71396ec0f58b6`, 563150 bytes
- `results/experiment_2d1r_wu_spectral_control/milestone_validation.json` — SHA-256 `2b4f9fd3dd3862a90245542162a54fdb85158a2b4be391285b84ea0878fad71d`, 38182 bytes
- `results/experiment_2d1r_wu_spectral_control/recurrent_controls.json` — SHA-256 `e2f7b36d068a773ff3170d81bca3b5a351c3814f76b16f540abee8c94e4641cf`, 10070 bytes
- `results/experiment_2d1r_wu_spectral_control/scale_diagnostics.json` — SHA-256 `485793fc23fabf4e79ac8ff6f7d237fa8e4cdd59145e8e8969d6a1e175ed433a`, 13509 bytes
- `results/experiment_2d1r_wu_spectral_control/self_composition.json` — SHA-256 `2520d5a4ca9be4dbd3d4cc0be2eb1746b34bed7b410e8abd45b7574dedf4774d`, 154045 bytes
- `results/experiment_2d1r_wu_spectral_control/failed_lineage_comparison.json` — SHA-256 `c9f8d2aa1314b5b490954e91b5b674feb7d1cf071c909ff62047a7a632c9ce1e`, 1528 bytes
- `results/experiment_2d1r_wu_spectral_control/terminal_hard_failure.json` — SHA-256 `d0acf752dd97d1f64fbeb42f064d47978dedcd136e609c68f827af4d76e096a5`, 944 bytes
- `results/experiment_2d1r_wu_spectral_control/projection_summary.json` — SHA-256 `23417f98ed903234dca5e7e3521dc5c0a01930d706ae2f254827aa127c7f040f`, 1378 bytes
- `results/experiment_2d1r_wu_spectral_control/FINAL_AUDIT.json` — SHA-256 `2eef41696fc175ba506ac1bcacbd6271a770b7bfae81029fd09be0d7dec2710f`, 2275 bytes
- `results/experiment_2d1_triangle_recurrent/training_metrics.jsonl` — SHA-256 `2f1685d8c213764e9232f5f7a289423b2c012d7d4efe0b18a89fed159ce43f53`, 1923723 bytes
- `results/experiment_2d1_triangle_recurrent/milestone_validation.json` — SHA-256 `3a8bf26e253aef17ce843d7190572ff8fcf226b2948600b2d0ac0ad6f05f9c30`, 69659 bytes
- `results/experiment_2d1a_recurrent_scale_forensics/wu_diagnostics.json` — SHA-256 `353a0ed58ce0e45b3dd6ed5edb03e8028d7249c6329b631c0abff0d5f06e81cd`, 3903 bytes
- `results/experiment_2d1a_recurrent_scale_forensics/wg_diagnostics.json` — SHA-256 `d5f99a374bc6213de3ee583823dfb4732537ce0206ca1734ca555ac3106c7deb`, 3750 bytes
- `results/experiment_2d1a_recurrent_scale_forensics/fusion_decomposition.json` — SHA-256 `75c0b889c1368ab1a3ec1cd4b183d3b264e0f4b0cf2e82465e30a95461c5dbd2`, 433789 bytes
- `results/experiment_2d1a_recurrent_scale_forensics/layerwise_rms.json` — SHA-256 `c11c743b1229019c02c6a32a719781f688550887b25f3873004c25c1279d389c`, 450728 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P1_recurrent_scale.png` — SHA-256 `6254778f8e5d45a975a3997db0c7e0863cbbce8df15c1e3ee15f994aede2ef66`, 167102 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P2_wu_spectral_norm.png` — SHA-256 `8c40f9adaa65dccd14551d8e7e188da6e60c148cec0d7aa0857877c75041c003`, 67803 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P3_training_ce.png` — SHA-256 `470f6cfcba1eb8490c8383c176f1f07e0d971b9020db86eaf53418b65988fd98`, 132535 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P4_projection_scale.png` — SHA-256 `4fa0bc481383cfda51b0362388c49dbe6fcf333ccc50f275e0951a01779a2a68`, 40365 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P5_self_composition.png` — SHA-256 `3bf6541704b5914ce03e06f8b2a2d9ce1cc81fc665ac938d436bac692de3003f`, 71622 bytes
- `results/experiment_2d1r_wu_spectral_control/plots/P6_validation_trajectory.png` — SHA-256 `7414fa0f6d455ae7c61b39ca18a042c9f884acb3e8d75e0ff30b9403a77089bc`, 52658 bytes

### Explicitly unavailable quantities

- `Stage-C plain/real validation at updates 955, 956, and 1500`: **NOT AVAILABLE**
- `Stage-C zero and shuffled validation at updates 954, 1000, 1100, and 1200`: **NOT AVAILABLE**
- `C1908 argmax agreement for any control pairing`: **NOT AVAILABLE**
- `C1908 recurrent-state elementwise variance across rows`: **NOT AVAILABLE**
- `C1908 recurrent-state elementwise variance across token positions`: **NOT AVAILABLE**
- `C1908 real/shuffled/zero representation-level cosine, norm, and variance comparisons`: **NOT AVAILABLE**
- `F-vs-E cosine and X-vs-E cosine at C1908`: **NOT AVAILABLE**
- `2D1R B1/layerwise residual diagnostics after C954`: **NOT AVAILABLE**
- `G RMS in 2D1R training metrics (gate mean/std/saturation were saved instead)`: **NOT AVAILABLE**
- `Original 2D1 canonical plain/recurrent validation at matched updates 1000 and 1100`: **NOT AVAILABLE**
- `Update-1915 training/pass losses, detailed fusion scales, projection values, W_g diagnostics, gradient norm, and LR because the stop preceded the optimizer step`: **NOT AVAILABLE**
- `A clean C1908 rho-by-window intervention separating the Stage-D changes`: **NOT AVAILABLE**

Arithmetic derived from saved scalars is labeled as derived. No missing result was recreated with training or a new result-bearing evaluation.

# EXPERIMENT 2D1R POSTMORTEM COMPLETE
