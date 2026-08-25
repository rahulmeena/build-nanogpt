# Experiment 2D1A — Recurrent Scale-Instability Forensics

EXPERIMENT 2D1A ROOT CAUSE:
W_U AMPLIFICATION DOMINATES

RECOMMENDED 2D1R STABILIZATION:
W_U NORM CONTROL

RECOMMENDED RESTART CHECKPOINT:
update 954

## Exact failure trajectory

The frozen Stage-A recurrent-input RMS reference was `0.03550996296108` and the exact 10x hard threshold was `0.35509962961078`. Scale responded immediately to the Stage-C transition (RMS `0.06386530` at C954, `0.09687792` at update 955, and `0.22227934` at update 956), but the first logged 10x crossing occurred only at update 1091. Updates 1158 and 1159 were the first two terminal consecutive crossings, and attempted update 1160 produced the third value `0.39886176586151123` before logging, triggering the preregistered stop.

## Checkpoint scale decomposition and learned matrices

C954 COMMON-C repeated composition: **OSCILLATORY**. C1000: **CONTRACTIVE**. C1100: **OSCILLATORY**. C1100 under COMMON-B: **OSCILLATORY**.

W_u C1100/C954 spectral ratio: `1.33735161`; Frobenius ratio: `1.16603138`. W_g spectral ratio: `1.02229945`. Full singular values, activation ratios, gate saturation, and position bins are in the JSON artifacts.

The largest C1100 COMMON-C pass-3 sublayer gain was `B1_post_mlp` at `2.15900826x`. Finite-difference output amplification changed from `82.84207746` at C954 to `24.27548984` at C1100.

## Stabilization probes

The selected probe was `F3`. It bounded maximum recurrent-input RMS at `0.28084406` with late-pass CE `3.20379856`. The choice prioritizes bounded recurrence and late state-change behavior; CE is a secondary non-catastrophicity check, not the sole selection rule.

F1 and F2 also bounded scale, but their late-pass CEs were `10.23534799` and `9.30581546`, versus native `3.17187527`. F3 preserved a reasonable `3.20379856` CE while reducing maximum recurrent-input RMS to `0.28084406`; therefore W_u norm control is the only predefined probe satisfying both the scale and non-catastrophic-CE criteria.

## Direct Q1–Q17 answers

- **Q1:** The full recurrent_input tensor after prefix selection and before position embedding crossed 10x: terminal RMS 0.398861765862 versus threshold 0.355099629611.
- **Q2:** Scale rose immediately with the Stage-B→C configuration change (RMS 0.063865 at C954, 0.096878 at update 955, and 0.222279 at update 956), then learned W_u growth pushed it to the first 10x crossing only at update 1091.
- **Q3:** C954 COMMON-C remains scale-bounded below 10x but is dynamically OSCILLATORY; pass-3 recurrent-input RMS averaged 0.250367194414 (native 0.147837396711), with severe COMMON-C CE degradation documented in self_composition.json.
- **Q4:** C1100 under COMMON-B remains below 10x (maximum recurrent-input RMS 0.250372827053) but is descriptively OSCILLATORY across 32 passes.
- **Q5:** Yes. ZN is affine-free RMS-normalized and remained close to unit RMS apart from the exact zero state at position zero; see fusion_decomposition.json.
- **Q6:** Under COMMON-C pass 3, U/ZN activation gain rose from 0.343656 at C954 to 0.495588 at C1100; spectral norms were 1.026231766 and 1.372432709.
- **Q7:** C1100/C954 W_u spectral ratio is 1.337352 and Frobenius ratio is 1.166031.
- **Q8:** No. Gate saturation was absent: |G_PRE|>5 fractions C954=0, C1100=0, with G near 0.95 mean.
- **Q9:** The 10x threshold first appears before B1: C1100 COMMON-C X/E is 10.067054x. The residual stack then grows B1 input to B12 post-MLP by 7.659010x, but it is not the first threshold-crossing location.
- **Q10:** B1 post-MLP is the first and largest single abnormal residual gain (2.159008x from B1 post-attention).
- **Q11:** C954: OSCILLATORY, C1000: CONTRACTIVE, C1100: OSCILLATORY.
- **Q12:** No; it decreased by 70.70%: C954=82.842077, C1100=24.275490.
- **Q13:** F3 best met the preregistered bounding/stability criteria; full candidates are in fix_probe_results.json.
- **Q14:** Yes; the selected zero-training probe bounded recurrence with late-pass CE 3.203799.
- **Q15:** W_U AMPLIFICATION DOMINATES.
- **Q16:** W_U NORM CONTROL.
- **Q17:** update 954, the clean Stage-B boundary before prolonged Stage-C adaptation.

## Integrity audit

`33/33` checks passed. Overall: **PASS**. No optimizer, scheduler, GradScaler, backward, optimizer step, parameter update, or training target occurred. All checkpoint file hashes and in-memory native parameter hashes were unchanged.

## Scientific boundary

This result diagnoses recurrent dynamical scale only. It is not evidence that triangle KV geometry fails, nor that one B12 recurrent source is insufficient. No 2D1R training was launched.
