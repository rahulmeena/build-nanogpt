# Experiment 2D6F Final Report

## Result

**NATIVE B6 W1024 PRACTICALLY EQUIVALENT; REMOVE B7→B6 FOR SIMPLICITY AND SPEED**

Panel: **fresh disjoint confirmation panel** — 2,048 paired sequences and 2,097,152 targets per condition.

- Fresh Fixed CE: `3.057998092452`
- Fresh New CE: `3.058005905410`
- Fresh Fixed − New: `-0.000007812958`; paired 95% CI `[-0.000078528277, +0.000062775552]`; SE `0.000036077824`
- Fresh New penalty: `+0.000007812958`; paired 95% CI `[-0.000062775552, +0.000078528277]`
- Positive-sequence count for Fixed − New: `1007` / 2,048
- Reused-panel Fixed − New: `+0.000060215693`
- Stratified pooled Fixed − New: `+0.000026201368`; 95% CI `[-0.000023300395, +0.000075540409]`
- Stratified pooled New penalty: `-0.000026201368`; 95% CI `[-0.000075540409, +0.000023300395]`
- Panel heterogeneity H = D_fresh − D_reused: `-0.000068028651`; 95% CI `[-0.000165907925, +0.000029247806]`
- Practical margin: `delta_CE = 0.0001`
- Final recommendation: **Delete B7→B6 for architectural simplicity and the previously measured speed advantage.**
- Audit: `PASS`
- Git branch: `experiment-2d6-fresh-panel-zero-training-confirmation`
- Git tag: `experiment-2d6-fresh-panel-zero-training-confirmation-final`
- Pod: `EXITED / stopped`; persistent volume retained

## Interpretation

1. The fresh point estimate favors **Fixed** by `+0.000007812958` CE.
2. Fresh data establish **practical equivalence and native-W1024 noninferiority**, but not statistical superiority for either model and not material inferiority for New.
3. The stratified pooled result resolves the original uncertainty as **practical equivalence** under `delta_CE = 0.0001`.
4. Reused and fresh point estimates are directionally opposite, but the heterogeneity analysis finds no material directional conflict.
5. **Delete B7→B6** for simplicity and the previously measured `+4.913%` A100 throughput advantage; that benchmark is supporting evidence, not a cross-hardware guarantee.
6. No more training or evaluation is warranted under this protocol.

ZERO TRAINING: 0 OPTIMIZER STEPS / 0 BACKWARD CALLS / 0 TRAINING TARGETS

STOPPED AFTER EXACTLY TWO FRESH-PANEL CONDITIONS
