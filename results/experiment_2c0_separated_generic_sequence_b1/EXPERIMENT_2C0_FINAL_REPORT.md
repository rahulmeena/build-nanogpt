# Experiment 2C0 — Final Report

Experiment 2C0 stopped at its preregistered zero-shot gate. The fixed generic branch reproduced the 2B5 generic-template result, and the separated sequence branch reduced raw CE, but coherent shuffled centered states outperformed aligned states on every canonical batch. No optimizer was constructed and no result training occurred.

## Git

2B5 frozen tag: `experiment-2b5-decomposition-final`  
2B5 parent commit: `6c8112f267f64751080eaa7799ad2f76a93fa591`  
2C0 branch: `experiment-2c0-separated-generic-sequence-b1`  
Implementation commit: `91ac28fd94e8b9e3df040e3c3d7d24832b39f037`  
Results commit: `72ba90bbbd24c26268e9bbacc7dc7b9b2cadd08a`  
Final report commit: `the immutable commit containing this file`

## Generic branch

Generic calibration source: final Experiment 2B3 reader/gate and exact Experiment 2B5 final-checkpoint means  
Generic μ provenance: `results/experiment_2b5_mean_residual_decomposition/calibration_means/C3_2B3_FINAL.pt`  
G SHA: `cc169dc9b2cddd88657f6e88edffbdd01ccf50edd3279a94619d1b02216a703d`  
G RMS: `0.0102207763`  
Generic-only expected loss: `4.7776873112`  
Generic-only measured loss: `4.7776865301`  
Regression: `PASS`

## Sequence centering

Calibration manifest: `results/experiment_2b4_memory_content_mask_depth/part_a_calibration_manifest.json`  
ν16 SHA: `651ef4bb85866814dbb21b6bf006bc556778bb1d591089632d724bb210c2b3e7`  
ν17 SHA: `ec4ab9d3a99720be156ce98a8554170e0b9b827b607229d3c3d3c8bd6d00d439`  
ν20 SHA: `3d65cbe87fc56f89dbce01fcdd88b2601f8b2d27c0d831ae5bf9e3f4d399b67a`  
ν24 SHA: `63b3cc410df843b93664723e670b5c7b9892e3e8ef376529cf49aa4535033fae`  
Mean centered calibration residual, maximum absolute component: `1.492e-08`

## Initialization

2B1 source checkpoint: `/workspace/migration/experiment_2b1_5m/result/checkpoints/checkpoint_updates_000010.pt`  
SHA: `5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a`  
Copied query norm: `1.9570869207`  
Copied RMSNorm displacement: `1.9478669167`  
Old effective gate: `0.1580765992`  
New effective gate: `0.0790382996`

## Zero-shot controls

Generic only: `4.7776865301`  
Generic + real sequence: `4.7503485353`  
Generic + shuffled sequence: `4.7373461495`  
Sequence only: `5.8830049967`  
Gate zero: `4.7776865301`  
Sequence gain above generic: `0.0273379948`  
Specific gap: `-0.0130023858`  
Real wins: `0/20`  
Shuffled wins: `20/20`

## Distributed preflight

Not run. The zero-shot gate failed before optimizer construction, so the protocol forbade smoke, 1→4 GPU migration equivalence, and result training.

## Training

Updates: `0`  
Targets: `0`  
Runtime: `N/A`  
Targets/sec: `N/A`  
Peak training VRAM: `N/A`

## Final controls

No trained final checkpoint exists. The zero-shot controls above are the terminal Experiment 2C0 result.

## Primary metrics

Training real gain: `N/A — training forbidden by gate`  
Initial specific gap: `-0.0130023858`  
Final specific gap: `N/A`  
Specific gap gain: `N/A`  
Real wins: `0/20`  
Shuffled wins: `20/20`  
Batchmean gap: `N/A — final trained control not reached`

## Sequence reader

| metric | initial | final |
|---|---:|---:|
| effective gate | 0.07903830 | N/A |
| query norm | 1.95708692 | N/A |
| RMS displacement | 1.94786692 | N/A |
| mean-feedback ratio | 0.15801769 | N/A |

## Integrity

- trainable_parameters_exactly_1537: PASS
- base_gradients_none: PASS
- old_reader_gradients_none: PASS
- writer_gradients_none: PASS
- writers_never_active: PASS
- generic_G_frozen: PASS
- source_means_frozen: PASS
- historical_KV_temporal_gradients_none: PASS
- future_causality_pass: PASS
- row_isolation_pass: PASS
- zero_input_feedback_exactly_zero: PASS
- all_losses_finite: PASS
- generic_only_regression_pass: PASS
- zero_shot_gate_failed_as_observed: PASS
- result_optimizer_never_constructed: PASS
- optimizer_updates_zero: PASS
- backward_calls_zero: PASS
- parameter_updates_zero: PASS
- additional_training_targets_zero: PASS
- hellaswag_not_run: PASS

## Classification

SEPARATED SEQUENCE BRANCH IMPROVES GENERIC COMPENSATION ONLY

## Decisions A–F

A. NO. The initialized separated branch improved raw CE, but shuffled centered states were better on all 20 batches.  
B. NO. The zero-shot gate forbids reader training, so continuation beyond 5M is not authorized.  
C. YES, if Block 1 is retried. Architectural separation alone did not create positive aligned-sequence specificity.  
D. YES, as a new controlled experiment. A middle/higher destination should be tested before adding writers.  
E. YES. Writers should remain absent until a destination demonstrates a strong direct sequence-specific signal.  
F. YES. Keep temporal credit zero/one-step-free during the direct-reader stage.

No additional reader training, writers, auxiliary losses, destination changes, mask-depth changes, BPTT, or HellaSwag were launched.

# EXPERIMENT 2C0 COMPLETE
