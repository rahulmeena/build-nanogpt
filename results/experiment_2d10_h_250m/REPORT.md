# EXPERIMENT 2D10 — H-ONLY 250M CONTINUATION COMPLETE

**H CLEARS PRACTICAL MARGIN. Preferred tested 250M architecture: H.**
H should replace Dynamic as the preferred baseline under the prespecified adoption rule.

H only: 286 additional updates / 149,946,368 additional targets; 477 updates / 250,085,376 total adaptation targets. One reused 250M Dynamic control; exactly two final evaluations; no further experiment launched.

| Condition | CE | Perplexity |
|---|---:|---:|
| D250M | 3.058171304655 | 21.288591196818 |
| H250M | 3.057451815691 | 21.273279799248 |

D−H = **+0.000719488964**, ordinary paired 95% CI **[+0.000648156153, +0.000791817402]**. Positive favors H. exp(D−H) = 1.000719747858.
Sequence wins: H 2588; D 1508; ties 0.

| Contrast | Positive L>0 | Beyond margin L>δ | Negative U<0 | Material harm U<−δ | Practical equivalence | H noninferiority |
|---|---|---|---|---|---|---|
| D−H | True | True | False | False | False | True |

One prespecified primary comparison; 50,000 paired sequence bootstrap resamples, isolated default_rng(20260913), NumPy 2.5.2, linear 2.5/97.5 percentiles. Margin δ=0.0001. Bounds must strictly cross thresholds; touching does not pass. Adoption additionally requires every integrity and prescribed-state check to pass.

The 100M D−H mean was +0.000384687323, raw 95% CI [0.0003095565770348921, 0.0004601088883811287], adjusted 98.333333% CI [0.00029367442263335025, 0.0004768059918243274]. The descriptive mean change is +0.000334801641. These are different panels: this is not a paired test of growth, and losses were not pooled.

The old 100M report’s flag-table ordering bug is corrected in the new renderer by explicit keys, with a JSON-roundtrip test. The sealed old report, numerical artifacts and immutable tag remain unchanged. Its D−H/T−H beyond-margin flags are True and material-harm flags False; see 100M_REPORT_ERRATUM.md.

## H gates and measured costs

| Destination | b2 | Logit difference mean | λL mean | λR mean / std / range | BF16 λR mean | Entropy mean |
|---|---|---:|---:|---|---:|---:|
| B1 | [-0.020208023488521576, -1.195927381515503] | -1.09447126 | 0.74876010 | 0.25123990 / 0.01846941 / [0.13196518, 0.32189626] | 0.25123797 | 0.56278039 |
| B3 | [0.0171489454805851, -4.4651947021484375] | -4.62287702 | 0.99026208 | 0.00973792 / 0.00042257 / [0.00831248, 0.01150850] | 0.00973792 | 0.05478451 |
| B5 | [0.0075644501484930515, -3.122513771057129] | -3.16885725 | 0.95927437 | 0.04072563 / 0.00574987 / [0.02568752, 0.06961762] | 0.04072562 | 0.16983058 |

| Destination | H100M λR mean | H250M λR mean |
|---|---:|---:|
| B1 | 0.23200602 | 0.25123990 |
| B3 | 0.01023534 | 0.00973792 |
| B5 | 0.03912851 | 0.04072563 |

These gate-distribution comparisons use different panels and are descriptive.

| Router tensor | H100M norm | H250M norm | Change norm |
|---|---:|---:|---:|
| routers.0.W1 | 5.45128338 | 5.47244861 | 0.36474149 |
| routers.0.W2 | 0.02265168 | 0.03711884 | 0.01549593 |
| routers.0.b1 | 0.01003325 | 0.01617711 | 0.00667749 |
| routers.0.b2 | 1.20594977 | 1.19609810 | 0.01411272 |
| routers.2.W1 | 5.40429569 | 5.40626941 | 0.18303164 |
| routers.2.W2 | 0.01335508 | 0.02113866 | 0.00860707 |
| routers.2.b1 | 0.00565456 | 0.00696594 | 0.00236618 |
| routers.2.b2 | 4.46608118 | 4.46522763 | 0.00120380 |
| routers.4.W1 | 5.44396446 | 5.46929708 | 0.41475013 |
| routers.4.W2 | 0.01543978 | 0.04167144 | 0.02698764 |
| routers.4.b1 | 0.00714699 | 0.01388514 | 0.00922160 |
| routers.4.b2 | 3.13153608 | 3.12252293 | 0.01269839 |

Gate table covers eligible-memory positions. GATE_STATISTICS.json contains separate unavailable positions, FP32 and actual BF16 coefficients, full-panel moments/extrema and quantiles from at most 131,072 eligible positions per destination using isolated seed20260914. Unavailable memory is exactly local-only. Router norms and b2 are in H_REAL.json; H_PARAMETER_CHANGES.json records H100M-to-H250M tensor changes. DESCRIPTIVE_100M_COMPARISON.json describes gate distributions across different panels.

H registers 124,697,386 parameters; D registers 124,478,212. H adds 219,174 parameters / 876,696 FP32 bytes. Both retain exactly 33,289,728 BF16 persistent bytes per B=1 sequence; state delta is zero. No geometry or attention/router kernel changed.

H continuation training: 45.12 minutes. D evaluation: 18.78 minutes. H evaluation including diagnostics: 23.02 minutes. Historical D continuation training: 42.44 minutes.
One-GPU pod interval from resume to verified stop: 101.17 minutes / 1.686 GPU-hours. These workload timings do not isolate production inference overhead.

## Integrity and durable outputs

All 286 logical batches, streams, loader cursors, target counts and pass counts match the historical D continuation. There are 277 two-pass and 9 three-pass updates. H reaches global2767 and cumulative1,450,704,896 targets, with terminal cursor and next batch/stream identical to D250M. Active Adam counters advance286; router counters reach477. All four retired/dormant gate tensors, moments and counters remain unchanged. Complete trained H state was restored without fresh router initialization; strict checkpoint reopen passed.

The fresh 4096×1024-target panel is d64cce8b5be6048e24b28977f9098a191d7e6f4d987c3707edad450d01c7deea. Both evaluations used true incremental BF16 execution, FP32 token CE and FP64 accumulation, with cache resets and identical ordered sequences. D tensors and file are unchanged. Panel and recoverable historical target spans are disjoint.

**D250M** SHA256 `9714b2e3f53a8c15dfecfed3e9b56c358176c1f9f609bcce7e28c35b8a358a9b`

- Local: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m/D/scientific_cumulative_001450704896.pt`
- Persistent: `/workspace/exp2d9_dynamic_gating_250m/run/checkpoints/D/scientific_cumulative_001450704896.pt`

**H100M** SHA256 `d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c`

- Local: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d10_retrieval_aware_gating_100m/H/scientific_cumulative_001300758528.pt`
- Persistent: `/workspace/exp2d10_retrieval_gating_100m/run/checkpoints/H/scientific_cumulative_001300758528.pt`

**H250M** SHA256 `a93f9da817d67f4cd0cc56179009041b54ecad356835e26da895ee520fe7545c`

- Local: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d10_h_250m/H/scientific_cumulative_001450704896.pt`
- Persistent: `/workspace/exp2d10_h_250m/run/checkpoints/H/scientific_cumulative_001450704896.pt`

Final checkpoint and all required GPU artifacts have independent local/persistent hash verification. FINAL_AUDIT.json, CHECKPOINT_MANIFESTS.json, ARTIFACT_BACKUP_VERIFICATION.json and STOP_VERIFICATION.json record the evidence.
Pod nagj1hv18p3z2c is verified EXITED/stopped at 2026-09-05T12:43:02.227746+00:00; volume yhzyb27fb5 is retained. Statistics began after verified shutdown.

Scientific implementation commit: `f6ddcf1a54625114c44a42f38ad691fd3dd2b74b`. Result branch: `codex/experiment-2d10-h-250m`; final immutable tag: `experiment-2d10-h-250m-final`. Git push verification is retained alongside the local operations archive.

## Interpretation and stopping point

This compares complete trained architectures. It does not isolate retrieved-output inputs, token variation, mean scaling or initialization; no same-checkpoint router-ablation claim is available. Evaluation-sequence uncertainty is not training-seed replication and does not establish superiority to an independently trained Karpathy GPT-2 baseline.

If separately authorized, compare the preferred architecture against a from-scratch GPT-2 baseline under a new matched protocol.
No further training, window change, extra evaluation panel or from-scratch comparison was launched.
