# Experiment 2A3 — Teacher-Assisted Memory Reader, 100M → 250M

## Outcome

**MEMORY SIGNAL SATURATING**

Experiment 2A3 completed successfully at exactly **477 Experiment-2 updates** and **250,085,376 Experiment-2 student tokens**. The aligned teacher-history advantage remained positive on all 20 canonical validation batches and reached **0.1797709703 loss**, but its incremental growth fell sharply:

| Interval | Increase in shuffled-minus-real loss gap |
|---|---:|
| 100M → 150M | +0.0342535257 |
| 150M → 200M | +0.0037150621 |
| 200M → 250M | +0.0002277851 |

The signal did not reverse, but the final increment was only 0.66% of the 100M→150M increment. This satisfies the frozen classification rule for saturation.

No optimizer step was executed beyond the target, and no follow-on experiment was launched.

## Resume

| Field | Value |
|---|---|
| Starting checkpoint | `/workspace/build-nanogpt-exp2a0/runs/experiment_2a2_100m/checkpoints/checkpoint_updates_000191.pt` |
| Starting SHA-256 | `6c206a89422470061d7997764fbd9a5708be3d9043f8fab930dd4b800bd5cb95` |
| Starting updates | 191 |
| Starting tokens | 100,139,008 |
| Last consumed schedule step | 1144 |
| Next schedule step | 1145 |
| Restart learning rate | 0.0005992693258964253 |
| Starting next-batch hash | `9f39510b105f068966ef6c052edc015d695827c422da37495fa7c244b965af0b` |
| Resume verification | **PASS** |

The source checkpoint, checksum, verification sidecar, model, feedback-only AdamW state, four replay loaders, Python/NumPy/Torch CPU/Torch CUDA RNG state, update/token counters, immutable Experiment-1B lineage, and next-global-batch hash were verified before the first optimizer step. The existing feedback query, RMSNorm, gate, optimizer moments, loaders, and RNG were restored; none were reinitialized.

## Frozen protocol and integrity

- Architecture: unchanged Experiment 2A2 teacher-assisted Block-1 memory reader.
- Hardware: 1× NVIDIA A100-SXM4-80GB.
- Global update: 524,288 student tokens, reconstructed from the original four-rank B=64, T=1024, gradient-accumulation=2 stream.
- Trainable parameters: exactly 1,537 (query 768, RMSNorm 768, scalar gate 1).
- Frozen student-base hash: `1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd` throughout.
- Teacher: immutable Experiment-1B checkpoint, eval mode, `no_grad`, detached one-token-shifted memory.
- Training rows: exactly update indices 191–476; schedule steps 1145–1430; all replay hashes, token counts, and learning rates matched the oracle.
- Frozen student and teacher gradients: none.
- All losses, gradients, reader parameters, and optimizer moments: finite.
- Causality checks at updates 191, 286, 381, and 477: **PASS**. Position-zero memory was zero, v16/v17/v20/v24 prefix states were exact, and perturbed-future end-to-end prefix logits were bit-identical with maximum absolute difference 0.

## Training

| Field | Value |
|---|---:|
| Additional updates | 286 |
| Additional student tokens | 149,946,368 |
| Final Experiment-2 updates | 477 |
| Final Experiment-2 tokens | 250,085,376 |
| Overall mean training loss | 5.5230369816 |
| Final training loss | 5.4731246829 |
| Minimum training loss | 5.4195085168 |
| Maximum training loss | 5.6338368058 |

Interval detail:

| Updates | Rows | Mean loss | Last loss | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 192–286 | 95 | 5.5281202705 | 5.4872639775 | 5.4195085168 | 5.6252499819 |
| 287–381 | 95 | 5.5189995182 | 5.4962112308 | 5.4246547818 | 5.6338368058 |
| 382–477 | 96 | 5.5220020500 | 5.4731246829 | 5.4272706509 | 5.6191518903 |

These are feedback-reader training losses with a frozen base, not a from-scratch GPT-2 pretraining curve.

## Validation trajectory

The fixed controls are full-context = 4.0786544085 and masked/no-feedback = 5.9736744881, giving damage = 1.8950200796. Historical 5M–100M values are the previously audited frozen results and were not rerun.

| Tokens | Real val | Shuffled val | Shuffled−real gap | Total recovery % | Specific recovery % | Specific share % |
|---:|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 5.9533051014 | 5.9617962360 | 0.0084911346 | 1.0749 | 0.4481 | 41.6858 |
| 10,485,760 | 5.9229358435 | 5.9391546011 | 0.0162187576 | 2.6775 | 0.8559 | 31.9653 |
| 15,204,352 | 5.8944202185 | 5.9179884195 | 0.0235682011 | 4.1822 | 1.2437 | 29.7375 |
| 25,165,824 | 5.8353391409 | 5.8765912533 | 0.0412521124 | 7.2999 | 2.1769 | 29.8204 |
| 50,331,648 | 5.7143192530 | 5.7953289270 | 0.0810096741 | 13.6861 | 4.2749 | 31.2350 |
| 100,139,008 | 5.5957053900 | 5.7372799873 | 0.1415745974 | 19.9454 | 7.4709 | 37.4567 |
| 149,946,368 | 5.5728114128 | 5.7486395359 | 0.1758281231 | 21.1535 | 9.2784 | 43.8624 |
| 199,753,728 | 5.5706561565 | 5.7501993418 | 0.1795431852 | 21.2672 | 9.4745 | 44.5496 |
| 250,085,376 | 5.5700338840 | 5.7498048544 | 0.1797709703 | 21.3001 | 9.4865 | 44.5374 |

All new evaluations used the canonical 20×B64×T1024 BF16 prefix with hash `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

## 250M controls

| Control | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked Layer-1 / no feedback | 5.9736744881 |
| Real aligned teacher feedback | 5.5700338840 |
| Shuffled teacher feedback | 5.7498048544 |
| Gate forced to zero | 5.9736744881 |

Derived metrics:

| Metric | Value |
|---|---:|
| Damage | 1.8950200796 |
| Total recovery | 0.4036406040 |
| Total recovery fraction | 21.3001% |
| Sequence-specific recovery | 0.1797709703 |
| Sequence-specific recovery fraction | 9.4865% of original damage |
| Sequence-specific share of total recovery | 44.5374% |

Gate-zero equaled masked/no-feedback on every validation batch. The full-context and masked controls reproduced the pinned values exactly.

## Paired validation batches at 250M

| Statistic | Result |
|---|---:|
| Real wins | 20 |
| Shuffled wins | 0 |
| Ties | 0 |
| Mean(real − shuffled) | -0.1797709703 |
| Median(real − shuffled) | -0.1825425625 |
| Sample standard deviation | 0.0101229865 |
| Minimum(real − shuffled) | -0.1958169937 |
| Maximum(real − shuffled) | -0.1526241302 |

The descriptive paired 95% t interval for the opposite orientation, shuffled−real, is [0.1750332668, 0.1845086739]. It is reported descriptively only: the fixed contiguous validation batches are not assumed IID, and no claim of formal significance is made.

## Router evolution

| Tokens | Gate | tanh(gate) | Query norm | RMS displacement | Entropy | Normalized entropy | v16 | v17 | v20 | v24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,242,880 | 0.005998 | 0.005998 | 0.124895 | 0.090100 | 1.289204 | 0.929964 | 0.271206 | 0.251649 | 0.219422 | 0.257723 |
| 10,485,760 | 0.011995 | 0.011994 | 0.269662 | 0.226218 | 1.037263 | 0.748227 | 0.354613 | 0.159561 | 0.198184 | 0.287643 |
| 15,204,352 | 0.017393 | 0.017391 | 0.391837 | 0.351041 | 0.850197 | 0.613287 | 0.411335 | 0.118092 | 0.179551 | 0.291022 |
| 25,165,824 | 0.028787 | 0.028779 | 0.605271 | 0.581372 | 0.737608 | 0.532072 | 0.406141 | 0.165030 | 0.169009 | 0.259819 |
| 50,331,648 | 0.057569 | 0.057505 | 1.030883 | 1.044126 | 0.613797 | 0.442761 | 0.420169 | 0.145839 | 0.204620 | 0.229372 |
| 100,139,008 | 0.114390 | 0.113894 | 1.576196 | 1.588363 | 0.584330 | 0.421505 | 0.422467 | 0.134264 | 0.216281 | 0.226988 |
| 149,946,368 | 0.155276 | 0.154040 | 1.754378 | 1.756138 | 0.606062 | 0.437182 | 0.434299 | 0.123921 | 0.218139 | 0.223641 |
| 199,753,728 | 0.159776 | 0.158430 | 1.870173 | 1.865679 | 0.612769 | 0.442019 | 0.429305 | 0.127991 | 0.219102 | 0.223602 |
| 250,085,376 | 0.159529 | 0.158190 | 1.956499 | 1.946933 | 0.607931 | 0.438530 | 0.427155 | 0.128827 | 0.220112 | 0.223906 |

The gate plateaued near 0.16 after 150M while the query and RMSNorm norms continued to grow. Routing stayed concentrated at an effective source count of 1.8366, with v16 dominant. Mean routing weight is not itself causal importance.

## Source ablations at 250M

Real-feedback baseline loss: 5.5700338840.

| Source removed | Ablated loss | Delta vs real | Positive batches |
|---|---:|---:|---:|
| v16 | 5.8421126366 | +0.2720787525 | 20/20 |
| v17 | 5.6350365639 | +0.0650026798 | 20/20 |
| v20 | 5.6795891762 | +0.1095552921 | 20/20 |
| v24 | 5.6687029362 | +0.0986690521 | 20/20 |

The ranking remained **v16 > v20 > v24 > v17**, the same as at 100M. Every delta grew from 100M to 250M: v16 0.202792→0.272079, v17 0.050123→0.065003, v20 0.079622→0.109555, and v24 0.073164→0.098669. These are renormalized leave-one-source-out effects, not additive or independent source contributions.

## HellaSwag at 250M

The complete upstream validation set of 10,042 examples was evaluated after candidate/example isolation tests passed.

| System | Correct | Accuracy |
|---|---:|---:|
| Standard GPT-2 @500M, historical | 2,568 / 10,042 | 25.5726% |
| Full AttnRes @500M, historical | 2,532 / 10,042 | 25.2141% |
| Full-context current | 2,532 / 10,042 | 25.2141% |
| Masked Layer-1 / no feedback | 2,407 / 10,042 | 23.9693% |
| Real teacher feedback | 2,501 / 10,042 | 24.9054% |
| Gate forced to zero | 2,407 / 10,042 | 23.9693% |

Real feedback recovered **94 of the 125 examples** lost by masking (75.2%). It remained 31 examples, or 0.3087 percentage points, below full context, while improving 94 examples, or 0.9361 points, over masked/no-feedback.

Shuffled HellaSwag was deliberately skipped: candidate-row shuffling would exchange memory across the four answer alternatives and contaminate scoring. Candidate-zero logits and teacher memory were bit-exact when other candidates changed; example reset checks, position-zero memory, finiteness, and zero-equals-masked all passed.

These are not equal-token pretraining comparisons. The reader is 1,537 learned parameters on a frozen 500M-token Full-AttnRes base.

## Checkpoints and restart verification

| Milestone | Updates | Tokens | Checkpoint SHA-256 | Next-global-batch hash | Verification |
|---|---:|---:|---|---|---|
| 150M | 286 | 149,946,368 | `48afa92fcc1174f80278a3024edc9b05ca689c47eb1f24eea31bf3eb018aa364` | `94f21a6b52b3e14bddfd0221076172d2c04a9067dac6ca6e2e9ecfdaaed99ded` | PASS |
| 200M | 381 | 199,753,728 | `6b48b7d67f4831ca38616184ebe091338f1866eb51ffe5f6419a3e1ac64ef599` | `73dc271a2f06e5f841a8207a3d0243d09ad16b28106b39351381f76fc08d8af2` | PASS |
| 250M | 477 | 250,085,376 | `0702dc09c74b01eee8be504a7f5f89ca61fcc504cda8f34f30865d4ff9653d76` | `95081c5f68b7d05d6e39b68043f2714657c21ca05cc317549063ba9a4f9f6986` | PASS |

Remote checkpoint paths are under `/workspace/build-nanogpt-exp2a0/runs/experiment_2a3_250m/checkpoints/`. Each checkpoint was written atomically, hashed, reopened, and strict-reload verified for model, optimizer, four loaders, and all RNG state. Fresh-object restart verification passed at updates 286 and 381.

## Resources

| Metric | Value |
|---|---:|
| End-to-end completion wall time | 18,179.24 s (5 h 02 m 59 s) |
| Training-update wall time | 9,068.24 s (2 h 31 m 08 s) |
| Training forward time | 4,531.93 s |
| Training backward time | 4,524.32 s |
| Canonical evaluation time | 330.51 s |
| Full HellaSwag time | 5,597.37 s (1 h 33 m 17 s) |
| Peak allocated VRAM | 50,629.48 MiB |
| Peak reserved VRAM | 57,894 MiB |

## Interpretation and decisions

### MEMORY SIGNAL SATURATING

The correct-sequence history signal is genuine and retained: real feedback beat shuffled feedback on every new validation batch, total recovery reached 21.30% of the original masking damage, and 44.54% of the reader's recovered validation loss depended on correct sequence alignment. The source ablations also show all four high-level inputs remain useful under the renormalized intervention.

However, the primary sequence-specific signal flattened from 0.175828 at 150M to 0.179543 at 200M and 0.179771 at 250M. The gate and routing distribution also stabilized. The appropriate classification is therefore saturation rather than continued strengthening or reversal.

**Continue this exact teacher-assisted reader from 250M to 500M? YES, as a separately approved frozen-protocol run.** The signal is positive, nondecreasing, 20/20 batch-consistent, the invariants passed, and HellaSwag real feedback was substantially better than masked/no-feedback. The expected marginal gain is now small, so the value of a 500M continuation is chiefly to establish the plateau rather than to expect another large recovery jump.

**Begin self-generated high(t−1) → low(t) recurrence now? NO.** The preregistered gate required sequence-specific recovery of at least 10% of original damage; the result is 9.4865%. More importantly, this remains a detached full-context-teacher diagnostic. It does not establish that a student's own previous higher-layer state will be stable, sufficiently informative, or distribution-compatible under recurrence.

## Artifacts

- `result_summary.json`: authoritative machine-readable summary and raw HellaSwag predictions.
- `metrics.jsonl`: all 286 optimizer-update rows.
- `evaluations/`: raw paired losses, complete final controls, source ablations, and HellaSwag results.
- `causality_updates_*.json`: source-level and end-to-end no-future-leakage audits.
- `restart_audit_updates_*.json`: fresh-object exact-restart audits.
- `checkpoints/*.sha256` and `*.verification.json`: checkpoint manifest and strict verification sidecars.
- `plots/`: ten plots in PNG and PDF plus `plot_data.csv`, `plot_data.json`, and `plot_manifest.json`.

# EXPERIMENT 2A3 250M COMPLETE
