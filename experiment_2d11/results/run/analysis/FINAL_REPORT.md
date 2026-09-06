# Experiment 2D11 — COMPLETED

One fresh O1 + H model completed **19,072 updates / 9,999,220,736 processed logical targets**. The existing completed GPT-2 control was reused. H used 38,740 CE pass-equivalent updates / 20,310,917,120 CE pass-targets (mean 2.03125 passes). Both preregistered early screens permitted continuation. All requested intermediate comparisons completed without waiting for user responses.

## Final paired comparison

| Endpoint | H | Existing G | H benefit | Adjusted 97.5% interval | Classification |
|---|---:|---:|---:|---|---|
| Fresh 4,096-sequence CE | 3.01494812 | 3.03861379 | 0.02366567 nats (G − H) | [0.022745, 0.024584] | **Benefit** |
| Full 10,042-example HellaSwag | 30.9201% | 30.3127% | 0.6074 pp (H − G) | [-0.059749, 1.274646] pp | **Unresolved** |

H lowers fresh-panel perplexity by 2.3388%. Its CE is lower on 3,377/4,096 sequences, with 0 ties. Descriptive 95% intervals: CE [0.022852, 0.024475] nats; HellaSwag [0.019916, 1.185023] pp. The adjusted CE lower bound exceeds the historical 0.0001-nat small-effect reference; that reference does not establish value for the extra compute.

Paired HellaSwag correctness: both wrong 6,523; G wrong/H right 475; G right/H wrong 414; both right 2,630. Thus H gets 61 more examples correct. Its adjusted interval crosses zero, so these results **do not support a blanket overall-superiority claim**.

Bootstrap: 50,000 paired resamples, isolated seeds 20260918 (CE) and 20260919 (HellaSwag), NumPy 2.5.2, linear percentiles. Bonferroni 97.5% marginal intervals govern the two endpoint classifications. This is one trajectory per recipe, not seed replication or a causal isolation of the gate. H uses additional multipass compute.

## Matched intermediate comparisons

| Trained targets | H CE | Historical G CE | H HellaSwag | Historical G HellaSwag |
|---:|---:|---:|---:|---:|
| 524,288,000 | 4.052053 | 4.135291 | 25.9908% | 25.5029% |
| 1,048,576,000 | 3.610620 | 3.662819 | 26.7676% | 26.0705% |
| 2,621,440,000 | 3.321764 | 3.354195 | 27.9725% | 27.5244% |
| 5,242,880,000 | 3.169740 | 3.193808 | 29.4961% | 28.8190% |
| 7,864,320,000 | 3.084565 | 3.108659 | 30.3226% | 29.6355% |

These use the original G aggregate records; no paired confidence intervals are available for intermediate G weights. The final historical-style H monitoring CE is 3.05167600. The saved G original parallel evaluator reproduced its historical 3.0750441551208496 CE exactly in preflight. The fresh-panel paired scores above use the same incremental evaluation for both models and are the primary endpoint.

## Runtime, compute, and recovery

The exact assigned four-A100-80GB pod is **verified stopped**. Total cumulative billed time, including all startup/repair attempts, was **15.5260 pod-hours / 62.1039 GPU-hours**, below 24 hours. At the quoted $6.36/pod-hour, estimated compute cost is **$98.75**, excluding storage.

Retained scientific training: 13.5254 hours. Scheduled scientific evaluations: 59.30 minutes. Successful disposable preflight: 5.63 minutes. A controller network failure caused a verified pod stop and recovery from u6000, discarding 104 later updates / 266.15 training seconds; the raw tail remains archived. All billed attempts are included. The remaining 50.67 minutes combine earlier failed setup/preflight, saves, transfer tail, shutdown and idle time; those categories cannot be separately reconstructed reliably and are not presented as pure idle time.

Final GPU completion to verified provider stop took 9.15 minutes, used for independent checkpoint/artifact export and shutdown. Transfers overlapped training. Some earlier transfer acknowledgements were refreshed after recovery, so their durations do not reconstruct all historical transfer time. The billed-time chart uses the correct segment on each side of the restart and excludes stopped gaps.

The recovery restored verified model/optimizer/RNG/loader state and fixed identities. The first replayed update's loss and gradient norm matched; later repeated-step losses differed slightly (maximum observed over 23 steps: 7.3433e-5). We do not claim multi-step bitwise reproduction. The preregistered disposable next-step parameter/optimizer resume audit passed on all four ranks with zero maximum parameter error. Details and raw recovery evidence are retained.

Same-hardware disposable update timing: G one-pass 0.863s; H two-pass 2.498s; H three-pass 3.794s. Incremental B4 inference (768-token prime, 256 timed tokens), minimum rank throughput: G 404.11 tokens/s; H 269.45 tokens/s. These are disposable preflight-weight measurements, not final-weight throughput. Historical G ran on one A100, so a four-GPU H wall-time comparison is not architecture speedup.

H has 124,697,386 registered / 124,697,382 active parameters; G has 124,475,904. Persistent BF16 state per sequence: H 33,289,728 bytes, G 37,711,872 bytes. Rank-0 peak training allocation: 46.773 GB. See resource CSV/JSON and figures for evaluation throughput and measured training/GPU-hour coordinates.

## Integrity and artifacts

All 238 exported science-artifact hashes were rechecked locally. Every update's loader cursor and pass cadence match the frozen plan; all 24 monitoring CE panels, seven HellaSwag panels and final paired panels have complete unique ownership. Initial tensor identity, distributed gradient checks, checkpoint optimizer/RNG audits, terminal cursor, budget and provider stop all pass. Actual input token hashes are retained per update. CUDA determinism and controller fixes are documented separately from the frozen scientific identity.

Final H checkpoint (outside Git): `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d11/run/checkpoints/u19072.pt`.
Persistent copy: `/workspace/exp2d11/scientific_20260905_retry2/run/checkpoints/u19072.pt` on network volume yhzyb27fb5 (190 GB).
SHA-256: `bdcbe4ceede0af61b0cea177d677c31b4b26d3f002a3aabdba1f3bde6810011b`; 1,651,153,573 bytes.

The scientific bundle identity remains commit `54845a06bac1361e518c6729ed46f547553d21e3` plus patch SHA `f43e4de57af847bb67f3f23f17f7551e4afa90ca4db0f30e0cc0c665311497ce`. Operational recovery patch SHA: `ea58cb231892b91e2d06b26eb64549b1185b7c6a56ace16ab1c3b8b3264d259c`. The final result commit/tag bind the packaged evidence and local analysis code. Model weights and datasets are excluded from Git. No further training is authorized.

See `AUDIT.json`, `PAIRED_ENDPOINTS.json`, `RESOURCE_ACCOUNTING.json`, `HARDWARE_SUMMARY.json`, CSV/JSON data, PNG/PDF figures, and `TECHNICAL_DETAILS.md` here; complete run evidence is packaged alongside this analysis.

A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.
