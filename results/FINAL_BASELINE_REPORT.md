# GPT-2 124M / FineWeb-Edu 10B Final Baseline Report

## Assessment

**Successful reproduction.** The run completed all 19,073 planned optimizer updates, processed 9,999,745,024 tokens, completed its final validation and HellaSwag evaluations, and wrote a valid final checkpoint. The loss and HellaSwag curves agree with the shape and endpoint range of Karpathy's published GPT-2 124M / approximately 10B-token reference plot. No training-semantic changes were made relative to the recorded upstream implementation.

## Run Identity

| Field | Value |
| --- | --- |
| Run directory | `/workspace/build-nanogpt/runs/gpt2_124m_fineweb10b_20260810T141222Z` |
| Current Git commit | `0f7766f9a0c24e9c2854e698862f0a22c19174c7` |
| Upstream Git commit | `6104ab1b53920f6e2159749676073ff7d815c1fa` |
| Branch | `gpt2-124m-10b-reproduction` |
| Dataset | `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train` |
| Tokenizer | GPT-2 BPE via `tiktoken`, document delimiter `<|endoftext|>` |
| Model | GPT-2 small: 12 layers, 12 heads, 768 embedding width, 1024 context, padded vocabulary 50,304 |
| Parameters | 124,475,904 |
| Seed | 1337 |
| GPU | 1 x NVIDIA A100-SXM4-80GB (81,920 MiB reported) |
| PyTorch / CUDA | PyTorch 2.8.0+cu128 / CUDA 12.8 / driver 570.195.03 |
| Precision | BF16 autocast; FP32 parameters |

## Training Configuration

| Field | Value |
| --- | ---: |
| Microbatch `B` | 64 |
| Sequence length `T` | 1,024 |
| Gradient accumulation | 8 |
| World size | 1 |
| Global batch | 524,288 tokens |
| Optimizer | Fused AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 |
| LR schedule | 715-step linear warmup to 6e-4, then cosine decay to 6e-5 |
| Gradient clipping | Global norm 1.0 |
| Planned/completed optimizer updates | 19,073 / 19,073 (steps 0 through 19,072) |
| Processed tokens | 9,999,745,024 |

## Final Metrics

| Metric | Result |
| --- | ---: |
| Final training loss | 3.074965476989746 |
| Final validation loss | 3.0750441551208496 |
| Final HellaSwag normalized accuracy | 0.303326030671181 (30.3326%) |
| Final learning rate | 6.0000003953509464e-05 |
| Final gradient norm | 1.0098577737808228 |
| Best validation loss | 3.0750441551208496 at step 19,072 / 9.999220736B tokens |
| Best HellaSwag normalized accuracy | 0.303326030671181 at step 19,072 |
| Total wall-clock runtime | 18:05:26 (run timestamp to final metric write) |
| Timed training-loop runtime | 18:02:41.066 |
| Aggregate throughput | 153,934 tokens/sec, including periodic evaluation/generation |
| Steady-state throughput | 168,189 tokens/sec, excluding evaluation/generation steps |
| Peak allocated VRAM | 60,963.53 MiB (59.53 GiB) |

The final validation and HellaSwag records are measured immediately before the final optimizer update, at 9,999,220,736 processed tokens. The final training loss, learning rate, and gradient norm are measured after that final update, at 9,999,745,024 tokens. No values were interpolated.

## Completion Audit

- `metrics.jsonl` contains 19,073 training rows, 78 validation rows, and 78 HellaSwag rows. All numeric metrics are finite.
- The console ends with final validation, final HellaSwag, generation output, and the step 19,072 training record.
- Searches of the complete console log found no `Traceback`, CUDA OOM, standalone `nan`/`inf`, `Killed`, or `error` matches.
- No training process remained after completion. The wrapper's post-training plots and final summary exist, which are only produced after `train_gpt2.py` exits with status zero.
- `latest_progress_summary.json` is a preserved, stale mid-run monitoring snapshot from step 5,461. `final_chart_summary.json`, the complete metrics stream, console tail, and final checkpoint establish completion.

## Checkpoints

All checkpoint files are 497,958,271 bytes.

| Filename | Step | SHA256 |
| --- | ---: | --- |
| `model_05000.pt` | 5,000 | `07840883c257904b607c68c47e9a156930e163c1b3c91d0c19cf671fd8571e9d` |
| `model_10000.pt` | 10,000 | `bce7322ed7cc578088fabe1de24b6f8f17dbad0c1e1ac062b8e8d8343e202bf6` |
| `model_15000.pt` | 15,000 | `dd187f1b8b97f50e8014b2e56d6d446626bdb1a76f070d7ce27ab1d20678b348` |
| `model_19072.pt` | 19,072 | `924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871` |

The final checkpoint was loaded on CPU with the recorded GPT class using strict state-dict loading. It has 149 state-dict entries, no missing or unexpected keys, exactly 124,475,904 model parameters, and no non-finite tensors. Its config is `block_size=1024`, `vocab_size=50304`, `n_layer=12`, `n_head=12`, `n_embd=768`.

## Dataset Note

The verified files contain 9,953,989,344 tokens: 9,853,989,344 training tokens and 100,000,000 validation tokens. Upstream training processed 9,999,745,024 tokens, so **145,755,680 training tokens were repeated after the upstream DataLoader wrapped around**. This is expected behavior for the unchanged upstream loader and step count.

## Karpathy Reference Comparison

Karpathy's repository records the GPT-2 124M / approximately 10B-token setup but does not publish an exact numeric endpoint table. The reference values below are therefore visually estimated from the supplied Karpathy plot and are intentionally reported at limited precision.

| Metric | This run | Karpathy plot estimate | Absolute difference | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| Final training loss | 3.07497 | approximately 3.05 | approximately +0.025 | approximately +0.8% |
| Final validation loss | 3.07504 | approximately 3.10 | approximately -0.025 | approximately -0.8% |
| Final HellaSwag accuracy | 0.30333 | approximately 0.305 | approximately -0.00167 (-0.167 percentage point) | approximately -0.55% |

The comparison supports a successful reproduction: the loss curve has the same steep early fall and smooth diminishing improvement, final loss is in the same approximately 3.0-3.1 range, HellaSwag rises from chance-level approximately 0.25 to approximately 0.30 with similar noise, and the final HellaSwag value is within roughly two-tenths of a percentage point of the plotted reference endpoint. The small endpoint differences are well within the precision available from the reference image and do not justify rerunning or tuning the baseline.

## Preserved Outputs

- `results/baseline_learning_curve.csv` contains every saved evaluation point; training fields are joined only when a training record exists at the exact same token count.
- `results/final_plots/` contains PNG and PDF versions of train/validation loss, validation loss, HellaSwag accuracy, and throughput versus billions of processed tokens.
- The compact external archive contains the final checkpoint, logs, metadata, environment and Git reports, dataset verification, stage report, plots, this report, the learning-curve CSV, and checksum manifests. It excludes the FineWeb dataset, Hugging Face cache, and intermediate checkpoints.
