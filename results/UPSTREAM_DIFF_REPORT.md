# Upstream Difference Report

Baseline upstream:

```text
repository: https://github.com/karpathy/build-nanogpt.git
branch: master
SHA: 6104ab1b53920f6e2159749676073ff7d815c1fa
working branch: gpt2-124m-10b-reproduction
```

Equivalent comparison command:

```bash
git diff 6104ab1b53920f6e2159749676073ff7d815c1fa
```

## Summary

No intentional training-semantic deviations from upstream.

The only modified upstream source file is `train_gpt2.py`. Its model architecture, initialization, tokenizer, dataset loader, sequence length, microbatch, optimizer, learning-rate schedule, gradient clipping, BF16 autocast, validation loop, HellaSwag scoring, generation logic, seed, and max-step count are unchanged.

## Changed Files

| File | Category | Explanation |
| --- | --- | --- |
| `train_gpt2.py` | instrumentation-only | Adds JSON import, parameter/memory prints, optional env-controlled log/checkpoint/metrics paths, and JSONL metric writes. Default `log/` behavior remains when env vars are unset. |
| `.gitignore` | cloud convenience | Prevents committing FineWeb shards, run directories, checkpoints, logs, caches, and HellaSwag downloads. |
| `requirements-repro.txt` | documentation | Lists the small dependency set needed by upstream plus repro tooling. `transformers` is included because upstream `hellaswag.py` imports it for its standalone evaluator. |
| `UPSTREAM_BASELINE.json` | documentation | Records upstream repository URL, branch, SHA, status, diff, and capture time. |
| `REPRODUCTION.md` | documentation | Step-by-step reproduction, smoke-test, full-run, monitoring, storage, plotting, checkpoint, and deviation guidance. |
| `scripts/assert_training_config.py` | cloud convenience | Parses `train_gpt2.py` without importing it and fails if baseline constants or key training snippets drift. |
| `scripts/verify_environment.py` | cloud convenience | Reports and optionally enforces OS, Python, PyTorch, CUDA, NVIDIA, GPU, BF16, SDPA, fused AdamW, dependency, and Git state requirements. |
| `scripts/verify_dataset.py` | cloud convenience | Inspects `edu_fineweb10B` using upstream split discovery, checks shard dtype/token counts, and decodes samples. |
| `scripts/smoke_test.py` | cloud convenience | Runs full-size GPT-2 smoke checks without importing the training launcher or starting the 19,073-step run. |
| `scripts/run_smoke_test.sh` | cloud convenience | Creates a unique smoke run directory and wires strict environment, dataset, model, training, checkpoint, metrics, and plotting checks together. |
| `scripts/run_full_10b.sh` | cloud convenience | Guarded full-run wrapper that verifies Git, hardware, dataset, and constants, then captures metadata, console output, metrics, logs, checkpoints, and plots. |
| `scripts/plot_reproduction.py` | cloud convenience | Generates training/validation loss, validation-only loss, HellaSwag accuracy, and throughput plots from saved logs. |
| `runs/.gitkeep` | documentation | Keeps the ignored run-root directory present in Git. |
| `results/.gitkeep` | documentation | Keeps the results directory present in Git. |
| `results/UPSTREAM_DIFF_REPORT.md` | documentation | This report. |

## Training-Semantic Changes

None intentionally introduced.

If future work changes any architecture, optimizer, data-ordering, precision, schedule, evaluation, or batch behavior, it should be treated as a separate experiment and recorded as a training-semantic change.

