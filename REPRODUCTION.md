# GPT-2 124M / FineWeb-Edu 10B Reproduction

This repository is a faithful baseline harness around Andrej Karpathy's `build-nanogpt` GPT-2 124M pretraining script. The goal is to preserve upstream training behavior while adding reproducibility checks, structured metrics, run directories, plotting, and cloud safety wrappers.

## Clone

```bash
git clone https://github.com/karpathy/build-nanogpt.git build-nanogpt
cd build-nanogpt
git rev-parse HEAD
git status --short --branch
git diff
git switch -c gpt2-124m-10b-reproduction
```

Recorded upstream source:

```text
repository: https://github.com/karpathy/build-nanogpt.git
upstream branch: master
upstream SHA: 6104ab1b53920f6e2159749676073ff7d815c1fa
recorded at UTC: 2026-08-09T23:51:25Z
```

The metadata is also saved in `UPSTREAM_BASELINE.json`.

## Environment

Use a Linux CUDA machine with an NVIDIA A100 80GB GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
python scripts/verify_environment.py
```

For the strict baseline target:

```bash
python scripts/verify_environment.py \
  --require-cuda \
  --require-a100-80gb \
  --require-bf16 \
  --require-sdpa \
  --require-fused-adamw
```

The script reports OS, kernel, Python, PyTorch, CUDA, NVIDIA driver, GPU name/count/VRAM, BF16 support, cuDNN, scaled dot-product attention availability, fused AdamW availability, TF32/matmul precision, Git SHA, and current branch.

## Hardware

Primary target:

```text
1 x NVIDIA A100 80GB
```

The full reproduction asserts:

```text
B = 64
T = 1024
total_batch_size = 524288 tokens
world_size = 1
gradient_accumulation_steps = 8
```

If `B=64,T=1024` OOMs on A100 80GB, stop and record the stack trace, PyTorch version, CUDA version, GPU model, and peak memory. Do not reduce batch size or add memory-saving changes for this baseline.

## Data

Prepare exactly the upstream FineWeb-Edu sample:

```bash
python fineweb.py
```

Expected upstream settings:

```text
dataset: HuggingFaceFW/fineweb-edu
configuration: sample-10BT
split: train
local directory: edu_fineweb10B
tokenizer: tiktoken.get_encoding("gpt2")
document delimiter: <|endoftext|>
storage dtype: uint16
shard target size: 100,000,000 tokens
```

Place Hugging Face cache, `edu_fineweb10B/`, `runs/`, `log/`, and checkpoints on persistent cloud storage. Do not commit dataset shards or checkpoints.

## Verify Data

```bash
python scripts/verify_dataset.py --strict
```

For the full 10B-token baseline, use:

```bash
python scripts/verify_dataset.py \
  --strict \
  --min-total-tokens 9900000000 \
  --max-total-tokens 10200000000
```

This reports shard filenames, token counts, dtype, disk size, total tokens, upstream train/validation split discovery, first token IDs, boundary samples, and decoded samples.

## Smoke Test

Do this before any paid full run:

```bash
scripts/run_smoke_test.sh
```

The smoke runner checks environment, dataset, training constants, full GPT-2 124M instantiation, `B=64,T=1024` forward/backward, one 8-microstep optimizer update, short real-data training, validation, HellaSwag scoring on a small sample, checkpoint writing, metrics logging, and plot generation.

Optional knobs for smoke runtime only:

```bash
SMOKE_STEPS=2 SMOKE_VAL_STEPS=2 SMOKE_HELLASWAG_EXAMPLES=32 scripts/run_smoke_test.sh
```

These do not change production architecture or batch dimensions.

## Expected Smoke-Test Output

A successful smoke run writes a unique directory under `runs/smoke_<timestamp>/` containing:

```text
environment.json
dataset.json
console.log
metrics.jsonl
smoke_summary.json
checkpoints/smoke_model.pt
plots/plot_a_train_val_loss.png
plots/plot_b_val_loss.png
plots/plot_c_hellaswag_accuracy.png
plots/plot_d_throughput.png
```

The summary reports parameter count, initial loss, later smoke loss, validation loss, HellaSwag smoke accuracy, peak VRAM, tokens/sec, seconds per optimizer step, and projected full-run time from measured throughput.

## Full Run

Do not run this until the smoke test passes on the A100 80GB machine:

```bash
scripts/run_full_10b.sh
```

The full-run script verifies the expected branch and upstream base SHA, enforces clean Git state, checks environment, checks dataset completeness, asserts training constants, creates a unique run directory, records metadata, launches `train_gpt2.py`, captures stdout/stderr, writes checkpoints and logs into the run directory, and generates plots after training.

If you intentionally need to run with uncommitted local changes, make that explicit:

```bash
ALLOW_DIRTY_GIT=1 scripts/run_full_10b.sh
```

Prefer committing the reproduction harness before the real run so the recorded Git SHA fully identifies the experiment.

## Monitoring

Use `tmux` so SSH disconnects do not kill training:

```bash
tmux new -s gpt2
scripts/run_full_10b.sh
```

Detach:

```bash
Ctrl-b d
```

Reconnect:

```bash
tmux attach -t gpt2
```

GPU monitoring:

```bash
watch -n 1 nvidia-smi
```

Log monitoring:

```bash
tail -f runs/<run_name>/console.log
tail -f runs/<run_name>/metrics.jsonl
```

## Checkpoints

Upstream checkpoint cadence is preserved:

```text
step 5000
step 10000
step 15000
final step 19072
```

When launched through `scripts/run_full_10b.sh`, checkpoints are written to:

```text
runs/<run_name>/checkpoints/
```

The checkpoint schema remains upstream-compatible:

```text
model state_dict
model config
step
validation loss
```

Exact resume support is not implemented in this baseline because upstream checkpoints do not include optimizer state, RNG state, or data-loader position. Keeping uninterrupted training behavior faithful is more important for this control run. Use `tmux` and persistent storage for protection.

## Metrics

Karpathy's text log remains supported. Additional structured metrics are written when `NANOGPT_METRICS_FILE` is set:

```text
step
tokens
train_loss
val_loss
hellaswag_accuracy
lr
grad_norm
step_time_ms
tokens_per_second
gpu_allocated_mb
gpu_reserved_mb
gpu_peak_mb
```

The full-run wrapper writes:

```text
runs/<run_name>/log/log.txt
runs/<run_name>/metrics.jsonl
runs/<run_name>/console.log
```

## Plotting

Generate plots from saved logs:

```bash
python scripts/plot_reproduction.py runs/<run_name>
```

Outputs:

```text
plot_a_train_val_loss.png
plot_b_val_loss.png
plot_c_hellaswag_accuracy.png
plot_d_throughput.png
```

## Results

After the full experiment, create `results/BASELINE_REPORT.md` with:

```text
upstream Git SHA
hardware
software environment
dataset token count
training steps
tokens processed
microbatch
sequence length
gradient accumulation
effective global batch
final training loss
final validation loss
final HellaSwag accuracy
total wall-clock training time
average tokens/sec
peak GPU VRAM
```

Separate observed measurements from reference expectations. Do not tune or extend training to hit a target value.

## Differences From Upstream

Training semantics are intended to remain unchanged.

Instrumentation-only:

- `train_gpt2.py` can write JSONL metrics when `NANOGPT_METRICS_FILE` is set.
- `train_gpt2.py` can redirect log/checkpoint output with `NANOGPT_LOG_DIR` and `NANOGPT_CHECKPOINT_DIR`.
- `train_gpt2.py` prints parameter counts and CUDA memory snapshots.

Cloud convenience:

- `scripts/run_smoke_test.sh`
- `scripts/run_full_10b.sh`
- `scripts/verify_environment.py`
- `scripts/verify_dataset.py`
- `scripts/assert_training_config.py`
- `scripts/smoke_test.py`
- `scripts/plot_reproduction.py`

Documentation and safety:

- `REPRODUCTION.md`
- `UPSTREAM_BASELINE.json`
- `.gitignore`
- `requirements-repro.txt`
- `runs/.gitkeep`
- `results/.gitkeep`

No intentional training-semantic deviations from upstream.

