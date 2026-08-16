# Experiment 2B2A: four-GPU writer scaling

Experiment 2B2A continues the exact Experiment 2B2 writer-only objective from the
audited update-10 checkpoint. The base model and reader remain frozen. The four
memory writers are the only trainable tensors, and temporal credit remains exactly
one token.

The four checkpoint replay-loader states map directly to ranks 0–3. Each rank
consumes two `B64 × T1024` slices. Every slice's summed token loss is divided by
524,288, so the four rank-local accumulated gradients sum to the original global
mean gradient. After all recurrent backward chunks finish, the eight writer
gradients are flattened in stable parameter-name order into one FP32 buffer and
reduced with exactly one NCCL `all_reduce(SUM)`. There is no division after the
reduction. Global clipping to 1.0 follows synchronization, after which every rank
executes the same replicated AdamW step.

No stochastic operation is active in the result-bearing forward/backward path.
The migration nevertheless establishes deterministic, distinct Python, NumPy,
Torch CPU, and Torch CUDA RNG streams per rank and serializes all four streams at
each milestone.

Before update 11, an untouched checkpoint clone is used for two paths: the audited
one-GPU serialized eight-slice reference and the four-rank candidate. Loss,
unclipped writer gradient, and a temporary clipped AdamW step must meet the frozen
migration tolerances. Both temporary stepped states and all advanced loader clones
are discarded. Result training then reloads the original update-10 checkpoint.

Milestone checkpoints are coordinated at updates 20, 29, and 48. Training stops
for canonical real/shuffled evaluation at every reached milestone, and continuation
requires the preregistered gate. Update 49 is forbidden. HellaSwag is not run.
