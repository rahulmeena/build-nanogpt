# Experiment 2D1 — Triangle recurrent Transformer co-training

Experiment 2D1 starts from the immutable Experiment 2D0D parent commit and the
exact mature Standard GPT-2 124M checkpoint. It trains all GPT-2 weights plus a
single previous-top-state-to-input fusion path using next-token cross entropy
only. There is no teacher, reconstruction loss, reader/writer adapter,
Full-AttnRes, temporal AttnRes, or HellaSwag evaluation.

The recurrent source is `ln_f(B12(t-1))`. An affine-free RMS normalization is
followed by an initially scale-calibrated identity `W_u`; token embeddings pass
through an initially identity `W_g`, and `2*sigmoid(W_g(e_t))` gates the recurrent
value. A frozen curriculum coefficient `rho` blends token embeddings into the
recurrent input until Stage D, where the final recurrent architecture has
`rho=1` and no additive token-embedding shortcut on recurrent positions.

Training uses two temporal-parallel passes for ordinary updates and three passes
on every 32nd update. Recurrent sources remain attached so later-pass CE teaches
earlier passes what to write. Each recurrent pass uses a checkpointed random
plain-prefix length. The logical batch is fixed at 524,288 targets/update for
exactly 4,769 updates (2,500,329,472 targets).

The frozen window curriculum, blend schedule, optimization schedule,
milestones, forced restarts, and all hashes are machine-registered in
`configs/exp2d1_triangle_recurrent.json`. The final windows are:

`[64,82,106,136,175,226,290,374,481,619,796,1024]`.

Result training is guarded by all architecture/data/causality/gradient/smoke
preflights and by verification of an authenticated mechanism able to stop the
exact A100 RunPod pod after completion or terminal failure.
