# Experiment 1 Design: GPT-2 124M + Full Attention Residuals

## Authoritative definition

This implementation follows equations (2)--(4), Section 3.1, Section 5, and
Figure 1 of the Kimi Team technical report, [Attention Residuals
(arXiv:2603.15031)](https://arxiv.org/abs/2603.15031), cross-checked against the
[official MoonshotAI repository](https://github.com/MoonshotAI/Attention-Residuals).
The official repository contains the paper and Block AttnRes pseudocode, but no
executable Full AttnRes reference implementation. The paper and official
pseudocode agree with the experiment specification; no material discrepancy was
found.

For destination sublayer `l`, let `v_0` be the embedding state and let `v_i` for
`i >= 1` be the output of the `i`-th preceding Transformer sublayer. The paper
defines

```text
phi(q_l, k_i) = exp(q_l^T RMSNorm(k_i))
alpha_(i->l) = phi(q_l, k_i) / sum_(j=0)^(l-1) phi(q_l, k_j)
h_l = sum_(i=0)^(l-1) alpha_(i->l) v_i
q_l = w_l
k_i = v_i
```

There is deliberately no `1/sqrt(d)` score scaling.

## Mapping to GPT-2

- `v_0` is the sum of GPT-2's token embedding and learned absolute-position
  embedding. It participates in every depth routing operation.
- Each of the 12 causal self-attention outputs and 12 MLP outputs is a separate
  value. They are not cumulative hidden states and are not collapsed by block.
- Each of the 24 receiving sublayers has its own learned pseudo-query
  `w_l in R^768` and its own learned RMSNorm scale for keys.
- The query is input-independent and is initialized to exactly zero, as required
  by Section 5. Consequently, the initial depth distribution is exactly uniform
  over the available sources.
- RMSNorm is applied to keys only. Values remain raw sublayer outputs. This
  experiment uses `eps=1e-5`, matching GPT-2's existing normalization epsilon;
  the report requires RMSNorm but does not prescribe a different epsilon.
- Softmax is over dimension 0 of a `depth x B x T` score tensor. It is never over
  tokens, channels, or attention heads. Every token has its own depth weights,
  while the single pseudo-query and mixture are shared across channels.
- After `v_24` is produced, a 25th learned Full AttnRes operation aggregates
  `v_0...v_24`. That result is the representation passed to GPT-2's unchanged
  `ln_f` and tied language-model head. This implements the output aggregation
  shown in Figure 1 and described for the final output layer in Section 3.2.

The 12 GPT-2 blocks, 12 heads, width 768, context 1024, learned position
embeddings, causal attention, GELU MLPs, LayerNorms, tied LM head, tokenizer, and
all original parameter initialization remain unchanged. `residual_mode` selects
either the frozen `standard` computation or `full_attnres`.

## Parameter overhead

With 24 sublayers plus the final output aggregation, there are 25 routers. Each
adds one 768-element query and one 768-element RMSNorm scale:

```text
queries:             25 * 768 = 19,200
RMSNorm scales:      25 * 768 = 19,200
total added:                    38,400
```

No bias, key projection, value projection, multi-head routing, or final
projection is added. For the padded-vocabulary baseline (124,475,904
parameters), the expected Full AttnRes count is 124,514,304, an increase of
approximately 0.03085%.

## Compute and memory overhead

For a destination with `s` sources, routing performs `s` RMS normalizations,
`s` width-768 dot products, a depth softmax, and an `s`-source weighted sum. The
25 destinations process `1 + ... + 25 = 325` source/destination pairs, so routing
is `O(L^2 B T d)` arithmetic while the Transformer sublayers are unchanged.

At `B=64`, `T=1024`, and `d=768`, one BF16 residual value is 96 MiB. The maximum
25-value stack is therefore 2,400 MiB = 2.34375 GiB. In this GPT-2
implementation the initial embedding residual is FP32 under autocast, so the
mixed-dtype raw stack is approximately 2.4375 GiB (one FP32 embedding plus 24
BF16 sublayer outputs).

The paper describes the Full AttnRes value storage as `O(Ld)` and notes that the
values overlap with activations already retained by ordinary, non-recomputed
backpropagation. A naive framework implementation can nevertheless retain
quadratic normalized-key intermediates. The implementation therefore
checkpoints only the RMSNorm-and-score calculation, recomputing keys during
backward while retaining the paper-required raw values. It never stacks
`depth x B x T x d`; only the small `depth x B x T` logits/weights are stacked.
Actual A100 allocated and reserved peaks are measured rather than inferred.

## Instrumentation and ablation

Instrumentation is opt-in. It records, per destination, only mean source
weights and mean entropy over batch and tokens:

```text
H_l = -sum_i alpha_(i->l) log(alpha_(i->l))
```

The full per-token routing tensor is not logged. The offline ablation utility
masks one source logit from all later depth-routing softmaxes and compares
validation loss. Source 0 remains the unavoidable sole input to the first
sublayer but is masked from every later router; this limitation is reported by
the utility. Ablation is never run inside normal training.
