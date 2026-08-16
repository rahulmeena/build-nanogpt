# Experiment 2B2 — Memory-Only Writers with One-Step Temporal Credit

## Outcome

**MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY**

Training only the four rank-8 memory writers for exactly 10 updates / 5,242,880
tokens reduced canonical recurrent validation loss from the frozen 2B1 value of
**5.7013087273** to **5.5900331020**, a writer gain of **0.1112756252**. The
improvement is sequence-specific rather than generic compensation:
shuffled-minus-real increased from **0.0297215700** to **0.0359389782**, and real
memory beat shuffled memory on all 20/20 validation batches.

The writers recovered **20.2447%** of the Block-1 masking damage, up from 2B1's
14.3727% and close to the prior teacher-reader reference of 21.3001%. All four
learned writer residuals were useful on 20/20 ablation batches, with v16 making
the largest contribution. The reset sweep remained monotonic and the advantage
over frozen 2B1 grew with memory persistence, reaching approximately 0.1113 loss
for uninterrupted recurrence.

Every safety, replay, checkpoint, restart, gradient-boundary, validation, cache,
and frozen-weight audit passed. No teacher forward was used during training, no
gradient propagated beyond one recurrent transition, no optimizer update beyond
10 was run, and HellaSwag was not launched.

## Git

| Field | Value |
|---|---|
| Experiment-2B0 frozen tag | `experiment-2b0-zero-shot` |
| Experiment-2B0 full commit | `a8b271ee71ae1af77da8ddad022ce549be390682` |
| Experiment-2B1 frozen tag | `experiment-2b1-self-reader-5m` |
| Experiment-2B1 full commit | `ad2eb56b1fdbf20dde515693f6d2c9bd9034f444` |
| Experiment-2B2 branch | `experiment-2b2-memory-writers-1step` |
| Experiment-2B2 implementation commit | `5305709f93cee736d33b93402324a7d3fed40235` |
| Experiment-2B2 results commit | The commit containing this report; recorded in the handoff |

The final audit ran against the implementation commit recorded in both result
checkpoints. The frozen 2B0 and 2B1 tags resolved to their exact audited commits.

## Initialization

| Field | Value |
|---|---|
| Source 2B1 checkpoint | `/workspace/build-nanogpt-exp2a0/runs/experiment_2b1_5m/result/checkpoints/checkpoint_updates_000010.pt` |
| Source SHA-256 | `5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a` |
| Source next-global-batch SHA-256 | `3c4d0cd7905f16bfcfdd283cbc0799ff9f85c91ac3c521e886f0e403fc11ae57` |
| Writer sources | v16, v17, v20, v24 |
| Writer architecture | RMSNorm(x) → bias-free 768→8→768 residual adapter |
| Writer rank | 8 |
| Writer parameters | 49,152 |
| Initialization seed | 20260202 |
| W_down / W_up initialization | Normal(0, 0.02) / exact zero |
| Zero-writer identity regression | PASS; logits, final memory, and canonical loss exact |

The existing GPT-2/Full-AttnRes base and 1,537-parameter 2B1 reader loaded
exactly and remained frozen. The four writers started with an exact zero effect,
and a fresh AdamW optimizer was created without restoring the 2B1 reader moments.

## Gradient semantics and preflight

| Required boundary | Result |
|---|---|
| Writer input is detached raw student source | PASS |
| Loss(t+1) → writer memory(t) | PASS; finite and nonzero |
| Loss(t+2) → writer memory(t) | NONE |
| Historical-K/V temporal gradient | NONE |
| Reader gradient | NONE |
| Base / Full-AttnRes gradient | NONE |
| Stored writer output retains graph until one use | PASS |
| Block-1 historical K/V | NONE |
| Blocks 2–12 historical K/V | finite and detached |

The explicit FP32 writer-gradient reference matched with zero absolute and
relative error. Chunked backward matched the token-by-token reference to maximum
absolute error `1.1368683772e-13` and relative error `1.0601973706e-7`.
Writer perturbation changed next-token logits while leaving same-token logits
bit-exact. Future-suffix causality, row isolation, fresh reset, serialized
continuation, incremental equivalence, and T=8/16/32/64 stability all passed.

The disposable B2×T64 smoke completed three updates, checkpointed after update
2, resumed update 3 in a fresh process with an exact next-batch match, and was
discarded before result training.

## Training

| Field | Value |
|---|---|
| Hardware | NVIDIA A100-SXM4-80GB |
| Software | PyTorch 2.8.0+cu128; CUDA 12.8 |
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0 |
| Learning rate | constant 1.0e-4 |
| Gradient clip | 1.0 |
| Backward chunk | 16 tokens |
| Updates | exactly 10 |
| Targets per update | 524,288 |
| 2B2 tokens | 5,242,880 |
| Training wall time | 42,414.33 s = 11 h 46 m 54 s |
| Peak allocated VRAM | 40,879.88 MiB |
| Peak reserved VRAM | 80,562 MiB |
| Teacher training forwards | 0 |
| Final checkpoint SHA-256 | `a125c81acb9e4ec3395bd8b38dee8fade62012c642b102a1b6c4c0e0997f0637` |

Every update consumed its preregistered replay hash, included exactly 524,288
targets, produced finite writer gradients and optimizer moments, left all reader
and base gradients absent, and kept recurrent memory plus historical K/V finite.

| Update | Loss | Pre-clip grad | W_up norm | Max residual/source | v16 | v17 | v20 | v24 | Wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.712286 | 0.333165 | 0.015669 | 0.000000 | 0.636004 | 0.022570 | 0.129420 | 0.212006 | 4231.5 |
| 2 | 5.755789 | 0.340618 | 0.030700 | 0.003045 | 0.638293 | 0.021549 | 0.130589 | 0.209569 | 4220.1 |
| 3 | 5.684822 | 0.371281 | 0.045611 | 0.005989 | 0.631885 | 0.022087 | 0.130765 | 0.215263 | 4222.3 |
| 4 | 5.643227 | 0.405336 | 0.060428 | 0.008729 | 0.642783 | 0.021868 | 0.132609 | 0.202740 | 4223.1 |
| 5 | 5.705835 | 0.433982 | 0.075187 | 0.011812 | 0.635215 | 0.019219 | 0.133791 | 0.211776 | 4222.6 |
| 6 | 5.643296 | 0.462091 | 0.089972 | 0.018017 | 0.639592 | 0.021252 | 0.137858 | 0.201298 | 4312.0 |
| 7 | 5.680048 | 0.513454 | 0.104779 | 0.018934 | 0.635918 | 0.020591 | 0.137083 | 0.206409 | 4265.9 |
| 8 | 5.610978 | 0.560605 | 0.119632 | 0.022907 | 0.640457 | 0.019219 | 0.137025 | 0.203299 | 4255.0 |
| 9 | 5.618050 | 0.592301 | 0.134518 | 0.025782 | 0.636365 | 0.019462 | 0.135038 | 0.209135 | 4230.6 |
| 10 | 5.602771 | 0.621414 | 0.149428 | 0.031382 | 0.626214 | 0.019649 | 0.141019 | 0.213117 | 4231.2 |

## Forced restart

| Field | Value |
|---|---|
| Update-5 checkpoint SHA-256 | `565195ae3b66329506daafbb5075bd1392585e64f4069c0d2fbf3b7b5e3773ca` |
| Restored next-batch SHA-256 | `70d03aff4071751467ede46c4931e8c875b5a7f0d046265b773228d53461b63c` |
| Save process / resume process | PID 33043 / PID 33346 |
| Model / optimizer / four-loader strict reload | PASS |
| Python/NumPy/Torch CPU/Torch CUDA RNG reload | PASS |
| Fresh-process update-6 batch verification | PASS; exact restored hash |

The final checkpoint strict-reloaded at local update 10 with Adam steps 10 for
all eight writer tensors and next-global-batch SHA-256
`e3289bee6ed5a5b2fa1d2c05a615cd3f10f07c51b71aa091ee40380ebeedc21b`.

## Canonical validation

All conditions used 20×B64×T1024 BF16 batches with pinned aggregate SHA-256
`3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

| System | Validation loss |
|---|---:|
| Full context | 4.0786544085 |
| Masked L1 / no feedback | 5.9736744881 |
| Frozen 2B1 self, writers bypassed | 5.7013087273 |
| Trained writer real self | **5.5900331020** |
| Trained writer shuffled self | 5.6259720802 |
| Trained writer bypass | 5.7013087273 |
| Trained writer gate-zero | 5.9736480713 |
| Teacher sources passed through trained writers | **5.5216585398** |

The frozen full-context, masked, 2B1 bypass, and gate-zero controls reproduced
exactly. Writer bypass returning exactly to 2B1 and gate-zero returning exactly
to the masked control confirm that the improvement enters only through the
recurrent feedback path. Teacher sources plus trained writers improved by about
0.0504 over the prior raw teacher-memory result of 5.5720659256, showing that the
writer transform is compatible with both recurrent and teacher source states.

## Primary metrics

| Metric | Value |
|---|---:|
| Writer gain vs frozen 2B1 | **0.1112756252** |
| 2B1 specific gap | 0.0297215700 |
| Writer specific gap | **0.0359389782** |
| Specific-gap gain | **0.0062174082** |
| Writer recovery | 0.3836413860 |
| Writer recovery / masking damage | **20.2447%** |

The gain is more than eleven times the preregistered 0.01 improvement threshold,
and the sequence-specific gap grew rather than collapsing.

## Paired real vs shuffled

| Metric | Value |
|---|---:|
| Real wins | 20 |
| Shuffled wins | 0 |
| Ties | 0 |
| Mean shuffled-minus-real | 0.0359389782 |
| Median | 0.0383517742 |
| Sample standard deviation | 0.0104833314 |
| Minimum | 0.0148568153 |
| Maximum | 0.0508966446 |

## Per-source writer behavior

| Source | Delta RMS | Source RMS | Delta/source | Cosine(source, adapted) |
|---|---:|---:|---:|---:|
| v16 | 0.001661624 | 0.193306805 | 0.008785535 | 0.999943763 |
| v17 | 0.002929814 | 0.191811036 | 0.015699967 | 0.999873528 |
| v20 | 0.001743716 | 0.194265940 | 0.009319029 | 0.999938840 |
| v24 | 0.001371943 | 0.186641607 | 0.007584243 | 0.999952897 |

The writers learned small corrective transformations: mean residual RMS was
only 0.76–1.57% of source RMS, and source/adapted cosine remained above 0.99987.
The maximum per-example residual/source ratio observed in canonical validation
was 3.70%, so the gain did not require a large representation rewrite.

## Routing

| Quantity | Frozen 2B1 zero-writer | Trained writer self |
|---|---:|---:|
| v16 | 0.6340336651 | 0.6263023943 |
| v17 | 0.0206797247 | 0.0174007127 |
| v20 | 0.1305237375 | 0.1416889235 |
| v24 | 0.2147629216 | 0.2146078154 |
| Entropy | 0.4845177144 | 0.4797579363 |
| Top-down RMS | 0.1623093784 | 0.1624164350 |
| Feedback RMS | 0.0256773381 | 0.0256942729 |

The frozen reader still emphasized v16, but useful behavior did not come from a
single writer: disabling each learned residual increased loss on all 20 batches.

## Reset horizon

Only recurrent high→low memory was reset; Blocks 2–12 K/V and absolute position
continued. The 2B2 trajectory was monotonic, and its advantage over frozen 2B1
grew steadily with persistence.

| Interval | Frozen 2B1 | 2B2 writers | 2B2 − 2B1 |
|---:|---:|---:|---:|
| 1 | 5.9737106323 | 5.9737106323 | +0.0000000000 |
| 2 | 5.8261914968 | 5.7768767834 | -0.0493147135 |
| 4 | 5.7499492407 | 5.6721416473 | -0.0778075933 |
| 8 | 5.7218983650 | 5.6283501625 | -0.0935482025 |
| 16 | 5.7104103088 | 5.6082828522 | -0.1021274567 |
| 32 | 5.7059802055 | 5.5992109060 | -0.1067692995 |
| 64 | 5.7035196781 | 5.5941849947 | -0.1093346834 |
| 128 | 5.7021847963 | 5.5918534994 | -0.1103312969 |
| Never | 5.7013566494 | **5.5900463343** | -0.1113103151 |

The batched nine-horizon endpoint differs slightly from standalone canonical
loss because the larger BF16 GEMM shape selects different reductions.

## Writer-residual ablation

Each ablation disabled only the learned residual while retaining the raw source.
It did not remove or renormalize the whole source.

| Disabled writer residual | Ablated loss | Delta vs trained real | Positive batches |
|---|---:|---:|---:|
| v16 | 5.6620023489 | +0.0719692469 | 20/20 |
| v17 | 5.5912817478 | +0.0012486458 | 20/20 |
| v20 | 5.6040416002 | +0.0140084982 | 20/20 |
| v24 | 5.6069802999 | +0.0169471979 | 20/20 |

v16 supplied the largest learned correction, followed by v24 and v20. v17's
effect was small but consistently favorable. Whole-source ablation was not run;
the protocol made it optional, while writer-residual ablation was the required
test of learned writer transformations.

## Representation drift

Each cell is `cosine / RMS difference / norm ratio` against the full-context
teacher state. The trained writers changed the raw recurrent state only
slightly, consistent with the high source/adapted cosines above.

### Teacher vs raw recurrent student

| Positions | v16 | v17 | v20 | v24 |
|---|---|---|---|---|
| 1–16 | 0.769352/0.133738/1.009902 | 0.776086/0.142733/1.037734 | 0.788235/0.131204/0.958914 | 0.757091/0.139524/0.912884 |
| 17–32 | 0.597026/0.183190/0.980629 | 0.681316/0.176011/0.995346 | 0.623493/0.180119/0.913231 | 0.567770/0.186858/0.886814 |
| 33–64 | 0.555078/0.191877/0.964243 | 0.686566/0.176636/0.958481 | 0.583535/0.188404/0.895069 | 0.523343/0.195374/0.883672 |
| 65–128 | 0.545717/0.192225/0.940553 | 0.717429/0.168925/0.896612 | 0.572595/0.190287/0.882942 | 0.508066/0.197406/0.875162 |
| 129–256 | 0.560408/0.187207/0.935105 | 0.749162/0.159961/0.845053 | 0.585627/0.186875/0.882011 | 0.521459/0.195283/0.871900 |
| 257–512 | 0.581223/0.182253/0.932925 | 0.772027/0.152246/0.811645 | 0.607645/0.181886/0.886121 | 0.546302/0.192424/0.884990 |
| 513–1023 | 0.591837/0.181931/0.902449 | 0.787098/0.142020/0.807039 | 0.631031/0.180120/0.885249 | 0.578815/0.187326/0.912254 |

### Teacher vs adapted writer memory

| Positions | v16 | v17 | v20 | v24 |
|---|---|---|---|---|
| 1–16 | 0.769360/0.133746/1.010053 | 0.776099/0.142741/1.037873 | 0.788273/0.131193/0.958889 | 0.757083/0.139531/0.912945 |
| 17–32 | 0.597033/0.183204/0.980809 | 0.681310/0.176031/0.995595 | 0.623536/0.180106/0.913166 | 0.567750/0.186866/0.886855 |
| 33–64 | 0.555084/0.191892/0.964428 | 0.686520/0.176666/0.958756 | 0.583566/0.188393/0.895000 | 0.523308/0.195386/0.883726 |
| 65–128 | 0.545722/0.192240/0.940740 | 0.717333/0.168967/0.896949 | 0.572640/0.190272/0.882860 | 0.508020/0.197420/0.875224 |
| 129–256 | 0.560415/0.187221/0.935289 | 0.748997/0.160019/0.845474 | 0.585683/0.186856/0.881909 | 0.521400/0.195301/0.871977 |
| 257–512 | 0.581221/0.182268/0.933107 | 0.771807/0.152316/0.812163 | 0.607708/0.181864/0.885993 | 0.546227/0.192447/0.885081 |
| 513–1023 | 0.591791/0.181953/0.902631 | 0.786877/0.142089/0.807712 | 0.631075/0.180103/0.885136 | 0.578737/0.187350/0.912344 |

Raw unrounded position-bin data are preserved in `evaluation/canonical.json` and
`FINAL_AUDIT.json`. These diagnostics do not assume that more teacher-like
memory is intrinsically better.

## Integrity and scope audit

The final audit passed every required check:

- exactly 49,152 writer parameters were trainable;
- exactly 10 updates and 5,242,880 targets were consumed;
- all ten replay hashes and all per-update target counts were exact;
- teacher training forward calls were zero;
- all reader and frozen-base gradients were absent;
- the one-token temporal gradient existed and the two-token path was absent;
- stored historical K/V had no `grad_fn`; recurrent memory graph lifetime was exact;
- update-5 and update-10 checkpoints strict-reloaded and rehashed;
- the forced update-5 process exit and update-6 fresh process were distinct;
- frozen base and reader hashes remained bit-identical;
- canonical and diagnostic validation hashes were exact;
- authorized optimizer updates were exhausted at 10; beyond-10 updates: 0;
- HellaSwag artifacts were absent.

HellaSwag was not run. Because the 5M result is clearly positive, the audit
recommends considering it after 5M, but only with separate approval. Its planning
estimate is approximately 16–32 A100 hours for one recurrent control or 80–160
A100 hours for five controls; these are estimates, not benchmarks.

## Final classification and decisions

### MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY

The writer gain of 0.111276 cleared the 0.01 threshold, the sequence-specific
gap improved rather than collapsing, all four writer residuals were consistently
useful, longer memory persistence received progressively larger gains, and every
integrity invariant passed.

**Decision A — Continue the same memory-writer experiment from 5M toward
approximately 25M tokens? YES.** The preregistered 5M continuation gate is
decisively positive. This report does not launch that continuation.

**Decision B — Is there enough evidence to jointly train the memory writers and
the existing 1,537-parameter reader? YES.** The writer-only result establishes a
large causal gain while the frozen reader remains stable. Joint training should
be defined as a new experiment with its own controls and approval.

**Decision C — Is there enough evidence to increase temporal credit beyond one
token? NO.** One-step credit already produced a large improvement, so longer
horizons would confound the mechanism before the one-step result is replicated
or extended. Do not infer permission to launch multi-token BPTT.

Do not launch any additional optimizer step from this checkpoint without a new,
explicitly approved experiment.

# EXPERIMENT 2B2 5M COMPLETE
