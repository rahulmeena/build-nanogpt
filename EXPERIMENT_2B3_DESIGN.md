# Experiment 2B3 — Joint Writer + Reader Co-Adaptation

Experiment 2B3 starts exclusively from the canonical Experiment 2B2A update-29
checkpoint (`86c663…ba84`). It preserves the four recurrent writer modules and
the existing top-down reader weights. The writer AdamW state resumes at step 29;
the reader receives a fresh AdamW state at step 0. The frozen GPT-2 / Full-AttnRes
base is never trainable.

The execution uses four A100-SXM4-80GB ranks, two B64×T1024 microsteps per rank,
and one combined FP32 `all_reduce(SUM)` per 524,288-target update. The combined
buffer has one disjoint rank slot per local flattened gradient; after the single
collective, every rank sums those slots in fixed rank order. This preserves the
required single synchronization while making fresh-reader Adam-step comparisons
reproducible instead of depending on NCCL's floating-point reduction order.
Writer and reader gradients are clipped separately to 1.0 after synchronization. Temporal
credit remains exactly one token because every next writer input is detached and
Blocks 2–12 historical K/V are detached.

Before result training, the protocol requires a deterministic FP32 joint-gradient
test, future-causality and row-isolation checks, and a no-step 1-GPU versus 4-GPU
loss/gradient comparison followed by discarded cloned optimizer steps. Both
writer and reader gradients and optimizer updates must independently satisfy the
frozen equivalence tolerances.

The result run consumes exactly nine global batches beginning at writer update
29. It must checkpoint and terminate after joint update 5, then resume in four
fresh processes for updates 6–9. The final next-batch hash must equal the
historical writer-only update-38 cursor
`7f6d8d…a7`, establishing a matched-data counterfactual. No tenth update,
longer BPTT, base-layer unfreezing, teacher training call, or HellaSwag run is
authorized.
