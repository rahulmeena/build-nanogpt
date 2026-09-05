# 2D10 review — H wins the 100M screen; a report table needs correction

Reviewed 2026-09-05. Read the complete [final report](</Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b/results/experiment_2d10_retrieval_aware_gating_100m/EXPERIMENT_2D10_FINAL_REPORT.md>), saved bootstrap/decision/audit artifacts, router implementation and relevant analysis code. Verified clean worktree at `ef9d33ca5ad89a1b0b18db159310fc3e3dbe3a9c`. Independently reproduced all four CE means and all six contrast means/win counts from the four ordered 4096-sequence JSON arrays; checked common sequence/panel identities and target counts. Recomputed every primary flag from its saved adjusted interval. No bootstrap rerun, GPU execution, live pod query or independent full checkpoint-backup rehash was performed in this review.

## Scientific result

| Primary contrast | Mean CE | Adjusted 98.333333% CI |
|---|---:|---|
| D−T | +0.000002105033 | [−0.000054686209,+0.000058844481] |
| D−H | +0.000384687323 | [+0.000293674423,+0.000476805992] |
| T−H | +0.000382582291 | [+0.000289903849,+0.000474652899] |

H establishes benefit beyond the 0.0001 margin over both D and T after the prespecified three-comparison adjustment. T and D establish practical equivalence within ±0.0001 at this training age. The D−H lower bound exceeds the margin by 0.000193674423 CE; this is a clearer margin clearance than the earlier barely-passing 2D9 adoption comparison. The absolute mean benefit remains small: approximately 0.03846% lower perplexity than D. This is a comparison within the recurrent family, not a matched from-scratch victory over ordinary GPT-2.

H is the leading 100M candidate. The sealed 2D9 250M Dynamic checkpoint remains the current mature baseline pending a matched comparison at that age. Different panels' absolute CE values are not a training curve.

## Correction to the sealed report's flag table

The Markdown table incorrectly shows `Margin=False` and `Harm=True` for D−H and T−H. Both should be `Margin=True` and `Harm=False`. The numerical intervals, `PAIRED_BOOTSTRAP.json` flags, `SCREENING_DECISION.json` and user-facing summary are correct; the screening result is unchanged.

Correct table:

| Primary contrast | Positive | Beyond margin | Negative | Material harm | Equivalent | Second noninferior |
|---|---|---|---|---|---|---|
| D−T | False | False | False | False | True | True |
| D−H | True | True | False | False | False | True |
| T−H | True | True | False | False | False | True |

Cause: [experiment_2d10_analysis.py:216](</Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d3a_1b/scripts/experiment_2d10_analysis.py:216>) renders `v.values()` under a fixed header. JSON is saved with sorted keys, so after reload the dictionary order is `beyond_margin, material_harm, negative, positive, practical_equivalence, second_condition_noninferiority`, which does not match the column order. A report-generator correction should render explicit keys in this order:

```python
flag_columns = (
    "positive", "beyond_margin", "negative", "material_harm",
    "practical_equivalence", "second_condition_noninferiority",
)
# Join str(v[key]) for key in flag_columns, rather than v.values().
```

This review records the erratum separately. It does not rewrite the sealed report, code, result tag or numerical artifacts. The existing recorded audit did not catch the Markdown column mismatch; it is not evidence that the numerical flags were wrong. Correct the generator in a subsequent code revision before reusing it.

## What the result does and does not establish

T's retrieved-output MLP addition gives no material CE advantage over D at 100M under this recipe. That is not proof that such inputs can never help after more training, different windows or training from scratch.

H's complete trained architecture is better here, but this does not isolate context-dependent branch competition as the cause. H changes both local/recurrent normalization and the initial function, while T preserves the parent's initial function. H's small disposable-batch initial CE was already below the parent's; that one training-batch diagnostic cannot quantify or explain its final validation advantage. There is no trained constant-softmax or parameter-matched query-only MLP control in this experiment.

H does not win simply by using a larger raw recurrent coefficient. At B1 its eligible mean coefficients are about 0.768 local and 0.232 recurrent. Their ratio is about 0.302 (ratio of means, not mean token-wise ratio), comparable in scale to T's mean additive recurrent coefficient 0.317 with local coefficient1. Branch coefficient values are not fractions of predictive information; branch vectors and the coadapted backbone differ.

H's recurrent coefficients vary, but its B3/B5 variations are small (std0.000216/0.000864). Small or large variation alone does not identify causal utility. The result supports the tested softmax architecture while leaving the contributions of mean scaling, token variation, retrieved inputs and optimization unresolved. Sequence bootstrap CIs do not establish training-seed replication.

## Provenance, cost and proposed next step

Both new arms replayed exactly 191 updates /100,139,008 targets from original O1, with the saved 100M Static/Dynamic controls reused. The recorded audits establish matched batches/cursors/pass cadence, strict reopens, no updates to H's retired scalar gates, four fresh-panel evaluations and verified backups. Persistent state is 33,289,728 bytes/sequence in all conditions. T/H training took about30.45/30.14 minutes; the pod interval was95.73 minutes with two GPUs. Evaluation timings include candidate gate diagnostics and do not isolate router inference overhead. Pod `nagj1hv18p3z2c` is recorded stopped, volume `yhzyb27fb5` retained; current live status was not queried here.

```text
H100M SHA: d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c
T100M SHA: 7ed29fb5adc1e5aade2fad0e8db8e90233951fec5922ec13e75a7e861b2e6019
Global update: 2481
Inherited targets: 1,300,758,528
Result commit: ef9d33ca5ad89a1b0b18db159310fc3e3dbe3a9c
Branch: codex/experiment-2d10-retrieval-aware-gating-100m
Tag: experiment-2d10-retrieval-aware-gating-100m-final
```

A reasonable next decision is to continue only H to 250M total: another286 updates /149,946,368 targets, global updates2482–2767, replaying the saved 2D9 250M continuation ledger. Reuse the already-trained 250M D control and evaluate on a new common panel. S250M could be a descriptive reference if included in the future protocol; no S/D retraining is necessary. One GPU can perform this bounded continuation and evaluation. Preserve H's learned router and complete optimizer/RNG/loader state and its retired-gate exceptions. No continuation protocol, extra model control, new training or GPU operation was executed by this review.

Subsequent planning update: the user requested an H-only250M handoff. The [continuation protocol](</Users/rahul/Documents/GPT-2 Enhancement/project_context/EXPERIMENT_2D10_H_250M_CONTINUATION_PROTOCOL.md>) is now prepared; this does not change the sealed100M result or launch GPU work.
