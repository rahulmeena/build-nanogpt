# H versus historical GPT-2 at 7,864,320,000 targets

Completed updates: 15,000. Both scores use matching training exposure.

| Metric | Fresh H | Historical GPT-2 |
|---|---:|---:|
| Validation CE (lower is better) | 3.08456509 | 3.10865855 |
| HellaSwag accuracy | 30.3226% | 29.6355% |

CE gap H minus G: -0.02409347 nats. HellaSwag difference H minus G: 0.6871 percentage points.

This is a historical aggregate comparison of single training trajectories. No fresh paired evaluation or paired confidence interval is available for the intermediate GPT-2 weights. It is not a completed matched 10B comparison.

A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.
