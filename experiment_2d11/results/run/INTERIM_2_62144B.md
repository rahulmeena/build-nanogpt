# H versus historical GPT-2 at 2,621,440,000 targets

Completed updates: 5,000. Both scores use matching training exposure.

| Metric | Fresh H | Historical GPT-2 |
|---|---:|---:|
| Validation CE (lower is better) | 3.32176397 | 3.35419464 |
| HellaSwag accuracy | 27.9725% | 27.5244% |

CE gap H minus G: -0.03243067 nats. HellaSwag difference H minus G: 0.4481 percentage points.

This is a historical aggregate comparison of single training trajectories. No fresh paired evaluation or paired confidence interval is available for the intermediate GPT-2 weights. It is not a completed matched 10B comparison.

A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.
