# H versus historical GPT-2 at 5,242,880,000 targets

Completed updates: 10,000. Both scores use matching training exposure.

| Metric | Fresh H | Historical GPT-2 |
|---|---:|---:|
| Validation CE (lower is better) | 3.16973968 | 3.19380760 |
| HellaSwag accuracy | 29.4961% | 28.8190% |

CE gap H minus G: -0.02406792 nats. HellaSwag difference H minus G: 0.6772 percentage points.

This is a historical aggregate comparison of single training trajectories. No fresh paired evaluation or paired confidence interval is available for the intermediate GPT-2 weights. It is not a completed matched 10B comparison.

A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.
