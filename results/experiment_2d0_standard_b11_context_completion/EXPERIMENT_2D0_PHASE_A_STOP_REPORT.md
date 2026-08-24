# Experiment 2D0 Phase-A Stop Report

## Outcome

Experiment 2D0 stopped after its preregistered B11 window sweep because none of the allowed shortened windows caused enough validation damage to qualify for completion-module training.

The full-context Standard GPT-2 regression passed at 3.0750437753 validation loss. Shortening only B11 produced:

- W896: +0.0003025990 damage
- W768: +0.0006950885 damage
- W512: +0.0023090646 damage

All are below the required +0.01 minimum. W768 therefore failed the preferred-window rule, and W512 failed the specified fallback rule. No post-hoc window was introduced.

## Scientific interpretation

On this mature approximately-10B-token Standard GPT-2, removing as many as 512 oldest B11 KV positions changes overall canonical validation CE by only about 0.00231. B11 and B12 representations drift measurably, but next-token quality remains too close to the full-context oracle for the registered recovery fraction to be a meaningful Phase-B target.

This is a geometry-calibration result, not evidence for or against top-down completion. No completion module was trained, no SELF recurrence was evaluated, and the planned 100,139,008-target adaptation run did not start.

## Integrity and cost state

- Source checkpoint provenance, SHA, architecture, and historical validation regression passed.
- All four windows used the same canonical batches.
- Phase A performed zero training and produced no trained checkpoint.
- All four A100 GPUs were idle after the sweep.
- The three old pods are no longer required; all needed assets are verified on `tropical_red_gerbil`.

## Required decision

Explicit user authorization is required to introduce a shorter B11 window and freeze a new sweep rule. Until then, Experiment 2D0 remains stopped after Phase A and must not proceed to recurrence training.

