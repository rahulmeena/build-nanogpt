# Experiment 2B2A exploratory 20M override

The preregistered 15M gate is authoritative. If it fails, the canonical Experiment
2B2A result remains stopped and classified at update 29.

On 2026-08-17, before the 15M gate completed, the user explicitly requested that a
failed gate be followed by an exploratory continuation to the natural ~20M point so
the already-running four-GPU pod would remain productive while unattended.

The exploratory endpoint is update 38, or 19,922,944 cumulative writer-training
tokens. It must start from the immutable, audited update-29 checkpoint in a fresh
four-process launch, retain all scientific and distributed semantics, use distinct
artifact labels, and never overwrite or alter the canonical 15M result. It does not
retroactively pass the preregistered continuation gate and is not evidence for the
canonical 2B2A continuation decision.
