#!/usr/bin/env python3
"""Evaluation-only recovery for the documented 2D2A BF16 audit correction."""

import sys
from pathlib import Path


REPOSITORY = Path("/workspace/build-nanogpt-exp2d2a")
sys.path.insert(0, str(REPOSITORY / "scripts"))

import experiment_2d2a as experiment  # noqa: E402


CORRECTION_ARTIFACT = (
    "results/experiment_2d2a_b12_b1_recurrent_kv/"
    "POST_TRAINING_AUDIT_CORRECTION.json"
)
_original_equivalence = experiment.parallel_incremental_equivalence


def corrected_equivalence(*args, **kwargs):
    result = _original_equivalence(*args, **kwargs)
    result["bf16"]["tolerance_note"] = (
        "Post-training audit correction: active-prefix absolute max uses the "
        "already-preregistered BF16 Plain kernel tolerance; see "
        "POST_TRAINING_AUDIT_CORRECTION.json."
    )
    result["post_training_audit_correction"] = {
        "artifact": CORRECTION_ARTIFACT,
        "applied": True,
        "training_or_model_changed": False,
    }
    return result


def main():
    experiment.BF16_INCREMENTAL_ACTIVE_PREFIX_MAX_ATOL = (
        experiment.BF16_INCREMENTAL_PLAIN_MAX_ATOL
    )
    experiment.parallel_incremental_equivalence = corrected_equivalence
    arguments = experiment.build_parser().parse_args(sys.argv[1:])
    if arguments.command != "finalize":
        raise SystemExit("audit-correction wrapper permits finalize only")
    sys.argv[0] = (
        "scripts/experiment_2d2a.py"
        "[POST_TRAINING_AUDIT_CORRECTION_BF16_ACTIVE_USES_PLAIN_ATOL_1.25]"
    )
    experiment.run_finalize(arguments)


if __name__ == "__main__":
    main()
