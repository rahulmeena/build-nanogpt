#!/usr/bin/env python3
"""Append-only exact-pod binding for the resumed 2D5C post-training run.

The original guard remains frozen as historical evidence for the pod that
disappeared mid-evaluation.  This wrapper changes only the live pod identity
and the derived exact stop command before delegating to that tested guard.
The retained network-volume identity and every guard policy remain unchanged.
"""

from __future__ import annotations

from typing import Sequence

import experiment_2d5c_runpod_guard as guard


POD_ID = "7kk5yyti00rnrp"
POD_NAME = "grand_amber_catshark"
EXACT_STOP_ARGV = ("runpodctl", "pod", "stop", POD_ID, "-o", "json")
EXACT_STOP_COMMAND = " ".join(EXACT_STOP_ARGV)


def bind_guard_identity() -> None:
    """Bind all pod-dependent guard globals to the exact replacement pod."""

    guard.POD_ID = POD_ID
    guard.POD_NAME = POD_NAME
    guard.EXACT_STOP_ARGV = EXACT_STOP_ARGV
    guard.EXACT_STOP_COMMAND = EXACT_STOP_COMMAND


def main(argv: Sequence[str] | None = None) -> int:
    bind_guard_identity()
    return guard.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
