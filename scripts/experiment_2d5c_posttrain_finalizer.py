#!/usr/bin/env python3
"""Bind the frozen 2D5C finalizer to the exact post-training replacement pod.

The original finalizer remains unchanged as historical evidence for the pod
that disappeared during evaluation.  This append-only wrapper changes only
the live pod identity and its derived exact stop command before delegating to
the frozen, stdlib-only finalizer.  Scientific checks and the retained network
volume identity remain unchanged.
"""

from __future__ import annotations

from typing import Sequence

import experiment_2d5c_finalizer as finalizer


POD_ID = "7kk5yyti00rnrp"
POD_NAME = "grand_amber_catshark"
EXACT_STOP_COMMAND = f"runpodctl pod stop {POD_ID} -o json"


def bind_finalizer_identity() -> None:
    """Bind all pod-dependent finalizer globals to the exact stopped pod."""

    finalizer.POD_ID = POD_ID
    finalizer.POD_NAME = POD_NAME
    finalizer.EXACT_STOP_COMMAND = EXACT_STOP_COMMAND


def main(argv: Sequence[str] | None = None) -> int:
    bind_finalizer_identity()
    return finalizer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
