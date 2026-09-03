"""Experiment 2D8 overlap-width geometries.

N and O1 reproduce the sealed Experiment 2D7 implementations. O2 changes
only the minimum eligible recurrent lags at B3 and B5.
"""

from __future__ import annotations

import copy
import hashlib
import json

import experiment_2d6_core as d6_core
import experiment_2d7_core as d7_core


GEOMETRIES = {
    "N": {0: 2, 2: 32, 4: 64},
    "O1": {0: 1, 2: 31, 4: 63},
    "O2": {0: 1, 2: 30, 4: 62},
}
CONDITIONS = {"N": "N_REAL", "O1": "O1_REAL", "O2": "O2_REAL"}
LOCAL_WINDOWS = dict(d7_core.LOCAL_WINDOWS)
PARAMETER_COUNT = d7_core.PARAMETER_COUNT


def architecture_manifest(arm: str) -> dict:
    arm = str(arm).upper()
    if arm not in GEOMETRIES:
        raise ValueError(f"unknown 2D8 arm: {arm}")
    # Start from the accepted post-2D6 manifest, then change only eligibility.
    source_arm = "N" if arm == "N" else "O"
    value = copy.deepcopy(d7_core.architecture_manifest(source_arm))
    value.update(
        experiment="2D8",
        arm=arm,
        condition=CONDITIONS[arm],
        description="trained-overlap-width-N-O1-O2",
        recurrent_minimum_lags={
            f"B{index + 1}": minimum
            for index, minimum in GEOMETRIES[arm].items()
        },
    )
    for block in value["blocks"]:
        index = int(block["block"]) - 1
        if index in GEOMETRIES[arm]:
            block["recurrent_lags"][0] = GEOMETRIES[arm][index]
            block["recurrent_lags"][1] = 1023
    return value


def architecture_fingerprint(arm: str) -> str:
    return hashlib.sha256(
        json.dumps(
            architecture_manifest(arm), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()


class OverlapWidthGPT(d7_core.BoundaryAlignmentGPT):
    """Post-2D6 model with a configured N/O1/O2 eligibility boundary."""

    def __init__(self, base, arm: str):
        self.geometry_arm = str(arm).upper()
        if self.geometry_arm not in GEOMETRIES:
            raise ValueError(f"unknown 2D8 arm: {arm}")
        self.recurrent_minimum_lags = dict(GEOMETRIES[self.geometry_arm])
        self._last_b6_local_capacity = None
        d6_core.B6NativeNoB7RecurrenceGPT.__init__(self, base)

    def architecture_fingerprint(self):
        return architecture_fingerprint(self.geometry_arm)


# The 2D7 training engine expects this constructor name.
BoundaryAlignmentGPT = OverlapWidthGPT
GEOMETRY_NAMES = CONDITIONS


__all__ = [
    "BoundaryAlignmentGPT",
    "CONDITIONS",
    "GEOMETRIES",
    "GEOMETRY_NAMES",
    "LOCAL_WINDOWS",
    "OverlapWidthGPT",
    "PARAMETER_COUNT",
    "architecture_fingerprint",
    "architecture_manifest",
]
