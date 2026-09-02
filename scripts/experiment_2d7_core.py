"""Experiment 2D7 trained boundary-alignment geometries.

N/O/G are parameter-identical subclasses of the sealed 2D6 B6-native model.
Only the minimum eligible recurrent lag at B1, B3 and B5 changes.
"""

from __future__ import annotations

import hashlib
import json

import torch

import experiment_2d3a_core as fixed_core
import experiment_2d6_core as d6_core
from experiment_2d2b_core import RECURRENT_MAX_LAG


GEOMETRIES = {
    "N": {0: 2, 2: 32, 4: 64},
    "O": {0: 1, 2: 31, 4: 63},
    "G": {0: 3, 2: 33, 4: 65},
}
GEOMETRY_NAMES = {
    "N": "BASELINE_REAL",
    "O": "OVERLAP1_REAL",
    "G": "GAP1_REAL",
}
LOCAL_WINDOWS = dict(d6_core.LOCAL_WINDOWS)
PARAMETER_COUNT = d6_core.EXPECTED_PARAMETER_COUNT


def architecture_manifest(arm: str) -> dict:
    arm = str(arm).upper()
    if arm not in GEOMETRIES:
        raise ValueError(f"unknown 2D7 arm: {arm}")
    minimum = GEOMETRIES[arm]
    blocks = []
    sources = {0: 12, 2: 10, 4: 8}
    for block_index in range(12):
        if block_index in minimum:
            blocks.append({
                "block": block_index + 1,
                "local_lags": [0, LOCAL_WINDOWS[block_index] - 1],
                "recurrent_source_block": sources[block_index],
                "recurrent_lags": [minimum[block_index], RECURRENT_MAX_LAG],
                "separate_local_recurrent_softmax": True,
            })
        else:
            blocks.append({
                "block": block_index + 1,
                "local_lags": [0, 1023],
                "recurrent_source_block": None,
                "recurrent_lags": None,
            })
    return {
        "experiment": "2D7",
        "arm": arm,
        "condition": GEOMETRY_NAMES[arm],
        "description": "trained-boundary-alignment-N-O-G",
        "blocks": blocks,
        "recurrent_minimum_lags": {
            f"B{index + 1}": value for index, value in minimum.items()
        },
        "local_windows": {
            "B1": 2, "B2": 1024, "B3": 32, "B4": 1024,
            "B5": 64, "B6": 1024, "B7": 1024, "B8": 1024,
            "B9": 1024, "B10": 1024, "B11": 1024, "B12": 1024,
        },
        "active_recurrent_gates": ["g_rec", "g_rec_b3", "g_rec_b5"],
        "dormant_compatibility_parameter": "g_rec_b6",
        "b7_to_b6_computational_link": False,
        "b7_recurrent_ring": False,
        "parameter_count": PARAMETER_COUNT,
        "source_identity": "j=t-lag",
        "new_parameters": 0,
    }


def architecture_fingerprint(arm: str) -> str:
    return hashlib.sha256(json.dumps(
        architecture_manifest(arm), sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()


class BoundaryAlignmentGPT(d6_core.B6NativeNoB7RecurrenceGPT):
    """2D6 architecture with a configured recurrent eligibility boundary."""

    def __init__(self, base, arm: str):
        self.geometry_arm = str(arm).upper()
        if self.geometry_arm not in GEOMETRIES:
            raise ValueError(f"unknown 2D7 arm: {arm}")
        self.recurrent_minimum_lags = dict(GEOMETRIES[self.geometry_arm])
        self._last_b6_local_capacity = None
        super().__init__(base)

    def architecture_fingerprint(self):
        return architecture_fingerprint(self.geometry_arm)

    def recurrent_mask(
        self,
        block_index,
        query_length,
        source_length,
        device,
        bank_mode="full",
        query_offset=0,
        source_offset=0,
    ):
        self._validate_bank_mode(bank_mode)
        if block_index not in self.recurrent_minimum_lags:
            raise ValueError(f"B{block_index + 1} has no recurrent mask")
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        minimum = self.recurrent_minimum_lags[block_index]
        if bank_mode in {"full", "old_only"}:
            return (lag >= minimum) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= minimum) & (lag < minimum + 2)
        if bank_mode == "recent_only" and minimum == 2:
            return (lag >= 2) & (lag <= 31)
        return torch.zeros_like(lag, dtype=torch.bool)

    def _incremental_ordinary_block(self, residual, block_index, cache, capacity):
        if block_index == 5:
            self._last_b6_local_capacity = int(capacity)
        return super()._incremental_ordinary_block(
            residual, block_index, cache, capacity
        )

    def incremental_step(
        self,
        token,
        state,
        control="all_real",
        recurrent_permutation=None,
        return_diagnostics=False,
        bank_mode="full",
        diagnostic_attention_weights=True,
        return_block_states=False,
    ):
        self._validate_incremental_state(state)
        mode = self._validate_bank_mode(bank_mode)
        if control != "all_real" or recurrent_permutation is not None:
            raise ValueError("2D7 accepts only unpermuted all_real execution")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")

        position = torch.tensor([state.position], device=token.device, dtype=torch.long)
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(position)
        rings = {
            0: (state.h12_ring, state.h12_positions),
            2: (state.h10_ring, state.h10_positions),
            4: (state.h8_ring, state.h8_positions),
        }
        updated_caches = []
        diagnostics = {}
        captures = {}
        all_block_states = {} if return_block_states else None
        upper_capacity = int(self.config.block_size) - 1
        for block_index in range(fixed_core.TOTAL_LAYERS):
            if block_index in d6_core.ACTIVE_SPECIAL_BLOCKS:
                ring, positions = rings[block_index]
                bank = self._incremental_bank_from_ring(
                    ring,
                    positions,
                    state.position,
                    self.recurrent_minimum_lags[block_index],
                    mode,
                )
                residual, cache, diagnostic = self._incremental_special_block(
                    residual,
                    block_index,
                    state.caches[block_index],
                    bank,
                    None,
                    None,
                    LOCAL_WINDOWS[block_index] - 1,
                    return_diagnostics,
                    diagnostic_attention_weights,
                )
                diagnostics[block_index] = diagnostic
            else:
                residual, cache = self._incremental_ordinary_block(
                    residual, block_index, state.caches[block_index], upper_capacity
                )
            updated_caches.append(cache)
            if block_index in d6_core.ACTIVE_SOURCE_BLOCKS:
                captures[block_index] = residual
            if all_block_states is not None:
                all_block_states[block_index] = residual

        h8, h10, h12 = captures[7], captures[9], captures[11]
        logits = self.base.lm_head(self.base.transformer.ln_f(h12))
        next_h8, next_h8_positions = self._append_ring(
            state.h8_ring, state.h8_positions, h8, state.position
        )
        next_h10, next_h10_positions = self._append_ring(
            state.h10_ring, state.h10_positions, h10, state.position
        )
        next_h12, next_h12_positions = self._append_ring(
            state.h12_ring, state.h12_positions, h12, state.position
        )
        next_state = d6_core.B6NativeIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h8_ring=next_h8,
            h8_positions=next_h8_positions,
            h10_ring=next_h10,
            h10_positions=next_h10_positions,
            h12_ring=next_h12,
            h12_positions=next_h12_positions,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics and not return_block_states:
            return logits, next_state
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "geometry_arm": self.geometry_arm,
            "links": {f"b{index + 1}": row for index, row in diagnostics.items()},
            "cache_audit": self.incremental_cache_audit(next_state),
            "block_states": all_block_states,
            "b7_to_b6_link_executed": False,
        }


__all__ = [
    "BoundaryAlignmentGPT",
    "GEOMETRIES",
    "GEOMETRY_NAMES",
    "LOCAL_WINDOWS",
    "PARAMETER_COUNT",
    "architecture_fingerprint",
    "architecture_manifest",
]
