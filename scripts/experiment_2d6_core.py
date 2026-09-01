"""Experiment 2D6 B6-native W1024 kernel.

The active architecture is parameter-identical to the accepted 2D3A Fixed
model.  B1/B3/B5 retain their mature recurrent paths.  B6 executes as one
ordinary causal W1024 attention block, while ``g_rec_b6`` remains registered
only as dormant checkpoint/optimizer compatibility state.  Incremental state
deliberately has no H7 ring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Tuple

import torch

import experiment_2d3a_core as fixed_core
from experiment_2d2a_core import LayerKVCache
from experiment_2d2b_core import RECURRENT_RING_CAPACITY


EXPERIMENT = "2D6"
EXPECTED_PARAMETER_COUNT = 124_475_908
SOURCE_CHECKPOINT_SHA256 = (
    "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
)
FIXED_CONTROL_CHECKPOINT_SHA256 = (
    "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
)
ACTIVE_SPECIAL_BLOCKS = (0, 2, 4)
ACTIVE_SOURCE_BLOCKS = (7, 9, 11)
LOCAL_WINDOWS = {0: 2, 2: 32, 4: 64}
MIN_LAGS = dict(LOCAL_WINDOWS)
BLOCK_GEOMETRY = (
    (1, 0, 2, 12, 2, 1023),
    (2, 1, 1024, None, None, None),
    (3, 2, 32, 10, 32, 1023),
    (4, 3, 1024, None, None, None),
    (5, 4, 64, 8, 64, 1023),
    (6, 5, 1024, None, None, None),
    (7, 6, 1024, None, None, None),
    (8, 7, 1024, None, None, None),
    (9, 8, 1024, None, None, None),
    (10, 9, 1024, None, None, None),
    (11, 10, 1024, None, None, None),
    (12, 11, 1024, None, None, None),
)
ARCHITECTURE_MANIFEST = {
    "experiment": EXPERIMENT,
    "description": "b6-w1024-no-b7-recurrence",
    "block_geometry": [list(row) for row in BLOCK_GEOMETRY],
    "active_writers": {"B1": "B12", "B3": "B10", "B5": "B8"},
    "inactive_compatibility_state": {
        "parameter": "g_rec_b6",
        "state_dict_key_preserved": True,
        "optimizer_entry_preserved": True,
        "forward_use": False,
        "gradient_required": None,
        "optimizer_update": False,
        "weight_decay": False,
    },
    "b7_to_b6_computational_link": False,
    "b7_recurrent_ring": False,
    "parameter_count": EXPECTED_PARAMETER_COUNT,
    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
}
ARCHITECTURE_FINGERPRINT = hashlib.sha256(
    json.dumps(
        ARCHITECTURE_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class B6NativeIncrementalState:
    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h8_ring: torch.Tensor
    h8_positions: Tuple[int, ...]
    h10_ring: torch.Tensor
    h10_positions: Tuple[int, ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]


class B6NativeNoB7RecurrenceGPT(
    fixed_core.AlternatingIntegrationRecurrentPyramidGPT
):
    """Fixed writer model with B6 promoted to native W1024 and no H7 ring."""

    architecture_fingerprint_sha256 = ARCHITECTURE_FINGERPRINT

    def __init__(self, base):
        super().__init__(base)
        self._active_special_branch_calls = {0: 0, 2: 0, 4: 0}
        self._b6_recurrent_branch_calls = 0

    @classmethod
    def architecture_fingerprint(cls):
        return cls.architecture_fingerprint_sha256

    def _parallel_special_block(self, residual, block_index, *args, **kwargs):
        if block_index == 5:
            self._b6_recurrent_branch_calls += 1
            raise RuntimeError("2D6 forbids the B7-to-B6 recurrent branch")
        if block_index not in ACTIVE_SPECIAL_BLOCKS:
            raise RuntimeError("2D6 encountered an unregistered recurrent destination")
        self._active_special_branch_calls[block_index] += 1
        return super()._parallel_special_block(residual, block_index, *args, **kwargs)

    def forward_pass(
        self,
        tokens,
        targets=None,
        b1_recurrent_source=None,
        b3_recurrent_source=None,
        b5_recurrent_source=None,
        b6_recurrent_source=None,
        b1_recurrent_permutation=None,
        b3_recurrent_permutation=None,
        b5_recurrent_permutation=None,
        b6_recurrent_permutation=None,
        b1_gate_override=None,
        b3_gate_override=None,
        b5_gate_override=None,
        b6_gate_override=None,
        full_counterfactual_blocks=(),
        activation_checkpointing=False,
        return_diagnostics=False,
        capture_all_block_states=False,
        bank_mode="full",
    ):
        if b6_recurrent_source is not None or b6_recurrent_permutation is not None:
            raise ValueError("2D6 forbids B7-to-B6 recurrent input")
        if b6_gate_override is not None:
            raise ValueError("2D6 dormant B6 gate cannot be overridden")
        requested = tuple(int(value) for value in full_counterfactual_blocks)
        if requested not in ((), (5,)):
            raise ValueError("2D6 permits only its mandatory B6-native path")
        return super().forward_pass(
            tokens,
            targets=targets,
            b1_recurrent_source=b1_recurrent_source,
            b3_recurrent_source=b3_recurrent_source,
            b5_recurrent_source=b5_recurrent_source,
            b6_recurrent_source=None,
            b1_recurrent_permutation=b1_recurrent_permutation,
            b3_recurrent_permutation=b3_recurrent_permutation,
            b5_recurrent_permutation=b5_recurrent_permutation,
            b6_recurrent_permutation=None,
            b1_gate_override=b1_gate_override,
            b3_gate_override=b3_gate_override,
            b5_gate_override=b5_gate_override,
            b6_gate_override=None,
            full_counterfactual_blocks=(5,),
            activation_checkpointing=activation_checkpointing,
            return_diagnostics=return_diagnostics,
            capture_all_block_states=capture_all_block_states,
            bank_mode=bank_mode,
        )

    def forward_multi_pass(
        self,
        tokens,
        targets=None,
        num_passes=2,
        b1_recurrent_permutation=None,
        b3_recurrent_permutation=None,
        b5_recurrent_permutation=None,
        b1_gate_override=None,
        b3_gate_override=None,
        b5_gate_override=None,
        activation_checkpointing=False,
        return_diagnostics=False,
        bank_mode="full",
        **forbidden,
    ):
        if forbidden:
            raise ValueError(f"2D6 forbidden multi-pass options: {sorted(forbidden)}")
        if int(num_passes) not in (2, 3):
            raise ValueError("2D6 permits exactly two or three passes")
        results = [
            self.forward_pass(
                tokens,
                targets=targets,
                activation_checkpointing=activation_checkpointing,
                return_diagnostics=return_diagnostics,
                bank_mode=bank_mode,
            )
        ]
        for _ in range(1, int(num_passes)):
            previous = results[-1]
            results.append(
                self.forward_pass(
                    tokens,
                    targets=targets,
                    b1_recurrent_source=previous["h12"],
                    b3_recurrent_source=previous["h10"],
                    b5_recurrent_source=previous["h8"],
                    b1_recurrent_permutation=b1_recurrent_permutation,
                    b3_recurrent_permutation=b3_recurrent_permutation,
                    b5_recurrent_permutation=b5_recurrent_permutation,
                    b1_gate_override=b1_gate_override,
                    b3_gate_override=b3_gate_override,
                    b5_gate_override=b5_gate_override,
                    activation_checkpointing=activation_checkpointing,
                    return_diagnostics=return_diagnostics,
                    bank_mode=bank_mode,
                )
            )
        weights = (0.25, 0.75) if len(results) == 2 else (0.20, 0.40, 0.40)
        weighted = None
        if targets is not None:
            weighted = sum(
                weight * row["loss"] for weight, row in zip(weights, results)
            )
        final = results[-1]
        return {
            "passes": tuple(results),
            "pass_weights": weights,
            "pass_losses": tuple(row["loss"] for row in results),
            "loss": weighted,
            **{
                name: final[name]
                for name in ("h7", "h8", "h10", "h12", "top", "logits")
            },
            "diagnostics": tuple(row["diagnostics"] for row in results),
        }

    def init_incremental_state(self, batch_size, device=None, dtype=None, **kwargs):
        if kwargs:
            raise ValueError(f"2D6 incremental state forbids options: {sorted(kwargs)}")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = (
            torch.bfloat16
            if dtype is None and device.type == "cuda"
            else reference.dtype if dtype is None else dtype
        )
        empty = torch.empty(
            (batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype
        )
        return B6NativeIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * fixed_core.TOTAL_LAYERS,
            h8_ring=empty,
            h8_positions=(),
            h10_ring=empty.clone(),
            h10_positions=(),
            h12_ring=empty.clone(),
            h12_positions=(),
        )

    @staticmethod
    def incremental_cache_lengths(state):
        if not isinstance(state, B6NativeIncrementalState):
            raise TypeError("state must be B6NativeIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

    def _expected_cache_lengths(self, state):
        upper = int(self.config.block_size) - 1
        return tuple(
            min(
                state.position,
                LOCAL_WINDOWS[block_index] - 1
                if block_index in LOCAL_WINDOWS
                else upper,
            )
            for block_index in range(fixed_core.TOTAL_LAYERS)
        )

    def _validate_incremental_state(self, state):
        if not isinstance(state, B6NativeIncrementalState):
            raise TypeError("2D6 incremental_step requires B6NativeIncrementalState")
        if not 0 <= state.position <= int(self.config.block_size):
            raise ValueError("incremental position outside model context")
        if len(state.caches) != fixed_core.TOTAL_LAYERS:
            raise ValueError("incremental cache count mismatch")
        lengths = self.incremental_cache_lengths(state)
        expected = self._expected_cache_lengths(state)
        if lengths != expected:
            raise ValueError(f"incremental cache lengths {lengths} != {expected}")
        positions = tuple(
            range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position)
        )
        for name in ("h8", "h10", "h12"):
            ring = getattr(state, f"{name}_ring")
            current_positions = getattr(state, f"{name}_positions")
            if current_positions != positions:
                raise ValueError(f"incremental {name} positions mismatch")
            if tuple(ring.shape) != (
                state.batch_size,
                len(positions),
                int(self.config.n_embd),
            ):
                raise ValueError(f"incremental {name} ring shape mismatch")
        if hasattr(state, "h7_ring") or hasattr(state, "h7_positions"):
            raise ValueError("2D6 state must not contain an H7 ring")
        for block_index, cache in enumerate(state.caches):
            if cache is None:
                continue
            if cache.key.ndim != 4 or cache.key.shape != cache.value.shape:
                raise ValueError(f"B{block_index + 1} cache shape mismatch")
            if cache.key.size(0) != state.batch_size:
                raise ValueError(f"B{block_index + 1} cache batch mismatch")

    @staticmethod
    def _control_sets(control):
        if control != "all_real":
            raise ValueError("2D6 new architecture exposes only all_real")
        return set(), set()

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
            raise ValueError("2D6 new architecture accepts only unpermuted all_real")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")

        position = torch.tensor(
            [state.position], device=token.device, dtype=torch.long
        )
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(
            position
        )
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
            if block_index in ACTIVE_SPECIAL_BLOCKS:
                ring, positions = rings[block_index]
                bank = self._incremental_bank_from_ring(
                    ring,
                    positions,
                    state.position,
                    MIN_LAGS[block_index],
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
                    residual,
                    block_index,
                    state.caches[block_index],
                    upper_capacity,
                )
            updated_caches.append(cache)
            if block_index in ACTIVE_SOURCE_BLOCKS:
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
        next_state = B6NativeIncrementalState(
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
            "links": {
                f"b{index + 1}": row for index, row in diagnostics.items()
            },
            "cache_audit": self.incremental_cache_audit(next_state),
            "block_states": all_block_states,
            "b7_to_b6_link_executed": False,
        }

    @staticmethod
    def _storage(tensor):
        logical = tensor.numel() * tensor.element_size()
        physical = tensor.untyped_storage().nbytes()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_bytes": logical,
            "physical_bytes": physical,
            "exact": bool(
                tensor.storage_offset() == 0
                and tensor.is_contiguous()
                and logical == physical
            ),
        }

    def incremental_cache_audit(self, state):
        self._validate_incremental_state(state)
        lengths = self.incremental_cache_lengths(state)
        cache_storage = []
        tensors = []
        for index, cache in enumerate(state.caches):
            cache_storage.append(
                {
                    "block": index + 1,
                    "key": None if cache is None else self._storage(cache.key),
                    "value": None if cache is None else self._storage(cache.value),
                }
            )
            if cache is not None:
                tensors.extend((cache.key, cache.value))
        rings = {
            "h8": state.h8_ring,
            "h10": state.h10_ring,
            "h12": state.h12_ring,
        }
        ring_storage = {
            name: self._storage(tensor) for name, tensor in rings.items()
        }
        tensors.extend(rings.values())
        unique = {}
        for tensor in tensors:
            raw = tensor.untyped_storage()
            unique[(str(tensor.device), raw.data_ptr(), raw.nbytes())] = raw.nbytes()
        limits = [1, 1023, 31, 1023, 63, 1023] + [1023] * 6
        physical = (
            all(row["exact"] for row in ring_storage.values())
            and all(
                row["key"] is None
                or (row["key"]["exact"] and row["value"]["exact"])
                for row in cache_storage
            )
            and sum(t.numel() * t.element_size() for t in tensors)
            == sum(unique.values())
        )
        passed = (
            all(value <= limit for value, limit in zip(lengths, limits))
            and all(ring.size(1) <= RECURRENT_RING_CAPACITY for ring in rings.values())
            and physical
            and lengths[5] <= 1023
        )
        return {
            "position": state.position,
            "cache_lengths": list(lengths),
            "cache_limits": limits,
            "historical_local_kv_positions": {
                f"B{index + 1}": length for index, length in enumerate(lengths)
            },
            "b6_historical_local_kv": lengths[5],
            "ring_lengths": {
                name: tensor.size(1) for name, tensor in rings.items()
            },
            "b7_ring_present": False,
            "ring_storage": ring_storage,
            "cache_storage": cache_storage,
            "logical_payload_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in tensors
            ),
            "actual_unique_storage_bytes": sum(unique.values()),
            "physical_storage_exact": bool(physical),
            "passed": bool(passed),
        }

    def incremental_logits(self, tokens, return_diagnostics=False):
        if not torch.is_tensor(tokens) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        state = self.init_incremental_state(
            tokens.size(0), device=tokens.device,
            dtype=torch.bfloat16 if tokens.device.type == "cuda" else None,
        )
        logits = []
        diagnostics = []
        maxima = [0] * fixed_core.TOTAL_LAYERS
        for position in range(tokens.size(1)):
            result = self.incremental_step(
                tokens[:, position], state,
                return_diagnostics=return_diagnostics,
                diagnostic_attention_weights=False,
            )
            if return_diagnostics:
                current, state, row = result
                diagnostics.append(row)
            else:
                current, state = result
            logits.append(current)
            maxima = [
                max(old, new)
                for old, new in zip(maxima, self.incremental_cache_lengths(state))
            ]
        return {
            "logits": torch.cat(logits, dim=1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "cache_audit": self.incremental_cache_audit(state),
        }


Experiment2D6Model = B6NativeNoB7RecurrenceGPT

__all__ = [
    "ACTIVE_SOURCE_BLOCKS",
    "ACTIVE_SPECIAL_BLOCKS",
    "ARCHITECTURE_FINGERPRINT",
    "ARCHITECTURE_MANIFEST",
    "BLOCK_GEOMETRY",
    "B6NativeIncrementalState",
    "B6NativeNoB7RecurrenceGPT",
    "EXPECTED_PARAMETER_COUNT",
    "Experiment2D6Model",
    "FIXED_CONTROL_CHECKPOINT_SHA256",
    "LOCAL_WINDOWS",
    "MIN_LAGS",
    "RECURRENT_RING_CAPACITY",
    "SOURCE_CHECKPOINT_SHA256",
]
