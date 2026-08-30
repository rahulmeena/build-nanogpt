"""Experiment 2D5C fixed-writer B3/B5 representation-pressure kernels.

The scientific C model is parameter-identical to the accepted 2D3A fixed
writer.  It changes only attention geometry: B3 and B5 retain one native
historical K/V position (W2) and read their existing B10 and B8 recurrent
writer rings at lags 2--1023.  B1 and B6 remain unchanged.

``FixedControlEvaluationGPT`` keeps the accepted W32/W64 geometry but adds the
same tightly scoped evaluation controls.  It exists so the sealed Fixed-100M
checkpoint can be evaluated without changing ``experiment_2d3a_core``.
Neither class registers a parameter, buffer, projection, router, or module.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional

import torch
from torch.nn import functional as F

import experiment_2d3a_core as fixed_core
from experiment_2d2b_core import (
    FullB12ToB1RecurrentKVGPT,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
)


class _ImmutableJSONDict(dict):
    """A JSON-serializable dict whose public mutation operations are blocked."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("2D5C scientific constants are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


EXPERIMENT = "2D5C"
EXPECTED_PARAMETER_COUNT = 124_475_908
C_SOURCE_CHECKPOINT_SHA256 = (
    "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
)
FIXED_CONTROL_CHECKPOINT_SHA256 = (
    "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
)
TOTAL_LAYERS = fixed_core.TOTAL_LAYERS
SPECIAL_BLOCKS = fixed_core.SPECIAL_BLOCKS
SOURCE_BLOCKS = fixed_core.SOURCE_BLOCKS
GATE_NAMES = MappingProxyType(dict(fixed_core.GATE_NAMES))

# Human-facing artifacts use B1--B12; runtime code uses zero-based indices.
FIXED_WRITER_SOURCES = MappingProxyType({0: 11, 2: 9, 4: 7, 5: 6})
# This human-facing mapping remains directly JSON-serializable for audits.
FIXED_WRITERS = _ImmutableJSONDict(
    {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"}
)

FIXED_CONTROL_LOCAL_WINDOWS = MappingProxyType({0: 2, 2: 32, 4: 64, 5: 512})
FIXED_CONTROL_MIN_LAGS = FIXED_CONTROL_LOCAL_WINDOWS
C_LOCAL_WINDOWS = MappingProxyType({0: 2, 2: 2, 4: 2, 5: 512})
C_MIN_LAGS = C_LOCAL_WINDOWS
LOCAL_WINDOWS_C = C_LOCAL_WINDOWS
MIN_LAGS_C = C_MIN_LAGS
LOCAL_WINDOWS_FIXED = FIXED_CONTROL_LOCAL_WINDOWS
MIN_LAGS_FIXED = FIXED_CONTROL_MIN_LAGS

# The experiment-facing aliases intentionally describe C, not the control.
LOCAL_WINDOWS = C_LOCAL_WINDOWS
MIN_LAGS = C_MIN_LAGS
MAX_RECURRENT_ENTRIES = MappingProxyType({
    block: RECURRENT_MAX_LAG - lag + 1 for block, lag in C_MIN_LAGS.items()
})

INCREMENTAL_CONTROLS = (
    "all_real",
    "b3_off",
    "b3_shuffled",
    "b5_off",
    "b5_shuffled",
    "b3_b5_off",
    "b3_b5_shuffled",
)

# Tuples only: directly JSON-serializable and recursively immutable.
FIXED_CONTROL_BLOCK_GEOMETRY = (
    (1, 0, 2, 12, 2, 1023),
    (2, 1, 1024, None, None, None),
    (3, 2, 32, 10, 32, 1023),
    (4, 3, 1024, None, None, None),
    (5, 4, 64, 8, 64, 1023),
    (6, 5, 512, 7, 512, 1023),
    (7, 6, 1024, None, None, None),
    (8, 7, 1024, None, None, None),
    (9, 8, 1024, None, None, None),
    (10, 9, 1024, None, None, None),
    (11, 10, 1024, None, None, None),
    (12, 11, 1024, None, None, None),
)
C_BLOCK_GEOMETRY = (
    (1, 0, 2, 12, 2, 1023),
    (2, 1, 1024, None, None, None),
    (3, 2, 2, 10, 2, 1023),
    (4, 3, 1024, None, None, None),
    (5, 4, 2, 8, 2, 1023),
    (6, 5, 512, 7, 512, 1023),
    (7, 6, 1024, None, None, None),
    (8, 7, 1024, None, None, None),
    (9, 8, 1024, None, None, None),
    (10, 9, 1024, None, None, None),
    (11, 10, 1024, None, None, None),
    (12, 11, 1024, None, None, None),
)

FIXED_CONTROL_ARCHITECTURE_FINGERPRINT_INPUTS = (
    ("experiment", EXPERIMENT),
    ("model_weight_lineage", "accepted-2d3a-fixed-writer"),
    ("checkpoint_sha256", FIXED_CONTROL_CHECKPOINT_SHA256),
    ("geometry", "fixed-b3-w32-b5-w64"),
    ("block_geometry", FIXED_CONTROL_BLOCK_GEOMETRY),
    ("controls", INCREMENTAL_CONTROLS),
    ("recurrent_ring_capacity", RECURRENT_RING_CAPACITY),
    ("expected_parameter_count", EXPECTED_PARAMETER_COUNT),
)
ARCHITECTURE_FINGERPRINT_INPUTS = (
    ("experiment", EXPERIMENT),
    ("model_weight_lineage", "accepted-2d3a-fixed-writer"),
    ("checkpoint_sha256", C_SOURCE_CHECKPOINT_SHA256),
    ("geometry", "b3-w2-b5-w2-representation-pressure"),
    ("block_geometry", C_BLOCK_GEOMETRY),
    ("controls", INCREMENTAL_CONTROLS),
    ("recurrent_ring_capacity", RECURRENT_RING_CAPACITY),
    ("expected_parameter_count", EXPECTED_PARAMETER_COUNT),
)

# These hashes are the canonical, sorted-JSON fingerprints of the complete
# C/Fixed manifests used by the 2D5C driver.  Keeping them here makes a wrong
# architecture fail before either training or evaluation begins.
ARCHITECTURE_FINGERPRINT_C = (
    "019d822dd89986c269e985fba8d1277a15d476dd73a0dac0d8c35e07e7315c12"
)
ARCHITECTURE_FINGERPRINT_FIXED = (
    "be345a9fe3b486f601c3af1564ce90f51de51c84daf6e89885126b094adfaac2"
)


class _B3B5ControlledFixedWriterGPT(
    fixed_core.AlternatingIntegrationRecurrentPyramidGPT
):
    """Shared geometry-safe implementation for C and its Fixed control."""

    local_windows: Mapping[int, int]
    minimum_lags: Mapping[int, int]
    architecture_fingerprint_inputs: tuple
    architecture_fingerprint_sha256: str

    @classmethod
    def architecture_fingerprint(cls) -> str:
        return cls.architecture_fingerprint_sha256

    def local_mask(self, block_index: int, length: int, device) -> torch.Tensor:
        if block_index not in self.local_windows:
            raise ValueError(f"B{block_index + 1} has no special local mask")
        key = (block_index, int(length), str(device))
        mask = self._local_mask_cache.get(key)
        if mask is None or mask.device != torch.device(device):
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            window = self.local_windows[block_index]
            mask = (source <= query) & (source >= query - (window - 1))
            self._local_mask_cache[key] = mask
        return mask

    @classmethod
    def recurrent_mask(
        cls,
        block_index: int,
        query_length: int,
        source_length: int,
        device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        if block_index not in cls.minimum_lags:
            raise ValueError(f"B{block_index + 1} has no recurrent mask")
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        minimum = cls.minimum_lags[block_index]
        if bank_mode in {"full", "old_only"}:
            return (lag >= minimum) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= minimum) & (lag < minimum + 2)
        if bank_mode == "recent_only" and minimum == 2:
            return (lag >= 2) & (lag <= 31)
        return torch.zeros_like(lag, dtype=torch.bool)

    def forward_pass(self, *args, full_counterfactual_blocks=(), **kwargs):
        if tuple(full_counterfactual_blocks):
            raise ValueError("2D5C kernels prohibit full-window counterfactual blocks")
        return super().forward_pass(
            *args, full_counterfactual_blocks=(), **kwargs
        )

    def init_incremental_state(
        self, batch_size, device=None, dtype=None, b6_full_native=False
    ):
        if b6_full_native:
            raise ValueError("2D5C kernels prohibit the B6 full-native counterfactual")
        return super().init_incremental_state(
            batch_size, device=device, dtype=dtype, b6_full_native=False
        )

    def _expected_cache_lengths(self, state):
        result = []
        upper_capacity = int(self.config.block_size) - 1
        for block_index in range(TOTAL_LAYERS):
            capacity = (
                self.local_windows[block_index] - 1
                if block_index in self.local_windows
                else upper_capacity
            )
            result.append(min(state.position, capacity))
        return tuple(result)

    def _incremental_bank_from_ring(
        self, ring, positions, state_position, minimum_lag, mode
    ):
        """Select a recurrent bank only from an exact causal ring index map.

        The inherited implementation is deliberately compact and relies on the
        incremental-state validator to establish that logical positions are a
        contiguous mapping onto physical ring slots.  2D5C audits this helper
        directly during preflight, so make those assumptions explicit here as
        fail-closed runtime invariants as well.  In particular, a current or
        future logical position must never be silently interpreted as an older
        physical slot after rollover.
        """
        logical_positions = tuple(int(value) for value in positions)
        if ring.ndim != 3:
            raise ValueError("2D5C recurrent ring must have shape [batch,time,channels]")
        if ring.size(1) != len(logical_positions):
            raise ValueError("2D5C recurrent ring and logical positions differ in length")
        if len(logical_positions) > RECURRENT_RING_CAPACITY:
            raise ValueError("2D5C recurrent ring exceeds its physical capacity")
        if logical_positions:
            expected = tuple(
                range(logical_positions[0], logical_positions[0] + len(logical_positions))
            )
            if logical_positions != expected:
                raise ValueError("2D5C recurrent logical positions must be contiguous")
            if logical_positions[-1] >= int(state_position):
                raise ValueError("2D5C recurrent ring contains a current or future position")
        bank = super()._incremental_bank_from_ring(
            ring, logical_positions, state_position, minimum_lag, mode
        )
        if bank.positions.numel():
            lags = int(state_position) - bank.positions
            if bool((lags <= 0).any()):
                raise RuntimeError("2D5C recurrent selection aliased current/future state")
        return bank

    @staticmethod
    def _control_sets(control):
        off = {
            "b3_off": {2},
            "b5_off": {4},
            "b3_b5_off": {2, 4},
        }.get(control, set())
        shuffled = {
            "b3_shuffled": {2},
            "b5_shuffled": {4},
            "b3_b5_shuffled": {2, 4},
        }.get(control, set())
        return off, shuffled

    @classmethod
    def control_sets(cls, control):
        """Return fresh target sets for one exact preregistered control."""
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown 2D5C incremental control: {control}")
        return cls._control_sets(control)

    @staticmethod
    def _diagnostic_tensor(tensor, retain_grad):
        if tensor is None:
            return None
        if not retain_grad:
            return tensor.detach()
        if tensor.requires_grad:
            tensor.retain_grad()
        return tensor

    def _incremental_special_block(
        self,
        residual,
        block_index,
        cache,
        recurrent_bank,
        permutation,
        gate_override,
        local_capacity,
        return_diagnostics,
        diagnostic_attention_weights,
        diagnostic_retain_grad=False,
        query_position=None,
    ):
        """Accepted incremental kernel plus optional, non-invasive diagnostics."""
        block = self.base.transformer.h[block_index]
        batch = residual.size(0)
        channels = int(self.config.n_embd)
        heads = int(self.config.n_head)
        head_size = channels // heads
        normalized = block.ln_1(residual)
        query, current_key, current_value = block.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, 1, heads, head_size).transpose(1, 2)
        current_key = current_key.view(batch, 1, heads, head_size).transpose(1, 2)
        current_value = current_value.view(batch, 1, heads, head_size).transpose(1, 2)
        local_keys = (
            current_key if cache is None else torch.cat((cache.key, current_key), dim=2)
        )
        local_values = (
            current_value
            if cache is None
            else torch.cat((cache.value, current_value), dim=2)
        )
        if local_keys.size(2) > local_capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess local KV")
        local_pre = F.scaled_dot_product_attention(
            query, local_keys, local_values, is_causal=False
        )
        next_cache = self._append_cache(
            current_key, current_value, cache, local_capacity
        )

        bank_values = None
        rec_key = rec_value = None
        recurrent_weights = None
        if recurrent_bank is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        elif recurrent_bank.values.size(1) == 0:
            bank_values = recurrent_bank.values
            recurrent_pre = torch.zeros_like(local_pre)
            recurrent_weights = local_pre.new_empty((batch, heads, 1, 0))
            coefficient = self._gate_coefficient(
                block_index, local_pre, gate_override
            )
        else:
            bank_values = recurrent_bank.values
            if permutation is not None:
                bank_values = bank_values[permutation]
            rec_key, rec_value = self.project_recurrent_kv(block_index, bank_values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                rec_key,
                rec_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics and diagnostic_attention_weights,
            )
            coefficient = self._gate_coefficient(
                block_index, local_pre, gate_override
            )
        gated_recurrent_pre = coefficient * recurrent_pre
        combined_pre = local_pre + gated_recurrent_pre
        combined = combined_pre.transpose(1, 2).contiguous().view(
            batch, 1, channels
        )
        attention_projection = block.attn.c_proj(combined)
        residual = residual + attention_projection
        residual = residual + block.mlp(block.ln_2(residual))
        if not return_diagnostics:
            return residual, next_cache, None

        local_valid = torch.ones(
            (1, local_keys.size(2)), device=local_keys.device, dtype=torch.bool
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_keys, local_valid)
            if diagnostic_attention_weights
            else None
        )
        if query_position is None:
            local_positions = None
            query_positions = None
        else:
            first_local = int(query_position) - local_keys.size(2) + 1
            local_positions = torch.arange(
                first_local,
                int(query_position) + 1,
                device=local_keys.device,
                dtype=torch.long,
            ).view(1, -1)
            query_positions = torch.tensor(
                [[int(query_position)]], device=local_keys.device, dtype=torch.long
            )
        local_flat = local_pre.transpose(1, 2).contiguous().view(batch, 1, channels)
        local_post = block.attn.c_proj(local_flat)
        recurrent_post = attention_projection - local_post
        expose = lambda value: self._diagnostic_tensor(
            value, diagnostic_retain_grad
        )
        retained_names = ()
        if diagnostic_retain_grad:
            retained_names = tuple(
                name
                for name, value in (
                    ("query", query),
                    ("recurrent_source_reads", bank_values),
                    ("recurrent_key_reads", rec_key),
                    ("recurrent_value_reads", rec_value),
                    ("local_pre_c_proj", local_pre),
                    ("recurrent_pre_c_proj", recurrent_pre),
                    ("gated_recurrent_pre_c_proj", gated_recurrent_pre),
                    ("combined_pre_c_proj", combined_pre),
                    ("gate_coefficient", coefficient),
                )
                if value is not None and value.requires_grad
            )
        return residual, next_cache, {
            "block": block_index + 1,
            "source_block": {0: 12, 2: 10, 4: 8, 5: 7}[block_index],
            "query_position": None if query_position is None else int(query_position),
            "query_positions": query_positions,
            "local_positions": local_positions,
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
            "local_valid_mask": local_valid,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            # Attention probabilities are diagnostic recomputations; detach
            # them even in gradient mode.  The retained tensors below are the
            # ones on the model's actual SDPA execution path.
            "local_attention_weights": (
                None if local_weights is None else local_weights.detach()
            ),
            "recurrent_attention_weights": (
                None if recurrent_weights is None else recurrent_weights.detach()
            ),
            "query": expose(query),
            "recurrent_source_reads": expose(bank_values),
            "recurrent_key_reads": expose(rec_key),
            "recurrent_value_reads": expose(rec_value),
            "local_pre_c_proj": expose(local_pre),
            "recurrent_pre_c_proj": expose(recurrent_pre),
            "gated_recurrent_pre_c_proj": expose(gated_recurrent_pre),
            "combined_pre_c_proj": expose(combined_pre),
            "local_post_c_proj": expose(local_post),
            "gated_recurrent_post_c_proj": expose(recurrent_post),
            "post_c_proj_isolation": (
                "actual_combined_projection_minus_local_projection"
            ),
            "local_output_rms": expose(
                local_pre.float().square().mean().sqrt()
            ),
            "recurrent_output_rms": expose(
                recurrent_pre.float().square().mean().sqrt()
            ),
            "gated_recurrent_output_rms": expose(
                gated_recurrent_pre.float().square().mean().sqrt()
            ),
            "gate_raw": expose(self.gate_parameter(block_index)),
            "gate_coefficient": expose(coefficient),
            "diagnostic_retain_grad": bool(diagnostic_retain_grad),
            "retained_tensor_names": retained_names,
        }

    @staticmethod
    def _append_attached_ring(ring, positions, value, position):
        """Diagnostic-only ring append that deliberately preserves autograd."""
        updated = torch.cat(
            (ring, value.to(device=ring.device, dtype=ring.dtype)), dim=1
        )
        updated_positions = (*positions, int(position))
        if updated.size(1) > RECURRENT_RING_CAPACITY:
            updated = updated[:, -RECURRENT_RING_CAPACITY:].clone(
                memory_format=torch.contiguous_format
            )
            updated_positions = updated_positions[-RECURRENT_RING_CAPACITY:]
        return updated, tuple(int(item) for item in updated_positions)

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
        diagnostic_retain_grad=False,
    ):
        self._validate_incremental_state(state)
        if diagnostic_retain_grad and not return_diagnostics:
            raise ValueError(
                "diagnostic_retain_grad requires return_diagnostics=True"
            )
        if diagnostic_retain_grad and not torch.is_grad_enabled():
            raise ValueError(
                "diagnostic_retain_grad requires autograd to be enabled"
            )
        if (
            diagnostic_retain_grad
            and state.position > 0
            and not state.h10_ring.requires_grad
        ):
            raise ValueError(
                "diagnostic_retain_grad must be enabled from the first step"
            )
        mode = self._validate_bank_mode(bank_mode)
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown 2D5C incremental control: {control}")
        if state.b6_full_native:
            raise ValueError("2D5C incremental state cannot use full-native B6")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")
        off, shuffled = self._control_sets(control)
        if shuffled:
            permutation = self._validate_permutation(
                recurrent_permutation, state.batch_size, token.device
            )
        elif recurrent_permutation is not None:
            raise ValueError("permutation is valid only for shuffled controls")
        else:
            permutation = None

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
            5: (state.h7_ring, state.h7_positions),
        }
        updated_caches = []
        diagnostics = {}
        captures = {}
        all_block_states = {} if return_block_states else None
        upper_capacity = int(self.config.block_size) - 1
        for block_index in range(TOTAL_LAYERS):
            if block_index in SPECIAL_BLOCKS:
                ring, positions = rings[block_index]
                bank = self._incremental_bank_from_ring(
                    ring,
                    positions,
                    state.position,
                    self.minimum_lags[block_index],
                    mode,
                )
                block_off = block_index in off
                residual, cache, diagnostic = self._incremental_special_block(
                    residual,
                    block_index,
                    state.caches[block_index],
                    None if block_off else bank,
                    permutation if block_index in shuffled else None,
                    0.0 if block_off else None,
                    self.local_windows[block_index] - 1,
                    return_diagnostics,
                    diagnostic_attention_weights,
                    diagnostic_retain_grad,
                    state.position,
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
            if block_index in SOURCE_BLOCKS:
                captures[block_index] = residual
            if all_block_states is not None:
                all_block_states[block_index] = (
                    residual if diagnostic_retain_grad else residual.detach()
                )

        h7, h8, h10, h12 = captures[6], captures[7], captures[9], captures[11]
        logits = self.base.lm_head(self.base.transformer.ln_f(h12))
        append_ring = (
            self._append_attached_ring
            if diagnostic_retain_grad
            else self._append_ring
        )
        # In diagnostic autograd mode, clone the two preregistered writer
        # representations at the exact ring-write edge.  Their retained
        # gradients then isolate the temporal recurrent path and exclude each
        # writer's ordinary same-token residual path while remaining attached
        # to the actual B8/B10 activations and parameters.
        h8_ring_write = h8.clone() if diagnostic_retain_grad else h8
        h10_ring_write = h10.clone() if diagnostic_retain_grad else h10
        next_h7, next_h7_positions = append_ring(
            state.h7_ring, state.h7_positions, h7, state.position
        )
        next_h8, next_h8_positions = append_ring(
            state.h8_ring, state.h8_positions, h8_ring_write, state.position
        )
        next_h10, next_h10_positions = append_ring(
            state.h10_ring, state.h10_positions, h10_ring_write, state.position
        )
        next_h12, next_h12_positions = append_ring(
            state.h12_ring, state.h12_positions, h12, state.position
        )
        next_state = fixed_core.PyramidIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h7_ring=next_h7,
            h7_positions=next_h7_positions,
            h8_ring=next_h8,
            h8_positions=next_h8_positions,
            h10_ring=next_h10,
            h10_positions=next_h10_positions,
            h12_ring=next_h12,
            h12_positions=next_h12_positions,
            b6_full_native=False,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics and not return_block_states:
            return logits, next_state
        writer_block_states = None
        if diagnostic_retain_grad:
            for value in (h8_ring_write, h10_ring_write):
                if value.requires_grad:
                    value.retain_grad()
            writer_block_states = {
                "b8": h8_ring_write,
                "b10": h10_ring_write,
            }
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "links": {
                f"b{index + 1}": row for index, row in diagnostics.items()
            },
            "cache_audit": self.incremental_cache_audit(next_state),
            "block_states": all_block_states,
            "writer_block_states": writer_block_states,
            "writer_gradient_scope": (
                None if not diagnostic_retain_grad else
                "temporal_recurrent_ring_write_edge_only"
            ),
            "diagnostic_retain_grad": bool(diagnostic_retain_grad),
        }

    def incremental_cache_audit(self, state):
        self._validate_incremental_state(state)
        lengths = self.incremental_cache_lengths(state)

        def storage(tensor):
            expected = tensor.numel() * tensor.element_size()
            actual = tensor.untyped_storage().nbytes()
            exact = (
                tensor.storage_offset() == 0
                and tensor.is_contiguous()
                and expected == actual
            )
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "expected_bytes": expected,
                "actual_bytes": actual,
                "exact": bool(exact),
            }

        cache_storage = []
        tensors = []
        for index, cache in enumerate(state.caches):
            cache_storage.append({
                "block": index + 1,
                "key": None if cache is None else storage(cache.key),
                "value": None if cache is None else storage(cache.value),
            })
            if cache is not None:
                tensors.extend((cache.key, cache.value))
        ring_tensors = {
            "h7": state.h7_ring,
            "h8": state.h8_ring,
            "h10": state.h10_ring,
            "h12": state.h12_ring,
        }
        ring_storage = {name: storage(tensor) for name, tensor in ring_tensors.items()}
        tensors.extend(ring_tensors.values())

        unique_storages = {}
        positive_storage_ids = []
        for tensor in tensors:
            raw = tensor.untyped_storage()
            identity = (str(tensor.device), raw.data_ptr(), raw.nbytes())
            unique_storages.setdefault(identity, raw.nbytes())
            if raw.nbytes():
                positive_storage_ids.append(identity)
        alias_free = len(positive_storage_ids) == len(set(positive_storage_ids))
        logical_payload_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        actual_unique_storage_bytes = sum(unique_storages.values())

        upper_capacity = int(self.config.block_size) - 1
        limits = [
            self.local_windows[index] - 1
            if index in self.local_windows
            else upper_capacity
            for index in range(TOTAL_LAYERS)
        ]
        physical = (
            alias_free
            and all(row["exact"] for row in ring_storage.values())
            and all(
                row["key"] is None
                or (row["key"]["exact"] and row["value"]["exact"])
                for row in cache_storage
            )
            and logical_payload_bytes == actual_unique_storage_bytes
        )
        passed = (
            all(value <= limit for value, limit in zip(lengths, limits))
            and all(
                tensor.size(1) <= RECURRENT_RING_CAPACITY
                for tensor in ring_tensors.values()
            )
            and physical
        )
        return {
            "position": state.position,
            "cache_lengths": list(lengths),
            "cache_limits": limits,
            "historical_local_kv_positions": {
                f"B{index + 1}": lengths[index] for index in range(TOTAL_LAYERS)
            },
            "b1_historical_local_kv": lengths[0],
            "b3_historical_local_kv": lengths[2],
            "b5_historical_local_kv": lengths[4],
            "b6_historical_local_kv": lengths[5],
            "ring_lengths": {
                name: tensor.size(1) for name, tensor in ring_tensors.items()
            },
            "ring_limit": RECURRENT_RING_CAPACITY,
            "no_h9_or_h11_ring": True,
            "cache_storage": cache_storage,
            "ring_storage": ring_storage,
            "logical_payload_bytes": logical_payload_bytes,
            "actual_unique_storage_bytes": actual_unique_storage_bytes,
            "unique_storage_count": len(unique_storages),
            "storage_alias_free": bool(alias_free),
            "physical_storage_exact": bool(physical),
            "b6_full_native": False,
            "passed": bool(passed),
        }


class FixedControlEvaluationGPT(_B3B5ControlledFixedWriterGPT):
    """Accepted Fixed geometry plus the seven preregistered B3/B5 controls."""

    local_windows = FIXED_CONTROL_LOCAL_WINDOWS
    minimum_lags = FIXED_CONTROL_MIN_LAGS
    architecture_fingerprint_inputs = FIXED_CONTROL_ARCHITECTURE_FINGERPRINT_INPUTS
    architecture_fingerprint_sha256 = ARCHITECTURE_FINGERPRINT_FIXED


class FixedWriterB3B5W2RepresentationPressureGPT(
    _B3B5ControlledFixedWriterGPT
):
    """Scientific 2D5C C arm: B3 W2 and B5 W2, fixed writers unchanged."""

    local_windows = C_LOCAL_WINDOWS
    minimum_lags = C_MIN_LAGS
    architecture_fingerprint_inputs = ARCHITECTURE_FINGERPRINT_INPUTS
    architecture_fingerprint_sha256 = ARCHITECTURE_FINGERPRINT_C


Experiment2D5CModel = FixedWriterB3B5W2RepresentationPressureGPT
RecurrentKVGPT = FixedWriterB3B5W2RepresentationPressureGPT
FixedWriterW2PressureGPT = FixedWriterB3B5W2RepresentationPressureGPT


__all__ = [
    "ARCHITECTURE_FINGERPRINT_INPUTS",
    "ARCHITECTURE_FINGERPRINT_C",
    "ARCHITECTURE_FINGERPRINT_FIXED",
    "C_SOURCE_CHECKPOINT_SHA256",
    "C_BLOCK_GEOMETRY",
    "C_LOCAL_WINDOWS",
    "C_MIN_LAGS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPERIMENT",
    "Experiment2D5CModel",
    "FIXED_CONTROL_ARCHITECTURE_FINGERPRINT_INPUTS",
    "FIXED_CONTROL_CHECKPOINT_SHA256",
    "FIXED_CONTROL_BLOCK_GEOMETRY",
    "FIXED_CONTROL_LOCAL_WINDOWS",
    "FIXED_CONTROL_MIN_LAGS",
    "FIXED_WRITER_SOURCES",
    "FIXED_WRITERS",
    "FixedControlEvaluationGPT",
    "FixedWriterB3B5W2RepresentationPressureGPT",
    "FixedWriterW2PressureGPT",
    "GATE_NAMES",
    "INCREMENTAL_CONTROLS",
    "LOCAL_WINDOWS",
    "LOCAL_WINDOWS_C",
    "LOCAL_WINDOWS_FIXED",
    "MAX_RECURRENT_ENTRIES",
    "MIN_LAGS",
    "MIN_LAGS_C",
    "MIN_LAGS_FIXED",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "RecurrentKVGPT",
]
