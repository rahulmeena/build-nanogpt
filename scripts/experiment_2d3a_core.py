"""Experiment 2D3A alternating-integration recurrent-pyramid kernel.

The only parameters added to the SHA-pinned 2D2G-A model are the scalar
gates for B10->B3, B8->B5, and B7->B6.  B12->B1 and its trained gate are
inherited unchanged.  Every recurrent arm reuses the destination LN/Q/K/V
and applies the destination c_proj exactly once after adding separately
softmaxed local and recurrent attention outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiment_2d2a_core import LayerKVCache
from experiment_2d2b_core import (
    ATTENTION_WEIGHT_ELEMENT_LIMIT,
    BANK_MODES,
    FullB12ToB1RecurrentKVGPT,
    FullRecurrentBank,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
)


TOTAL_LAYERS = 12
SPECIAL_BLOCKS = (0, 2, 4, 5)  # B1, B3, B5, B6
SOURCE_BLOCKS = (11, 9, 7, 6)  # B12, B10, B8, B7
DESTINATION_NAMES = ("b1", "b3", "b5", "b6")
SOURCE_NAMES = ("h12", "h10", "h8", "h7")
LOCAL_WINDOWS = {0: 2, 2: 32, 4: 64, 5: 512}
MIN_LAGS = dict(LOCAL_WINDOWS)
MAX_RECURRENT_ENTRIES = {
    block: RECURRENT_MAX_LAG - lag + 1 for block, lag in MIN_LAGS.items()
}
GATE_NAMES = {0: "g_rec", 2: "g_rec_b3", 4: "g_rec_b5", 5: "g_rec_b6"}

INCREMENTAL_CONTROLS = (
    "all_real",
    "new_links_off",
    "b1_off",
    "b3_off",
    "b5_off",
    "b6_off",
    "b1_shuffled",
    "b3_shuffled",
    "b5_shuffled",
    "b6_shuffled",
    "all_new_shuffled",
    "all_recurrent_shuffled",
    "b6_full_native",
)


@dataclass(frozen=True)
class PyramidIncrementalState:
    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h7_ring: torch.Tensor
    h7_positions: Tuple[int, ...]
    h8_ring: torch.Tensor
    h8_positions: Tuple[int, ...]
    h10_ring: torch.Tensor
    h10_positions: Tuple[int, ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]
    b6_full_native: bool = False


class AlternatingIntegrationRecurrentPyramidGPT(FullB12ToB1RecurrentKVGPT):
    """2D2G-A plus exactly three fresh scalar recurrent gates."""

    def __init__(self, base: nn.Module):
        super().__init__(base)
        self.g_rec_b3 = nn.Parameter(torch.zeros(()))
        self.g_rec_b5 = nn.Parameter(torch.zeros(()))
        self.g_rec_b6 = nn.Parameter(torch.zeros(()))
        self._local_mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    @property
    def g_rec_b1(self):
        return self.g_rec

    @property
    def recurrent_scale_b1(self):
        return self.g_rec.tanh()

    @property
    def recurrent_scale_b3(self):
        return self.g_rec_b3.tanh()

    @property
    def recurrent_scale_b5(self):
        return self.g_rec_b5.tanh()

    @property
    def recurrent_scale_b6(self):
        return self.g_rec_b6.tanh()

    def gate_parameter(self, block_index: int) -> torch.Tensor:
        if block_index not in GATE_NAMES:
            raise ValueError(f"B{block_index + 1} is not recurrent")
        return getattr(self, GATE_NAMES[block_index])

    def recurrent_scale(self, block_index: int) -> torch.Tensor:
        return self.gate_parameter(block_index).tanh()

    def local_mask(self, block_index: int, length: int, device) -> torch.Tensor:
        if block_index not in LOCAL_WINDOWS:
            raise ValueError(f"B{block_index + 1} has no special local mask")
        key = (block_index, int(length), str(device))
        mask = self._local_mask_cache.get(key)
        if mask is None or mask.device != torch.device(device):
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            window = LOCAL_WINDOWS[block_index]
            mask = (source <= query) & (source >= query - (window - 1))
            self._local_mask_cache[key] = mask
        return mask

    @staticmethod
    def recurrent_mask(
        block_index: int,
        query_length: int,
        source_length: int,
        device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        if block_index not in MIN_LAGS:
            raise ValueError(f"B{block_index + 1} has no recurrent mask")
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        minimum = MIN_LAGS[block_index]
        if bank_mode in {"full", "old_only"}:
            return (lag >= minimum) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= minimum) & (lag < minimum + 2)
        if bank_mode == "recent_only" and minimum == 2:
            return (lag >= 2) & (lag <= 31)
        return torch.zeros_like(lag, dtype=torch.bool)

    def build_recurrent_bank(
        self,
        block_index: int,
        recurrent_source: torch.Tensor,
        bank_mode: Optional[str] = None,
    ) -> FullRecurrentBank:
        if not torch.is_tensor(recurrent_source) or recurrent_source.ndim != 3:
            raise ValueError("recurrent source must have shape [batch,time,channel]")
        if recurrent_source.size(-1) != int(self.config.n_embd):
            raise ValueError("recurrent source channel width mismatch")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        length = recurrent_source.size(1)
        valid = self.recurrent_mask(
            block_index, length, length, recurrent_source.device, mode
        )
        positions = torch.arange(length, device=recurrent_source.device).view(1, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=positions.expand(length, length),
        )

    def project_recurrent_kv(self, block_index: int, values: torch.Tensor):
        if not torch.is_tensor(values) or values.ndim != 3:
            raise ValueError("recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if values.size(-1) != channels:
            raise ValueError("recurrent values channel width mismatch")
        block = self.base.transformer.h[block_index]
        normalized = block.ln_1(values)
        _, key, value = block.attn.c_attn(normalized).split(channels, dim=-1)
        batch, length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, length, heads, head_size).transpose(1, 2)
        value = value.view(batch, length, heads, head_size).transpose(1, 2)
        return key, value

    @staticmethod
    def _diagnostic_attention_weights(query, key, valid_mask):
        elements = query.size(0) * query.size(1) * query.size(2) * key.size(2)
        if elements > ATTENTION_WEIGHT_ELEMENT_LIMIT:
            return None
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores.mul_(query.size(-1) ** -0.5)
        mask = valid_mask.view(1, 1, query.size(2), key.size(2))
        row_valid = mask.any(dim=-1, keepdim=True)
        safe = torch.where(
            row_valid,
            scores.masked_fill(~mask, -torch.inf),
            torch.zeros_like(scores),
        )
        return F.softmax(safe, dim=-1) * mask.to(safe.dtype)

    def _gate_coefficient(self, block_index, reference, gate_override):
        if gate_override is None:
            coefficient = self.recurrent_scale(block_index)
        elif torch.is_tensor(gate_override):
            if gate_override.numel() != 1:
                raise ValueError("gate override must be scalar")
            coefficient = gate_override.reshape(())
        else:
            coefficient = reference.new_tensor(float(gate_override))
        return coefficient.to(device=reference.device, dtype=reference.dtype)

    def _parallel_special_block(
        self,
        residual,
        block_index,
        recurrent_source,
        recurrent_permutation,
        gate_override,
        return_diagnostics,
    ):
        block = self.base.transformer.h[block_index]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        normalized = block.ln_1(residual)
        query, local_key, local_value = block.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.local_mask(block_index, length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query, local_key, local_value, attn_mask=local_mask, is_causal=False
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )

        bank = None
        recurrent_weights = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            if recurrent_permutation is not None:
                recurrent_permutation = self._validate_permutation(
                    recurrent_permutation, batch, residual.device
                )
                recurrent_source = recurrent_source[recurrent_permutation]
            bank = self.build_recurrent_bank(block_index, recurrent_source)
            rec_key, rec_value = self.project_recurrent_kv(block_index, bank.values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                rec_key,
                rec_value,
                bank.valid_mask,
                return_weights=return_diagnostics,
            )
            coefficient = self._gate_coefficient(
                block_index, local_pre, gate_override
            )
        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        after_attention = residual + block.attn.c_proj(combined)
        output = after_attention + block.mlp(block.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        return output, {
            "block": block_index + 1,
            "source_block": {0: 12, 2: 10, 4: 8, 5: 7}[block_index],
            "local_attention_weights": local_weights,
            "recurrent_attention_weights": recurrent_weights,
            "local_valid_mask": local_mask,
            "recurrent_valid_mask": None if bank is None else bank.valid_mask,
            "recurrent_positions": None if bank is None else bank.positions,
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
            "gate_raw": self.gate_parameter(block_index),
            "gate_coefficient": coefficient,
        }

    @staticmethod
    def _validate_source(name, source, tokens, channels):
        if source is None:
            return
        expected = (tokens.size(0), tokens.size(1), channels)
        if not torch.is_tensor(source) or tuple(source.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if source.device != tokens.device:
            raise ValueError(f"{name} and tokens must share a device")

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
        bank_mode="full",
    ):
        if not torch.is_tensor(tokens) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        if targets is not None and tuple(targets.shape) != tuple(tokens.shape):
            raise ValueError("targets must match tokens")
        mode = self._validate_bank_mode(bank_mode)
        channels = int(self.config.n_embd)
        sources = {
            0: b1_recurrent_source,
            2: b3_recurrent_source,
            4: b5_recurrent_source,
            5: b6_recurrent_source,
        }
        permutations = {
            0: b1_recurrent_permutation,
            2: b3_recurrent_permutation,
            4: b5_recurrent_permutation,
            5: b6_recurrent_permutation,
        }
        overrides = {
            0: b1_gate_override,
            2: b3_gate_override,
            4: b5_gate_override,
            5: b6_gate_override,
        }
        for block_index, source in sources.items():
            self._validate_source(
                f"B{dict(((0, 12), (2, 10), (4, 8), (5, 7)))[block_index]} source",
                source,
                tokens,
                channels,
            )
            if source is None and permutations[block_index] is not None:
                raise ValueError("recurrent permutation requires a source")
        full = {int(value) for value in full_counterfactual_blocks}
        if not full.issubset(SPECIAL_BLOCKS):
            raise ValueError("full counterfactual is valid only for B1/B3/B5/B6")

        length = tokens.size(1)
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        residual = self.base.transformer.wte(tokens) + self.base.transformer.wpe(positions)
        use_checkpoint = bool(
            activation_checkpointing and self.training and torch.is_grad_enabled()
        )
        captures = {}
        diagnostics = {}
        previous_mode = self._active_bank_mode
        self._active_bank_mode = mode
        try:
            for block_index in range(TOTAL_LAYERS):
                block = self.base.transformer.h[block_index]
                if block_index in SPECIAL_BLOCKS and block_index not in full:
                    source = sources[block_index]
                    if use_checkpoint and not return_diagnostics:
                        if source is None:
                            residual = checkpoint(
                                lambda value, bi=block_index: self._parallel_special_block(
                                    value,
                                    bi,
                                    None,
                                    None,
                                    overrides[bi],
                                    False,
                                )[0],
                                residual,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        else:
                            residual = checkpoint(
                                lambda value, memory, bi=block_index: self._parallel_special_block(
                                    value,
                                    bi,
                                    memory,
                                    permutations[bi],
                                    overrides[bi],
                                    False,
                                )[0],
                                residual,
                                source,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                    else:
                        residual, diagnostics[block_index] = self._parallel_special_block(
                            residual,
                            block_index,
                            source,
                            permutations[block_index],
                            overrides[block_index],
                            return_diagnostics,
                        )
                else:
                    if use_checkpoint:
                        residual = checkpoint(
                            block,
                            residual,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        residual = block(residual)
                if block_index in SOURCE_BLOCKS:
                    captures[block_index] = residual
            h7, h8, h10, h12 = (
                captures[6], captures[7], captures[9], captures[11]
            )
            top = self.base.transformer.ln_f(h12)
            logits = self.base.lm_head(top)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            return {
                "h7": h7,
                "raw_h7": h7,
                "h8": h8,
                "raw_h8": h8,
                "h10": h10,
                "raw_h10": h10,
                "h12": h12,
                "raw_h12": h12,
                "top": top,
                "logits": logits,
                "loss": loss,
                "diagnostics": diagnostics if return_diagnostics else None,
                "full_counterfactual_blocks": tuple(sorted(full)),
            }
        finally:
            self._active_bank_mode = previous_mode

    def forward(self, tokens, targets=None, **kwargs):
        return self.forward_pass(tokens, targets=targets, **kwargs)

    def forward_multi_pass(
        self,
        tokens,
        targets=None,
        num_passes=2,
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
        bank_mode="full",
    ):
        if int(num_passes) not in (2, 3):
            raise ValueError("2D3A permits exactly two or three training passes")
        results = [
            self.forward_pass(
                tokens,
                targets,
                full_counterfactual_blocks=full_counterfactual_blocks,
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
                    targets,
                    b1_recurrent_source=previous["h12"],
                    b3_recurrent_source=previous["h10"],
                    b5_recurrent_source=previous["h8"],
                    b6_recurrent_source=previous["h7"],
                    b1_recurrent_permutation=b1_recurrent_permutation,
                    b3_recurrent_permutation=b3_recurrent_permutation,
                    b5_recurrent_permutation=b5_recurrent_permutation,
                    b6_recurrent_permutation=b6_recurrent_permutation,
                    b1_gate_override=b1_gate_override,
                    b3_gate_override=b3_gate_override,
                    b5_gate_override=b5_gate_override,
                    b6_gate_override=b6_gate_override,
                    full_counterfactual_blocks=full_counterfactual_blocks,
                    activation_checkpointing=activation_checkpointing,
                    return_diagnostics=return_diagnostics,
                    bank_mode=bank_mode,
                )
            )
        weights = (0.25, 0.75) if len(results) == 2 else (0.20, 0.40, 0.40)
        weighted = None
        if targets is not None:
            weighted = sum(weight * row["loss"] for weight, row in zip(weights, results))
        final = results[-1]
        return {
            "passes": tuple(results),
            "pass_weights": weights,
            "pass_losses": tuple(row["loss"] for row in results),
            "loss": weighted,
            **{name: final[name] for name in ("h7", "h8", "h10", "h12", "top", "logits")},
            "diagnostics": tuple(row["diagnostics"] for row in results),
        }

    def init_incremental_state(self, batch_size, device=None, dtype=None, b6_full_native=False):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        # Deployment state is explicitly BF16; keeping FP32 embedding weights
        # must not silently double the raw recurrent-ring footprint.
        dtype = (
            torch.bfloat16
            if dtype is None and device.type == "cuda"
            else reference.dtype if dtype is None else dtype
        )
        empty = torch.empty(
            (batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype
        )
        return PyramidIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * TOTAL_LAYERS,
            h7_ring=empty,
            h7_positions=(),
            h8_ring=empty.clone(),
            h8_positions=(),
            h10_ring=empty.clone(),
            h10_positions=(),
            h12_ring=empty.clone(),
            h12_positions=(),
            b6_full_native=bool(b6_full_native),
        )

    @staticmethod
    def incremental_cache_lengths(state):
        if not isinstance(state, PyramidIncrementalState):
            raise TypeError("state must be PyramidIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

    def _expected_cache_lengths(self, state):
        upper = min(state.position, int(self.config.block_size) - 1)
        result = []
        for block_index in range(TOTAL_LAYERS):
            if block_index in LOCAL_WINDOWS:
                capacity = LOCAL_WINDOWS[block_index] - 1
                if block_index == 5 and state.b6_full_native:
                    capacity = int(self.config.block_size) - 1
            else:
                capacity = int(self.config.block_size) - 1
            result.append(min(state.position, capacity))
        return tuple(result)

    def _validate_incremental_state(self, state):
        if not isinstance(state, PyramidIncrementalState):
            raise TypeError("incremental_step requires PyramidIncrementalState")
        if not 0 <= state.position <= int(self.config.block_size):
            raise ValueError("incremental position outside model context")
        if len(state.caches) != TOTAL_LAYERS:
            raise ValueError("incremental cache count mismatch")
        lengths = self.incremental_cache_lengths(state)
        expected = self._expected_cache_lengths(state)
        if lengths != expected:
            raise ValueError(f"incremental cache lengths {lengths} != {expected}")
        expected_positions = tuple(
            range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position)
        )
        for name, ring, positions in (
            ("h7", state.h7_ring, state.h7_positions),
            ("h8", state.h8_ring, state.h8_positions),
            ("h10", state.h10_ring, state.h10_positions),
            ("h12", state.h12_ring, state.h12_positions),
        ):
            if positions != expected_positions:
                raise ValueError(f"incremental {name} positions mismatch")
            expected_shape = (
                state.batch_size,
                len(expected_positions),
                int(self.config.n_embd),
            )
            if tuple(ring.shape) != expected_shape:
                raise ValueError(f"incremental {name} ring shape mismatch")
        for block_index, cache in enumerate(state.caches):
            if cache is None:
                continue
            if cache.key.ndim != 4 or cache.key.shape != cache.value.shape:
                raise ValueError(f"B{block_index + 1} cache shape mismatch")
            if cache.key.size(0) != state.batch_size:
                raise ValueError(f"B{block_index + 1} cache batch mismatch")

    def _incremental_bank_from_ring(self, ring, positions, state_position, minimum_lag, mode):
        mode = self._validate_bank_mode(mode)
        absolute = torch.tensor(positions, device=ring.device, dtype=torch.long)
        if absolute.numel() == 0:
            selected = absolute
        else:
            lag = int(state_position) - absolute
            if mode in {"full", "old_only"}:
                valid = (lag >= minimum_lag) & (lag <= RECURRENT_MAX_LAG)
            elif mode == "two_slot":
                valid = (lag >= minimum_lag) & (lag < minimum_lag + 2)
            elif mode == "recent_only" and minimum_lag == 2:
                valid = (lag >= 2) & (lag <= 31)
            else:
                valid = torch.zeros_like(lag, dtype=torch.bool)
            selected = absolute[valid]
        if selected.numel() == 0:
            values = ring[:, :0]
        else:
            values = ring.index_select(1, selected - positions[0])
        return FullRecurrentBank(
            values=values,
            valid_mask=torch.ones((1, selected.numel()), device=ring.device, dtype=torch.bool),
            positions=selected.view(1, -1),
        )

    @staticmethod
    def _append_ring(ring, positions, value, position):
        updated = torch.cat(
            (ring, value.detach().to(device=ring.device, dtype=ring.dtype)), dim=1
        )
        updated_positions: Sequence[int] = (*positions, int(position))
        if updated.size(1) > RECURRENT_RING_CAPACITY:
            updated = updated[:, -RECURRENT_RING_CAPACITY:].detach().clone(
                memory_format=torch.contiguous_format
            )
            updated_positions = updated_positions[-RECURRENT_RING_CAPACITY:]
        return updated, tuple(int(item) for item in updated_positions)

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
    ):
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
        local_keys = current_key if cache is None else torch.cat((cache.key, current_key), dim=2)
        local_values = current_value if cache is None else torch.cat((cache.value, current_value), dim=2)
        if local_keys.size(2) > local_capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess local KV")
        local_pre = F.scaled_dot_product_attention(
            query, local_keys, local_values, is_causal=False
        )
        next_cache = self._append_cache(current_key, current_value, cache, local_capacity)

        recurrent_weights = None
        if recurrent_bank is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        elif recurrent_bank.values.size(1) == 0:
            recurrent_pre = torch.zeros_like(local_pre)
            recurrent_weights = local_pre.new_empty((batch, heads, 1, 0))
            coefficient = self._gate_coefficient(block_index, local_pre, gate_override)
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
            coefficient = self._gate_coefficient(block_index, local_pre, gate_override)
        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, 1, channels)
        residual = residual + block.attn.c_proj(combined)
        residual = residual + block.mlp(block.ln_2(residual))
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "recurrent_attention_weights": recurrent_weights,
                "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
                "gate_coefficient": coefficient,
                "recurrent_positions": (
                    None if recurrent_bank is None else recurrent_bank.positions
                ),
            }
        return residual, next_cache, diagnostics

    def _incremental_ordinary_block(self, residual, block_index, cache, capacity):
        block = self.base.transformer.h[block_index]
        batch = residual.size(0)
        channels = int(self.config.n_embd)
        heads = int(self.config.n_head)
        head_size = channels // heads
        normalized = block.ln_1(residual)
        query, key, value = block.attn.c_attn(normalized).split(channels, dim=-1)
        query = query.view(batch, 1, heads, head_size).transpose(1, 2)
        key = key.view(batch, 1, heads, head_size).transpose(1, 2)
        value = value.view(batch, 1, heads, head_size).transpose(1, 2)
        keys = key if cache is None else torch.cat((cache.key, key), dim=2)
        values = value if cache is None else torch.cat((cache.value, value), dim=2)
        if keys.size(2) > capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess KV")
        attention = F.scaled_dot_product_attention(query, keys, values, is_causal=False)
        attention = attention.transpose(1, 2).contiguous().view(batch, 1, channels)
        residual = residual + block.attn.c_proj(attention)
        residual = residual + block.mlp(block.ln_2(residual))
        return residual, self._append_cache(key, value, cache, capacity)

    @staticmethod
    def _control_sets(control):
        off = {
            "new_links_off": {2, 4, 5},
            "b1_off": {0},
            "b3_off": {2},
            "b5_off": {4},
            "b6_off": {5},
            "b6_full_native": {5},
        }.get(control, set())
        shuffled = {
            "b1_shuffled": {0},
            "b3_shuffled": {2},
            "b5_shuffled": {4},
            "b6_shuffled": {5},
            "all_new_shuffled": {2, 4, 5},
            "all_recurrent_shuffled": {0, 2, 4, 5},
        }.get(control, set())
        return off, shuffled

    def incremental_step(
        self,
        token,
        state,
        control="all_real",
        recurrent_permutation=None,
        return_diagnostics=False,
        bank_mode="full",
        diagnostic_attention_weights=True,
    ):
        self._validate_incremental_state(state)
        mode = self._validate_bank_mode(bank_mode)
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown incremental control: {control}")
        wants_full_b6 = control == "b6_full_native"
        if state.b6_full_native != wants_full_b6:
            raise ValueError("incremental B6 cache geometry does not match control")
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

        position = torch.tensor([state.position], device=token.device, dtype=torch.long)
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(position)
        rings = {
            0: (state.h12_ring, state.h12_positions),
            2: (state.h10_ring, state.h10_positions),
            4: (state.h8_ring, state.h8_positions),
            5: (state.h7_ring, state.h7_positions),
        }
        updated_caches = []
        diagnostics = {}
        captures = {}
        upper_capacity = int(self.config.block_size) - 1
        for block_index in range(TOTAL_LAYERS):
            if block_index in SPECIAL_BLOCKS and not (
                block_index == 5 and wants_full_b6
            ):
                ring, positions = rings[block_index]
                bank = self._incremental_bank_from_ring(
                    ring,
                    positions,
                    state.position,
                    MIN_LAGS[block_index],
                    mode,
                )
                block_off = block_index in off
                residual, cache, diag = self._incremental_special_block(
                    residual,
                    block_index,
                    state.caches[block_index],
                    None if block_off else bank,
                    permutation if block_index in shuffled else None,
                    0.0 if block_off else None,
                    LOCAL_WINDOWS[block_index] - 1,
                    return_diagnostics,
                    diagnostic_attention_weights,
                )
                diagnostics[block_index] = diag
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
        h7, h8, h10, h12 = captures[6], captures[7], captures[9], captures[11]
        logits = self.base.lm_head(self.base.transformer.ln_f(h12))
        next_h7, next_h7_pos = self._append_ring(
            state.h7_ring, state.h7_positions, h7, state.position
        )
        next_h8, next_h8_pos = self._append_ring(
            state.h8_ring, state.h8_positions, h8, state.position
        )
        next_h10, next_h10_pos = self._append_ring(
            state.h10_ring, state.h10_positions, h10, state.position
        )
        next_h12, next_h12_pos = self._append_ring(
            state.h12_ring, state.h12_positions, h12, state.position
        )
        next_state = PyramidIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h7_ring=next_h7,
            h7_positions=next_h7_pos,
            h8_ring=next_h8,
            h8_positions=next_h8_pos,
            h10_ring=next_h10,
            h10_positions=next_h10_pos,
            h12_ring=next_h12,
            h12_positions=next_h12_pos,
            b6_full_native=state.b6_full_native,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics:
            return logits, next_state
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "links": {f"b{index + 1}": row for index, row in diagnostics.items()},
            "cache_audit": self.incremental_cache_audit(next_state),
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
        for index, cache in enumerate(state.caches):
            cache_storage.append(
                {
                    "block": index + 1,
                    "key": None if cache is None else storage(cache.key),
                    "value": None if cache is None else storage(cache.value),
                }
            )
        ring_storage = {
            "h7": storage(state.h7_ring),
            "h8": storage(state.h8_ring),
            "h10": storage(state.h10_ring),
            "h12": storage(state.h12_ring),
        }
        limits = [1, 1023, 31, 1023, 63, 1023 if state.b6_full_native else 511]
        limits += [1023] * 6
        physical = all(row["exact"] for row in ring_storage.values()) and all(
            row["key"] is None or (row["key"]["exact"] and row["value"]["exact"])
            for row in cache_storage
        )
        passed = all(value <= limit for value, limit in zip(lengths, limits)) and all(
            ring.size(1) <= RECURRENT_RING_CAPACITY
            for ring in (state.h7_ring, state.h8_ring, state.h10_ring, state.h12_ring)
        ) and physical
        return {
            "position": state.position,
            "cache_lengths": list(lengths),
            "cache_limits": limits,
            "ring_lengths": {
                "h7": state.h7_ring.size(1),
                "h8": state.h8_ring.size(1),
                "h10": state.h10_ring.size(1),
                "h12": state.h12_ring.size(1),
            },
            "ring_limit": RECURRENT_RING_CAPACITY,
            "no_h9_or_h11_ring": True,
            "cache_storage": cache_storage,
            "ring_storage": ring_storage,
            "physical_storage_exact": bool(physical),
            "b6_full_native": state.b6_full_native,
            "passed": bool(passed),
        }

    def incremental_logits(
        self,
        tokens,
        control="all_real",
        recurrent_permutation=None,
        return_diagnostics=False,
        bank_mode="full",
        diagnostic_attention_weights=True,
    ):
        if not torch.is_tensor(tokens) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        state = self.init_incremental_state(
            tokens.size(0),
            device=tokens.device,
            dtype=torch.bfloat16 if tokens.device.type == "cuda" else self.base.transformer.wte.weight.dtype,
            b6_full_native=control == "b6_full_native",
        )
        logits = []
        diagnostics = []
        maxima = [0] * TOTAL_LAYERS
        ring_maxima = {"h7": 0, "h8": 0, "h10": 0, "h12": 0}
        for position in range(tokens.size(1)):
            result = self.incremental_step(
                tokens[:, position],
                state,
                control,
                recurrent_permutation,
                return_diagnostics,
                bank_mode,
                diagnostic_attention_weights,
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
            for name in ring_maxima:
                ring_maxima[name] = max(
                    ring_maxima[name], getattr(state, f"{name}_ring").size(1)
                )
        return {
            "logits": torch.cat(logits, dim=1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "max_ring_lengths": ring_maxima,
            "cache_audit": self.incremental_cache_audit(state),
        }


Experiment2D3AModel = AlternatingIntegrationRecurrentPyramidGPT
RecurrentKVGPT = AlternatingIntegrationRecurrentPyramidGPT

__all__ = [
    "BANK_MODES",
    "INCREMENTAL_CONTROLS",
    "LOCAL_WINDOWS",
    "MIN_LAGS",
    "MAX_RECURRENT_ENTRIES",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "PyramidIncrementalState",
    "AlternatingIntegrationRecurrentPyramidGPT",
    "Experiment2D3AModel",
    "RecurrentKVGPT",
]
