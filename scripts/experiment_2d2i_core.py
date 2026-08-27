"""Architecture kernel for Experiment 2D2I.

2D2I preserves the finalized B12->B1, B11->B2, and B10->B3 recurrent paths
from 2D2E and adds exactly one parameter: the scalar gate for B9->B4
recurrent K/V.  The ordinary windows are B1 W2, B2 W32, B3 W64, B4 W128,
and B5-B12 W1024.  Their complementary recurrent banks cover lags 2..1023,
32..1023, 64..1023, and 128..1023.
Each recurrent arm reuses its destination LayerNorm and fused K/V slices,
uses a separate softmax, and shares the destination ``c_proj`` exactly once.

The parallel path stores one ``[B,T,C]`` source tensor per link and never
constructs ``[B,T,T,C]`` state copies.  The incremental path stores raw
B9/B10/B11/B12 rings plus at most 1, 31, 63, and 127 historical K/V entries
for B1/B2/B3/B4 respectively; B5-B12 retain ordinary full-context caches.
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
    LOCAL_WINDOW as B1_LOCAL_WINDOW,
    MAX_RECURRENT_ENTRIES as B1_MAX_RECURRENT_ENTRIES,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
)


TOTAL_LAYERS = 12
B1_INDEX = 0
B2_INDEX = 1
B3_INDEX = 2
B4_INDEX = 3
B9_INDEX = 8
B10_INDEX = 9
B11_INDEX = 10
B12_INDEX = 11

B2_LOCAL_WINDOW = 32
B2_RECURRENT_MIN_LAG = 32
B2_MAX_RECURRENT_ENTRIES = RECURRENT_MAX_LAG - B2_RECURRENT_MIN_LAG + 1
B3_LOCAL_WINDOW = 64
B3_RECURRENT_MIN_LAG = 64
B3_MAX_RECURRENT_ENTRIES = RECURRENT_MAX_LAG - B3_RECURRENT_MIN_LAG + 1
B4_LOCAL_WINDOW = 128
B4_RECURRENT_MIN_LAG = 128
B4_MAX_RECURRENT_ENTRIES = RECURRENT_MAX_LAG - B4_RECURRENT_MIN_LAG + 1

# Compatibility names used by the experiment driver describe the B3 geometry
# under test.  The inherited B1/B2 geometry remains available explicitly.
LOCAL_WINDOW = B1_LOCAL_WINDOW
MAX_RECURRENT_ENTRIES = B4_MAX_RECURRENT_ENTRIES

INCREMENTAL_CONTROLS = (
    "all_real",
    "b4_off",
    "b4_shuffled",
    "b4_full_counterfactual",
    "all_shuffled",
)


@dataclass(frozen=True)
class FourLinkIncrementalState:
    """Deployment state for the four graded mirrored recurrent links."""

    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h9_ring: torch.Tensor
    h9_positions: Tuple[int, ...]
    h10_ring: torch.Tensor
    h10_positions: Tuple[int, ...]
    h11_ring: torch.Tensor
    h11_positions: Tuple[int, ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]
    b4_full_cache: bool = False


class FourLinkRecurrentKVGPT(FullB12ToB1RecurrentKVGPT):
    """Final 2D2E model plus one B9->B4 gate and no other parameters."""

    def __init__(self, base: nn.Module):
        super().__init__(base)
        # Existing 2D2D tensor, restored with its complete optimizer state.
        self.g_rec_b2 = nn.Parameter(torch.zeros(()))
        # Existing 2D2E tensor, restored with its complete optimizer state.
        self.g_rec_b3 = nn.Parameter(torch.zeros(()))
        # The sole new learnable tensor versus final 2D2E.
        self.g_rec_b4 = nn.Parameter(torch.zeros(()))
        self._b2_local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self._b3_local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self._b4_local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @property
    def g_rec_b1(self) -> torch.Tensor:
        """Non-registering alias for the inherited, trained 2D2B gate."""

        return self.g_rec

    @property
    def recurrent_scale_b1(self) -> torch.Tensor:
        return self.g_rec.tanh()

    @property
    def recurrent_scale_b2(self) -> torch.Tensor:
        return self.g_rec_b2.tanh()

    @property
    def recurrent_scale_b3(self) -> torch.Tensor:
        return self.g_rec_b3.tanh()

    @property
    def recurrent_scale_b4(self) -> torch.Tensor:
        return self.g_rec_b4.tanh()

    def b2_local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the exact B2 W32 causal mask, including the current token."""

        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._b2_local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (
                source >= query - (B2_LOCAL_WINDOW - 1)
            )
            self._b2_local_mask_cache[key] = mask
        return mask

    @staticmethod
    def b2_recurrent_mask(
        query_length: int,
        source_length: int,
        device: torch.device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        """Return B2's exact older-memory mask without changing B1 geometry."""

        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        if bank_mode in {"full", "old_only"}:
            return (lag >= B2_RECURRENT_MIN_LAG) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= B2_RECURRENT_MIN_LAG) & (
                lag < B2_RECURRENT_MIN_LAG + 2
            )
        # ``recent_only`` means lags 2...31 for the inherited B1 diagnostic.
        # Those lags belong exclusively to B2 local attention in 2D2E.
        return torch.zeros_like(lag, dtype=torch.bool)

    def build_recurrent_bank_b2(
        self, recurrent_source: torch.Tensor, bank_mode: Optional[str] = None
    ) -> FullRecurrentBank:
        """Build B2's lags-32...1023 bank backed by one source tensor."""

        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("B11 recurrent source must have shape [batch,time,channel]")
        _, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd):
            raise ValueError("B11 recurrent source channel width does not match the model")
        if length < 1:
            raise ValueError("B11 recurrent source time dimension must be nonempty")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        valid = self.b2_recurrent_mask(
            length, length, recurrent_source.device, bank_mode=mode
        )
        source_positions = torch.arange(
            length, device=recurrent_source.device, dtype=torch.long
        ).view(1, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=source_positions.expand(length, length),
        )

    def b3_local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the exact B3 W64 causal mask, including the current token."""

        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._b3_local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (
                source >= query - (B3_LOCAL_WINDOW - 1)
            )
            self._b3_local_mask_cache[key] = mask
        return mask

    @staticmethod
    def b3_recurrent_mask(
        query_length: int,
        source_length: int,
        device: torch.device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        """Return B3's exact B10-memory mask for lags 64..1023."""

        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        if bank_mode in {"full", "old_only"}:
            return (lag >= B3_RECURRENT_MIN_LAG) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= B3_RECURRENT_MIN_LAG) & (
                lag < B3_RECURRENT_MIN_LAG + 2
            )
        return torch.zeros_like(lag, dtype=torch.bool)

    def build_recurrent_bank_b3(
        self, recurrent_source: torch.Tensor, bank_mode: Optional[str] = None
    ) -> FullRecurrentBank:
        """Build B3's lags-64..1023 bank backed by one B10 source tensor."""

        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("B10 recurrent source must have shape [batch,time,channel]")
        _, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd):
            raise ValueError("B10 recurrent source channel width does not match the model")
        if length < 1:
            raise ValueError("B10 recurrent source time dimension must be nonempty")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        valid = self.b3_recurrent_mask(
            length, length, recurrent_source.device, bank_mode=mode
        )
        source_positions = torch.arange(
            length, device=recurrent_source.device, dtype=torch.long
        ).view(1, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=source_positions.expand(length, length),
        )

    def b4_local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the exact B4 W128 causal mask, including current token."""

        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._b4_local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (
                source >= query - (B4_LOCAL_WINDOW - 1)
            )
            self._b4_local_mask_cache[key] = mask
        return mask

    @staticmethod
    def b4_recurrent_mask(
        query_length: int,
        source_length: int,
        device: torch.device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        """Return B4's exact B9-memory mask for lags 128..1023."""

        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        if bank_mode in {"full", "old_only"}:
            return (lag >= B4_RECURRENT_MIN_LAG) & (lag <= RECURRENT_MAX_LAG)
        if bank_mode == "two_slot":
            return (lag >= B4_RECURRENT_MIN_LAG) & (
                lag < B4_RECURRENT_MIN_LAG + 2
            )
        return torch.zeros_like(lag, dtype=torch.bool)

    def build_recurrent_bank_b4(
        self, recurrent_source: torch.Tensor, bank_mode: Optional[str] = None
    ) -> FullRecurrentBank:
        """Build B4's lags-128..1023 bank backed by one B9 source tensor."""

        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("B9 recurrent source must have shape [batch,time,channel]")
        _, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd):
            raise ValueError("B9 recurrent source channel width does not match the model")
        if length < 1:
            raise ValueError("B9 recurrent source time dimension must be nonempty")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        valid = self.b4_recurrent_mask(
            length, length, recurrent_source.device, bank_mode=mode
        )
        source_positions = torch.arange(
            length, device=recurrent_source.device, dtype=torch.long
        ).view(1, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=source_positions.expand(length, length),
        )

    @staticmethod
    def _diagnostic_attention_weights(
        query: torch.Tensor, key: torch.Tensor, valid_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Materialize exact diagnostic weights only for safely small probes."""

        elements = query.size(0) * query.size(1) * query.size(2) * key.size(2)
        if elements > ATTENTION_WEIGHT_ELEMENT_LIMIT:
            return None
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores.mul_(query.size(-1) ** -0.5)
        mask = valid_mask.view(1, 1, query.size(2), key.size(2))
        row_valid = mask.any(dim=-1, keepdim=True)
        masked = scores.masked_fill(~mask, -torch.inf)
        safe = torch.where(row_valid, masked, torch.zeros_like(masked))
        return F.softmax(safe, dim=-1) * mask.to(safe.dtype)

    def _gate_coefficient_b2(
        self, reference: torch.Tensor, gate_override: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_override is None:
            coefficient = self.recurrent_scale_b2
        elif isinstance(gate_override, torch.Tensor):
            if gate_override.numel() != 1:
                raise ValueError("B2 gate override must be scalar")
            coefficient = gate_override.reshape(())
        else:
            coefficient = reference.new_tensor(float(gate_override))
        return coefficient.to(device=reference.device, dtype=reference.dtype)

    def _gate_coefficient_b3(
        self, reference: torch.Tensor, gate_override: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_override is None:
            coefficient = self.recurrent_scale_b3
        elif isinstance(gate_override, torch.Tensor):
            if gate_override.numel() != 1:
                raise ValueError("B3 gate override must be scalar")
            coefficient = gate_override.reshape(())
        else:
            coefficient = reference.new_tensor(float(gate_override))
        return coefficient.to(device=reference.device, dtype=reference.dtype)

    def _gate_coefficient_b4(
        self, reference: torch.Tensor, gate_override: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_override is None:
            coefficient = self.recurrent_scale_b4
        elif isinstance(gate_override, torch.Tensor):
            if gate_override.numel() != 1:
                raise ValueError("B4 gate override must be scalar")
            coefficient = gate_override.reshape(())
        else:
            coefficient = reference.new_tensor(float(gate_override))
        return coefficient.to(device=reference.device, dtype=reference.dtype)

    def project_recurrent_kv_b2(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply B2's unchanged LN and fused K/V slices once per B11 source."""

        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 3:
            raise ValueError("B11 recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels:
            raise ValueError("B11 recurrent values have the wrong channel dimension")
        block2 = self.base.transformer.h[B2_INDEX]
        normalized = block2.ln_1(bank_values)
        _, key, value = block2.attn.c_attn(normalized).split(channels, dim=-1)
        batch, source_length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, source_length, heads, head_size).transpose(1, 2)
        value = value.view(batch, source_length, heads, head_size).transpose(1, 2)
        return key, value

    def project_recurrent_kv_b3(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply B3's unchanged LN and fused K/V slices once per B10 source."""

        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 3:
            raise ValueError("B10 recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels:
            raise ValueError("B10 recurrent values have the wrong channel dimension")
        block3 = self.base.transformer.h[B3_INDEX]
        normalized = block3.ln_1(bank_values)
        _, key, value = block3.attn.c_attn(normalized).split(channels, dim=-1)
        batch, source_length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, source_length, heads, head_size).transpose(1, 2)
        value = value.view(batch, source_length, heads, head_size).transpose(1, 2)
        return key, value

    def project_recurrent_kv_b4(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply B4's unchanged LN and fused K/V slices once per B9 source."""

        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 3:
            raise ValueError("B9 recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels:
            raise ValueError("B9 recurrent values have the wrong channel dimension")
        block4 = self.base.transformer.h[B4_INDEX]
        normalized = block4.ln_1(bank_values)
        _, key, value = block4.attn.c_attn(normalized).split(channels, dim=-1)
        batch, source_length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, source_length, heads, head_size).transpose(1, 2)
        value = value.view(batch, source_length, heads, head_size).transpose(1, 2)
        return key, value

    def _parallel_block2(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ):
        block2 = self.base.transformer.h[B2_INDEX]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block2.ln_1(residual)
        query, local_key, local_value = block2.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.b2_local_mask(length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query,
            local_key,
            local_value,
            attn_mask=local_mask,
            is_causal=False,
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )

        recurrent_weights = None
        recurrent_bank = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self.build_recurrent_bank_b2(recurrent_source)
            recurrent_key, recurrent_value = self.project_recurrent_kv_b2(
                recurrent_bank.values
            )
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                recurrent_key,
                recurrent_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics,
            )
            coefficient = self._gate_coefficient_b2(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        # Exactly one ordinary B2 c_proj after combining the two pre-projection arms.
        attention_output = block2.attn.c_proj(combined)
        after_attention = residual + attention_output
        output = after_attention + block2.mlp(block2.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return output, {
            "bank_mode": self._active_bank_mode,
            "recurrent_attention_weights": recurrent_weights,
            "local_attention_weights": local_weights,
            "attention_weights_omitted_for_memory": bool(
                recurrent_source is not None and recurrent_weights is None
            ),
            "local_attention_weights_omitted_for_memory": bool(
                return_diagnostics and local_weights is None
            ),
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec_b2,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "local_valid_mask": local_mask,
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
        }

    def _parallel_block3(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ):
        block3 = self.base.transformer.h[B3_INDEX]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block3.ln_1(residual)
        query, local_key, local_value = block3.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.b3_local_mask(length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query,
            local_key,
            local_value,
            attn_mask=local_mask,
            is_causal=False,
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )

        recurrent_weights = None
        recurrent_bank = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self.build_recurrent_bank_b3(recurrent_source)
            recurrent_key, recurrent_value = self.project_recurrent_kv_b3(
                recurrent_bank.values
            )
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                recurrent_key,
                recurrent_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics,
            )
            coefficient = self._gate_coefficient_b3(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        # Exactly one ordinary B3 c_proj after combining the two pre-projection arms.
        attention_output = block3.attn.c_proj(combined)
        after_attention = residual + attention_output
        output = after_attention + block3.mlp(block3.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return output, {
            "bank_mode": self._active_bank_mode,
            "recurrent_attention_weights": recurrent_weights,
            "local_attention_weights": local_weights,
            "attention_weights_omitted_for_memory": bool(
                recurrent_source is not None and recurrent_weights is None
            ),
            "local_attention_weights_omitted_for_memory": bool(
                return_diagnostics and local_weights is None
            ),
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec_b3,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "local_valid_mask": local_mask,
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
        }

    def _parallel_block4(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ):
        block4 = self.base.transformer.h[B4_INDEX]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block4.ln_1(residual)
        query, local_key, local_value = block4.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.b4_local_mask(length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query, local_key, local_value, attn_mask=local_mask, is_causal=False
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )

        recurrent_weights = None
        recurrent_bank = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self.build_recurrent_bank_b4(recurrent_source)
            recurrent_key, recurrent_value = self.project_recurrent_kv_b4(
                recurrent_bank.values
            )
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                recurrent_key,
                recurrent_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics,
            )
            coefficient = self._gate_coefficient_b4(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        # Exactly one ordinary B4 c_proj after combining both pre-projection arms.
        attention_output = block4.attn.c_proj(combined)
        after_attention = residual + attention_output
        output = after_attention + block4.mlp(block4.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return output, {
            "bank_mode": self._active_bank_mode,
            "recurrent_attention_weights": recurrent_weights,
            "local_attention_weights": local_weights,
            "attention_weights_omitted_for_memory": bool(
                recurrent_source is not None and recurrent_weights is None
            ),
            "local_attention_weights_omitted_for_memory": bool(
                return_diagnostics and local_weights is None
            ),
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec_b4,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "local_valid_mask": local_mask,
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
        }

    @staticmethod
    def _validate_source(
        name: str,
        source: Optional[torch.Tensor],
        tokens: torch.Tensor,
        channels: int,
    ) -> None:
        if source is None:
            return
        expected = (tokens.size(0), tokens.size(1), channels)
        if not isinstance(source, torch.Tensor) or tuple(source.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if source.device != tokens.device:
            raise ValueError(f"{name} and tokens must be on the same device")

    def forward_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        b1_recurrent_source: Optional[torch.Tensor] = None,
        b2_recurrent_source: Optional[torch.Tensor] = None,
        b3_recurrent_source: Optional[torch.Tensor] = None,
        b4_recurrent_source: Optional[torch.Tensor] = None,
        b1_recurrent_permutation: Optional[torch.Tensor] = None,
        b2_recurrent_permutation: Optional[torch.Tensor] = None,
        b3_recurrent_permutation: Optional[torch.Tensor] = None,
        b4_recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b4_gate_override: Optional[torch.Tensor] = None,
        b3_full_counterfactual: bool = False,
        b4_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        """Run one pass and expose exact post-MLP B9/B10/B11/B12 streams."""

        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        if targets is not None and tuple(targets.shape) != tuple(tokens.shape):
            raise ValueError("targets must match tokens")
        mode = self._validate_bank_mode(bank_mode)
        channels = int(self.config.n_embd)
        self._validate_source("B12 source", b1_recurrent_source, tokens, channels)
        self._validate_source("B11 source", b2_recurrent_source, tokens, channels)
        self._validate_source("B10 source", b3_recurrent_source, tokens, channels)
        self._validate_source("B9 source", b4_recurrent_source, tokens, channels)

        if b1_recurrent_source is None:
            if b1_recurrent_permutation is not None:
                raise ValueError("B1 permutation requires a B12 source")
            b1_source = None
        elif b1_recurrent_permutation is None:
            b1_source = b1_recurrent_source
        else:
            permutation = self._validate_permutation(
                b1_recurrent_permutation, tokens.size(0), tokens.device
            )
            b1_source = b1_recurrent_source[permutation]

        if b2_recurrent_source is None:
            if b2_recurrent_permutation is not None:
                raise ValueError("B2 permutation requires a B11 source")
            b2_source = None
        elif b2_recurrent_permutation is None:
            b2_source = b2_recurrent_source
        else:
            permutation = self._validate_permutation(
                b2_recurrent_permutation, tokens.size(0), tokens.device
            )
            b2_source = b2_recurrent_source[permutation]

        if b3_recurrent_source is None:
            if b3_recurrent_permutation is not None:
                raise ValueError("B3 permutation requires a B10 source")
            b3_source = None
        elif b3_recurrent_permutation is None:
            b3_source = b3_recurrent_source
        else:
            permutation = self._validate_permutation(
                b3_recurrent_permutation, tokens.size(0), tokens.device
            )
            b3_source = b3_recurrent_source[permutation]

        if b4_recurrent_source is None:
            if b4_recurrent_permutation is not None:
                raise ValueError("B4 permutation requires a B9 source")
            b4_source = None
        elif b4_recurrent_permutation is None:
            b4_source = b4_recurrent_source
        else:
            permutation = self._validate_permutation(
                b4_recurrent_permutation, tokens.size(0), tokens.device
            )
            b4_source = b4_recurrent_source[permutation]

        _, length = tokens.shape
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        residual = self.base.transformer.wte(tokens) + self.base.transformer.wpe(positions)
        use_checkpoint = bool(
            activation_checkpointing and self.training and torch.is_grad_enabled()
        )
        previous_mode = self._active_bank_mode
        self._active_bank_mode = mode
        try:
            b1_diagnostics = None
            if use_checkpoint and not return_diagnostics:
                if b1_source is None:
                    residual = checkpoint(
                        lambda value: self._parallel_block1(
                            value, None, b1_gate_override, False
                        )[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda value, memory: self._parallel_block1(
                            value, memory, b1_gate_override, False
                        )[0],
                        residual,
                        b1_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b1_diagnostics = self._parallel_block1(
                    residual, b1_source, b1_gate_override, return_diagnostics
                )

            b2_diagnostics = None
            if use_checkpoint and not return_diagnostics:
                if b2_source is None:
                    residual = checkpoint(
                        lambda value: self._parallel_block2(
                            value, None, b2_gate_override, False
                        )[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda value, memory: self._parallel_block2(
                            value, memory, b2_gate_override, False
                        )[0],
                        residual,
                        b2_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b2_diagnostics = self._parallel_block2(
                    residual, b2_source, b2_gate_override, return_diagnostics
                )

            b3_diagnostics = None
            block3 = self.base.transformer.h[B3_INDEX]
            if b3_full_counterfactual:
                if use_checkpoint:
                    residual = checkpoint(
                        block3,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = block3(residual)
            elif use_checkpoint and not return_diagnostics:
                if b3_source is None:
                    residual = checkpoint(
                        lambda value: self._parallel_block3(
                            value, None, b3_gate_override, False
                        )[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda value, memory: self._parallel_block3(
                            value, memory, b3_gate_override, False
                        )[0],
                        residual,
                        b3_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b3_diagnostics = self._parallel_block3(
                    residual, b3_source, b3_gate_override, return_diagnostics
                )

            b4_diagnostics = None
            block4 = self.base.transformer.h[B4_INDEX]
            if b4_full_counterfactual:
                if use_checkpoint:
                    residual = checkpoint(
                        block4,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = block4(residual)
            elif use_checkpoint and not return_diagnostics:
                if b4_source is None:
                    residual = checkpoint(
                        lambda value: self._parallel_block4(
                            value, None, b4_gate_override, False
                        )[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda value, memory: self._parallel_block4(
                            value, memory, b4_gate_override, False
                        )[0],
                        residual,
                        b4_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b4_diagnostics = self._parallel_block4(
                    residual, b4_source, b4_gate_override, return_diagnostics
                )

            h9 = None
            h10 = None
            h11 = None
            for block_index in range(4, TOTAL_LAYERS):
                block = self.base.transformer.h[block_index]
                if use_checkpoint:
                    residual = checkpoint(
                        block,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = block(residual)
                if block_index == B9_INDEX:
                    h9 = residual
                if block_index == B10_INDEX:
                    h10 = residual
                if block_index == B11_INDEX:
                    h11 = residual
            if h9 is None or h10 is None or h11 is None:
                raise RuntimeError("B9/B10/B11 source capture was not reached")
            h12 = residual
            top = self.base.transformer.ln_f(h12)
            logits = self.base.lm_head(top)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            return {
                "h9": h9,
                "raw_h9": h9,
                "h10": h10,
                "raw_h10": h10,
                "h11": h11,
                "raw_h11": h11,
                "h12": h12,
                "raw_h12": h12,
                "top": top,
                "logits": logits,
                "loss": loss,
                "diagnostics": {
                    "b1": b1_diagnostics,
                    "b2": b2_diagnostics,
                    "b3": b3_diagnostics,
                    "b4": b4_diagnostics,
                    "b3_full_counterfactual": bool(b3_full_counterfactual),
                    "b4_full_counterfactual": bool(b4_full_counterfactual),
                }
                if return_diagnostics
                else None,
            }
        finally:
            self._active_bank_mode = previous_mode

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs):
        return self.forward_pass(tokens, targets=targets, **kwargs)

    def forward_multi_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_passes: int = 2,
        b1_recurrent_permutation: Optional[torch.Tensor] = None,
        b2_recurrent_permutation: Optional[torch.Tensor] = None,
        b3_recurrent_permutation: Optional[torch.Tensor] = None,
        b4_recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b4_gate_override: Optional[torch.Tensor] = None,
        b3_full_counterfactual: bool = False,
        b4_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        """Run attached two/three-pass recurrence with unchanged CE weighting."""

        if int(num_passes) not in (2, 3):
            raise ValueError("2D2I multi-pass execution permits exactly two or three passes")
        results = [
            self.forward_pass(
                tokens,
                targets=targets,
                b3_full_counterfactual=b3_full_counterfactual,
                b4_full_counterfactual=b4_full_counterfactual,
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
                    b2_recurrent_source=previous["h11"],
                    b3_recurrent_source=previous["h10"],
                    b4_recurrent_source=previous["h9"],
                    b1_recurrent_permutation=b1_recurrent_permutation,
                    b2_recurrent_permutation=b2_recurrent_permutation,
                    b3_recurrent_permutation=b3_recurrent_permutation,
                    b4_recurrent_permutation=b4_recurrent_permutation,
                    b1_gate_override=b1_gate_override,
                    b2_gate_override=b2_gate_override,
                    b3_gate_override=b3_gate_override,
                    b4_gate_override=b4_gate_override,
                    b3_full_counterfactual=b3_full_counterfactual,
                    b4_full_counterfactual=b4_full_counterfactual,
                    activation_checkpointing=activation_checkpointing,
                    return_diagnostics=return_diagnostics,
                    bank_mode=bank_mode,
                )
            )
        weights = (0.25, 0.75) if int(num_passes) == 2 else (0.20, 0.40, 0.40)
        weighted_loss = None
        if targets is not None:
            weighted_loss = sum(
                weight * result["loss"] for weight, result in zip(weights, results)
            )
        final = results[-1]
        return {
            "passes": tuple(results),
            "pass_weights": weights,
            "pass_losses": tuple(result["loss"] for result in results),
            "loss": weighted_loss,
            "h9": final["h9"],
            "raw_h9": final["h9"],
            "h10": final["h10"],
            "raw_h10": final["h10"],
            "h11": final["h11"],
            "raw_h11": final["h11"],
            "h12": final["h12"],
            "raw_h12": final["h12"],
            "top": final["top"],
            "logits": final["logits"],
            "diagnostics": tuple(result["diagnostics"] for result in results),
        }

    def init_incremental_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        b4_full_cache: bool = False,
    ) -> FourLinkIncrementalState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        empty = torch.empty(
            (batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype
        )
        return FourLinkIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * int(self.config.n_layer),
            h9_ring=empty,
            h9_positions=(),
            h10_ring=empty.clone(),
            h10_positions=(),
            h11_ring=empty.clone(),
            h11_positions=(),
            h12_ring=empty.clone(),
            h12_positions=(),
            b4_full_cache=bool(b4_full_cache),
        )

    @staticmethod
    def incremental_cache_lengths(
        state: FourLinkIncrementalState,
    ) -> Tuple[int, ...]:
        if not isinstance(state, FourLinkIncrementalState):
            raise TypeError("state must be FourLinkIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

    def _validate_incremental_state(self, state: FourLinkIncrementalState) -> None:
        if not isinstance(state, FourLinkIncrementalState):
            raise TypeError("incremental_step requires FourLinkIncrementalState")
        if not 0 <= int(state.position) <= int(self.config.block_size):
            raise ValueError("incremental position is outside the model context")
        if len(state.caches) != int(self.config.n_layer):
            raise ValueError("incremental state has the wrong number of caches")
        upper = min(state.position, int(self.config.block_size) - 1)
        b4 = (
            upper
            if state.b4_full_cache
            else min(state.position, B4_LOCAL_WINDOW - 1)
        )
        expected = (
            min(state.position, B1_LOCAL_WINDOW - 1),
            min(state.position, B2_LOCAL_WINDOW - 1),
            min(state.position, B3_LOCAL_WINDOW - 1),
            b4,
        ) + (upper,) * 8
        lengths = self.incremental_cache_lengths(state)
        if lengths != expected:
            raise ValueError(f"incremental cache lengths {lengths} != {expected}")
        expected_positions = tuple(
            range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position)
        )
        for name, ring, positions in (
            ("h9", state.h9_ring, state.h9_positions),
            ("h10", state.h10_ring, state.h10_positions),
            ("h11", state.h11_ring, state.h11_positions),
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
            if cache.key.shape != cache.value.shape or cache.key.ndim != 4:
                raise ValueError(f"B{block_index + 1} cache K/V shapes do not match")
            if cache.key.size(0) != state.batch_size:
                raise ValueError(f"B{block_index + 1} cache batch mismatch")

    def _incremental_bank_from_ring(
        self,
        ring: torch.Tensor,
        positions: Tuple[int, ...],
        state_position: int,
        bank_mode: str,
        minimum_lag: int = 2,
    ) -> FullRecurrentBank:
        mode = self._validate_bank_mode(bank_mode)
        absolute = torch.tensor(positions, device=ring.device, dtype=torch.long)
        if absolute.numel() == 0:
            selected = absolute
        else:
            lag = int(state_position) - absolute
            if mode == "full":
                valid = (lag >= minimum_lag) & (lag <= RECURRENT_MAX_LAG)
            elif mode == "two_slot":
                valid = (lag >= minimum_lag) & (lag < minimum_lag + 2)
            elif mode == "recent_only":
                valid = (
                    (lag >= 2) & (lag <= 31)
                    if minimum_lag <= 2
                    else torch.zeros_like(lag, dtype=torch.bool)
                )
            else:
                valid = (lag >= max(minimum_lag, 32)) & (
                    lag <= RECURRENT_MAX_LAG
                )
            selected = absolute[valid]
        if selected.numel() == 0:
            values = ring[:, :0]
        else:
            first = positions[0]
            values = ring.index_select(1, selected - first)
        valid_mask = torch.ones(
            (1, selected.numel()), device=ring.device, dtype=torch.bool
        )
        return FullRecurrentBank(
            values=values,
            valid_mask=valid_mask,
            positions=selected.view(1, -1),
        )

    @staticmethod
    def _append_ring(
        ring: torch.Tensor,
        positions: Tuple[int, ...],
        value: torch.Tensor,
        position: int,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        updated = torch.cat((ring, value.detach()), dim=1)
        updated_positions: Sequence[int] = (*positions, int(position))
        if updated.size(1) > RECURRENT_RING_CAPACITY:
            updated = updated[:, -RECURRENT_RING_CAPACITY:].detach().clone(
                memory_format=torch.contiguous_format
            )
            updated_positions = updated_positions[-RECURRENT_RING_CAPACITY:]
        return updated, tuple(int(item) for item in updated_positions)

    def _incremental_special_block(
        self,
        residual: torch.Tensor,
        block_index: int,
        cache: Optional[LayerKVCache],
        recurrent_bank: Optional[FullRecurrentBank],
        permutation: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        local_capacity: int,
        return_diagnostics: bool,
        diagnostic_attention_weights: bool,
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
        if cache is None:
            local_keys, local_values = current_key, current_value
        else:
            local_keys = torch.cat((cache.key, current_key), dim=2)
            local_values = torch.cat((cache.value, current_value), dim=2)
        if local_keys.size(2) > local_capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess local KV")
        local_pre = F.scaled_dot_product_attention(
            query, local_keys, local_values, is_causal=False
        )
        next_cache = self._append_cache(
            current_key, current_value, cache, local_capacity
        )

        recurrent_weights = None
        if recurrent_bank is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        elif recurrent_bank.values.size(1) == 0:
            recurrent_pre = torch.zeros_like(local_pre)
            recurrent_weights = local_pre.new_empty((batch, heads, 1, 0))
            if block_index == B1_INDEX:
                coefficient = self._gate_coefficient(local_pre, gate_override)
            elif block_index == B2_INDEX:
                coefficient = self._gate_coefficient_b2(local_pre, gate_override)
            elif block_index == B3_INDEX:
                coefficient = self._gate_coefficient_b3(local_pre, gate_override)
            elif block_index == B4_INDEX:
                coefficient = self._gate_coefficient_b4(local_pre, gate_override)
            else:
                raise ValueError("special recurrent block must be B1, B2, B3, or B4")
        else:
            bank_values = recurrent_bank.values
            if permutation is not None:
                bank_values = bank_values[permutation]
            if block_index == B1_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv(bank_values)
                coefficient = self._gate_coefficient(local_pre, gate_override)
            elif block_index == B2_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv_b2(bank_values)
                coefficient = self._gate_coefficient_b2(local_pre, gate_override)
            elif block_index == B3_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv_b3(bank_values)
                coefficient = self._gate_coefficient_b3(local_pre, gate_override)
            elif block_index == B4_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv_b4(bank_values)
                coefficient = self._gate_coefficient_b4(local_pre, gate_override)
            else:
                raise ValueError("special recurrent block must be B1, B2, B3, or B4")
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                recurrent_key,
                recurrent_value,
                recurrent_bank.valid_mask,
                return_weights=(return_diagnostics and diagnostic_attention_weights),
            )
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

    def _incremental_ordinary_block(
        self,
        residual: torch.Tensor,
        block_index: int,
        cache: Optional[LayerKVCache],
        capacity: int,
    ):
        """Run one ordinary cached causal block for a single current token."""

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

    def incremental_step(
        self,
        token: torch.Tensor,
        state: FourLinkIncrementalState,
        control: str = "all_real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b4_gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = True,
    ):
        self._validate_incremental_state(state)
        mode = self._validate_bank_mode(bank_mode)
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown incremental control: {control}")
        wants_full_b4 = control == "b4_full_counterfactual"
        if state.b4_full_cache != wants_full_b4:
            raise ValueError("incremental state B4 cache geometry does not match control")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context is exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")

        shuffled_b1 = control == "all_shuffled"
        shuffled_b2 = control == "all_shuffled"
        shuffled_b3 = control == "all_shuffled"
        shuffled_b4 = control in {"all_shuffled", "b4_shuffled"}
        if shuffled_b1 or shuffled_b2 or shuffled_b3 or shuffled_b4:
            permutation = self._validate_permutation(
                recurrent_permutation, state.batch_size, token.device
            )
        elif recurrent_permutation is not None:
            raise ValueError("permutation is valid only for shuffled controls")
        else:
            permutation = None

        position = torch.tensor([state.position], dtype=torch.long, device=token.device)
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(position)

        b1_bank = self._incremental_bank_from_ring(
            state.h12_ring,
            state.h12_positions,
            state.position,
            mode,
            minimum_lag=2,
        )
        residual, next_b1_cache, b1_diag = self._incremental_special_block(
            residual,
            B1_INDEX,
            state.caches[B1_INDEX],
            b1_bank,
            permutation if shuffled_b1 else None,
            b1_gate_override,
            B1_LOCAL_WINDOW - 1,
            return_diagnostics,
            diagnostic_attention_weights,
        )

        b2_bank = self._incremental_bank_from_ring(
            state.h11_ring,
            state.h11_positions,
            state.position,
            mode,
            minimum_lag=B2_RECURRENT_MIN_LAG,
        )
        residual, next_b2_cache, b2_diag = self._incremental_special_block(
            residual,
            B2_INDEX,
            state.caches[B2_INDEX],
            b2_bank,
            permutation if shuffled_b2 else None,
            b2_gate_override,
            B2_LOCAL_WINDOW - 1,
            return_diagnostics,
            diagnostic_attention_weights,
        )

        b3_bank = self._incremental_bank_from_ring(
            state.h10_ring,
            state.h10_positions,
            state.position,
            mode,
            minimum_lag=B3_RECURRENT_MIN_LAG,
        )
        residual, next_b3_cache, b3_diag = self._incremental_special_block(
            residual,
            B3_INDEX,
            state.caches[B3_INDEX],
            b3_bank,
            permutation if shuffled_b3 else None,
            b3_gate_override,
            B3_LOCAL_WINDOW - 1,
            return_diagnostics,
            diagnostic_attention_weights,
        )

        b4_diag = None
        if wants_full_b4:
            residual, next_b4_cache = self._incremental_ordinary_block(
                residual,
                B4_INDEX,
                state.caches[B4_INDEX],
                int(self.config.block_size) - 1,
            )
        else:
            b4_bank = self._incremental_bank_from_ring(
                state.h9_ring,
                state.h9_positions,
                state.position,
                mode,
                minimum_lag=B4_RECURRENT_MIN_LAG,
            )
            b4_off = control == "b4_off"
            residual, next_b4_cache, b4_diag = self._incremental_special_block(
                residual,
                B4_INDEX,
                state.caches[B4_INDEX],
                None if b4_off else b4_bank,
                permutation if shuffled_b4 else None,
                0.0 if b4_off else b4_gate_override,
                B4_LOCAL_WINDOW - 1,
                return_diagnostics,
                diagnostic_attention_weights,
            )

        updated_caches = [next_b1_cache, next_b2_cache, next_b3_cache, next_b4_cache]
        upper_capacity = int(self.config.block_size) - 1
        h9 = None
        h10 = None
        h11 = None
        for block_index in range(4, TOTAL_LAYERS):
            residual, next_cache = self._incremental_ordinary_block(
                residual,
                block_index,
                state.caches[block_index],
                upper_capacity,
            )
            updated_caches.append(next_cache)
            if block_index == B9_INDEX:
                h9 = residual
            if block_index == B10_INDEX:
                h10 = residual
            if block_index == B11_INDEX:
                h11 = residual
        if h9 is None or h10 is None or h11 is None:
            raise RuntimeError("incremental B9/B10/B11 capture was not reached")
        h12 = residual
        top = self.base.transformer.ln_f(h12)
        logits = self.base.lm_head(top)

        next_h9, next_h9_positions = self._append_ring(
            state.h9_ring, state.h9_positions, h9, state.position
        )
        next_h10, next_h10_positions = self._append_ring(
            state.h10_ring, state.h10_positions, h10, state.position
        )
        next_h11, next_h11_positions = self._append_ring(
            state.h11_ring, state.h11_positions, h11, state.position
        )
        next_h12, next_h12_positions = self._append_ring(
            state.h12_ring, state.h12_positions, h12, state.position
        )
        next_state = FourLinkIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h9_ring=next_h9,
            h9_positions=next_h9_positions,
            h10_ring=next_h10,
            h10_positions=next_h10_positions,
            h11_ring=next_h11,
            h11_positions=next_h11_positions,
            h12_ring=next_h12,
            h12_positions=next_h12_positions,
            b4_full_cache=state.b4_full_cache,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics:
            return logits, next_state
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "b1": b1_diag,
            "b2": b2_diag,
            "b3": b3_diag,
            "b4": b4_diag,
            "cache_audit": self.incremental_cache_audit(next_state),
        }

    def incremental_cache_audit(self, state: FourLinkIncrementalState) -> dict:
        self._validate_incremental_state(state)
        lengths = self.incremental_cache_lengths(state)

        def storage_row(tensor: torch.Tensor) -> dict:
            expected_bytes = tensor.numel() * tensor.element_size()
            actual_bytes = tensor.untyped_storage().nbytes()
            exact = (
                tensor.storage_offset() == 0
                and tensor.is_contiguous()
                and actual_bytes == expected_bytes
            )
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "element_size_bytes": tensor.element_size(),
                "storage_offset": tensor.storage_offset(),
                "contiguous": tensor.is_contiguous(),
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "exact": bool(exact),
            }

        cache_storage = []
        for block_index, cache in enumerate(state.caches):
            cache_storage.append(
                {
                    "block": block_index + 1,
                    "key": None if cache is None else storage_row(cache.key),
                    "value": None if cache is None else storage_row(cache.value),
                }
            )
        h9_storage = storage_row(state.h9_ring)
        h10_storage = storage_row(state.h10_ring)
        h11_storage = storage_row(state.h11_ring)
        h12_storage = storage_row(state.h12_ring)
        physical = (
            h9_storage["exact"]
            and h10_storage["exact"]
            and h11_storage["exact"]
            and h12_storage["exact"]
            and all(
            row["key"] is None
            or (row["key"]["exact"] and row["value"]["exact"])
            for row in cache_storage
        )
        )
        upper_limit = int(self.config.block_size) - 1
        b4_limit = (
            upper_limit if state.b4_full_cache else B4_LOCAL_WINDOW - 1
        )
        passed = (
            lengths[0] <= 1
            and lengths[1] <= B2_LOCAL_WINDOW - 1
            and lengths[2] <= B3_LOCAL_WINDOW - 1
            and lengths[3] <= b4_limit
            and all(length <= upper_limit for length in lengths[4:])
            and state.h9_ring.size(1) <= RECURRENT_RING_CAPACITY
            and state.h10_ring.size(1) <= RECURRENT_RING_CAPACITY
            and state.h11_ring.size(1) <= RECURRENT_RING_CAPACITY
            and state.h12_ring.size(1) <= RECURRENT_RING_CAPACITY
            and physical
        )
        return {
            "position": state.position,
            "cache_lengths": lengths,
            "b1_historical_kv": lengths[0],
            "b1_historical_kv_limit": 1,
            "b2_historical_kv": lengths[1],
            "b2_historical_kv_limit": B2_LOCAL_WINDOW - 1,
            "b3_historical_kv": lengths[2],
            "b3_historical_kv_limit": B3_LOCAL_WINDOW - 1,
            "b4_historical_kv": lengths[3],
            "b4_historical_kv_limit": b4_limit,
            "b5_b12_historical_kv": lengths[4:],
            "b5_b12_historical_kv_limit": upper_limit,
            "h9_ring_length": int(state.h9_ring.size(1)),
            "h9_ring_limit": RECURRENT_RING_CAPACITY,
            "h9_ring_positions": state.h9_positions,
            "h10_ring_length": int(state.h10_ring.size(1)),
            "h10_ring_limit": RECURRENT_RING_CAPACITY,
            "h10_ring_positions": state.h10_positions,
            "h11_ring_length": int(state.h11_ring.size(1)),
            "h11_ring_limit": RECURRENT_RING_CAPACITY,
            "h11_ring_positions": state.h11_positions,
            "h12_ring_length": int(state.h12_ring.size(1)),
            "h12_ring_limit": RECURRENT_RING_CAPACITY,
            "h12_ring_positions": state.h12_positions,
            "b4_full_counterfactual": state.b4_full_cache,
            "physical_storage_exact": bool(physical),
            "cache_physical_storage": cache_storage,
            "h9_ring_physical_storage": h9_storage,
            "h10_ring_physical_storage": h10_storage,
            "h11_ring_physical_storage": h11_storage,
            "h12_ring_physical_storage": h12_storage,
            "passed": bool(passed),
        }

    def incremental_logits(
        self,
        tokens: torch.Tensor,
        control: str = "all_real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b4_gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = True,
    ) -> dict:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown incremental control: {control}")
        state = self.init_incremental_state(
            tokens.size(0),
            device=tokens.device,
            dtype=self.base.transformer.wte.weight.dtype,
            b4_full_cache=control == "b4_full_counterfactual",
        )
        logits = []
        diagnostics = []
        maxima = [0] * int(self.config.n_layer)
        maximum_h9 = 0
        maximum_h10 = 0
        maximum_h11 = 0
        maximum_h12 = 0
        for position in range(tokens.size(1)):
            result = self.incremental_step(
                tokens[:, position],
                state,
                control=control,
                recurrent_permutation=recurrent_permutation,
                b1_gate_override=b1_gate_override,
                b2_gate_override=b2_gate_override,
                b3_gate_override=b3_gate_override,
                b4_gate_override=b4_gate_override,
                return_diagnostics=return_diagnostics,
                bank_mode=bank_mode,
                diagnostic_attention_weights=diagnostic_attention_weights,
            )
            if return_diagnostics:
                current_logits, state, current_diagnostics = result
                diagnostics.append(current_diagnostics)
            else:
                current_logits, state = result
            logits.append(current_logits)
            lengths = self.incremental_cache_lengths(state)
            maxima = [max(old, new) for old, new in zip(maxima, lengths)]
            maximum_h9 = max(maximum_h9, int(state.h9_ring.size(1)))
            maximum_h10 = max(maximum_h10, int(state.h10_ring.size(1)))
            maximum_h11 = max(maximum_h11, int(state.h11_ring.size(1)))
            maximum_h12 = max(maximum_h12, int(state.h12_ring.size(1)))
        return {
            "logits": torch.cat(logits, dim=1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "max_h9_ring_length": maximum_h9,
            "max_h10_ring_length": maximum_h10,
            "max_h11_ring_length": maximum_h11,
            "max_h12_ring_length": maximum_h12,
            "cache_audit": self.incremental_cache_audit(state),
        }


RecurrentKVGPT = FourLinkRecurrentKVGPT
Experiment2D2IModel = FourLinkRecurrentKVGPT


__all__ = [
    "BANK_MODES",
    "B1_LOCAL_WINDOW",
    "B1_MAX_RECURRENT_ENTRIES",
    "B2_LOCAL_WINDOW",
    "B2_RECURRENT_MIN_LAG",
    "B2_MAX_RECURRENT_ENTRIES",
    "B3_LOCAL_WINDOW",
    "B3_RECURRENT_MIN_LAG",
    "B3_MAX_RECURRENT_ENTRIES",
    "B4_LOCAL_WINDOW",
    "B4_RECURRENT_MIN_LAG",
    "B4_MAX_RECURRENT_ENTRIES",
    "LOCAL_WINDOW",
    "MAX_RECURRENT_ENTRIES",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "INCREMENTAL_CONTROLS",
    "FourLinkIncrementalState",
    "FourLinkRecurrentKVGPT",
    "RecurrentKVGPT",
    "Experiment2D2IModel",
]
