"""Architecture kernel for Experiment 2D2B.

This module changes exactly one scientific property of the finalized 2D2A
kernel: the recurrent B12 bank is widened from ``[t-3, t-2]`` to every
eligible non-overlapping position ``[max(0, t-1023), ..., t-2]``.  The
wrapper still owns only the already-trained scalar ``g_rec`` and borrows B1's
existing LayerNorm, fused K/V rows, and output projection.

The parallel implementation projects one ``[B,T,C]`` source tensor and uses a
banded causal mask.  It never constructs ``[B,T,T,C]`` recurrent-state
copies.  The incremental implementation retains at most 1023 raw B12 states,
exposes all but the newest state to the full bank, and never reruns old tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from experiment_2d2a_core import (
    B12ToB1RecurrentKVGPT,
    LayerKVCache,
    RecurrentKVIncrementalState,
)


LOCAL_WINDOW = 2
MAX_RECURRENT_ENTRIES = 1022
RECURRENT_MAX_LAG = 1023
RECURRENT_RING_CAPACITY = 1023
BANK_MODES = ("full", "two_slot", "recent_only", "old_only")
ATTENTION_WEIGHT_ELEMENT_LIMIT = 32_000_000


@dataclass(frozen=True)
class FullRecurrentBank:
    """Shared recurrent source plus a query/key eligibility mask.

    ``values`` is the original ``[B,S,C]`` source tensor, not a repeated
    query-indexed copy.  ``valid_mask`` and ``positions`` are ``[T,S]``.
    ``positions`` is an expanded zero-stride view of absolute source indices.
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    positions: torch.Tensor


class FullB12ToB1RecurrentKVGPT(B12ToB1RecurrentKVGPT):
    """Final 2D2A model with a full-width token-indexed recurrent bank."""

    def __init__(self, base):
        super().__init__(base)
        self._active_bank_mode = "full"

    @staticmethod
    def _validate_bank_mode(bank_mode: str) -> str:
        if bank_mode not in BANK_MODES:
            raise ValueError(f"unknown recurrent bank mode: {bank_mode}")
        return bank_mode

    @staticmethod
    def recurrent_mask(
        query_length: int,
        source_length: int,
        device: torch.device,
        bank_mode: str = "full",
        query_offset: int = 0,
        source_offset: int = 0,
    ) -> torch.Tensor:
        """Return the exact absolute-position recurrent eligibility mask."""
        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        queries = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        sources = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = queries - sources
        if bank_mode == "full":
            valid = (lag >= 2) & (lag <= RECURRENT_MAX_LAG)
        elif bank_mode == "two_slot":
            valid = (lag >= 2) & (lag <= 3)
        elif bank_mode == "recent_only":
            valid = (lag >= 2) & (lag <= 31)
        else:
            valid = (lag >= 32) & (lag <= RECURRENT_MAX_LAG)
        return valid

    def build_recurrent_bank(
        self, recurrent_source: torch.Tensor, bank_mode: Optional[str] = None
    ) -> FullRecurrentBank:
        """Build a memory-efficient full bank backed by one source tensor."""
        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("recurrent source must have shape [batch,time,channel]")
        _, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd):
            raise ValueError("recurrent source channel width does not match the model")
        if length < 1:
            raise ValueError("recurrent source time dimension must be nonempty")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        valid = self.recurrent_mask(length, length, recurrent_source.device, mode)
        source_positions = torch.arange(
            length, device=recurrent_source.device, dtype=torch.long
        ).view(1, length)
        positions = source_positions.expand(length, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=positions,
        )

    def project_recurrent_kv(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the unchanged B1 LN and fused K/V slices once per source."""
        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 3:
            raise ValueError("recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels:
            raise ValueError("recurrent values have the wrong channel dimension")
        block1 = self.base.transformer.h[0]
        normalized = block1.ln_1(bank_values)
        _, key, value = block1.attn.c_attn(normalized).split(channels, dim=-1)
        batch, source_length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, source_length, heads, head_size).transpose(1, 2)
        value = value.view(batch, source_length, heads, head_size).transpose(1, 2)
        return key, value

    @staticmethod
    def _masked_recurrent_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Separate full-bank softmax using SDPA without source-state copies."""
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("invalid recurrent attention tensor ranks")
        if key.shape != value.shape:
            raise ValueError("recurrent keys and values must have identical shapes")
        if query.shape[:2] != key.shape[:2] or query.size(-1) != key.size(-1):
            raise ValueError("recurrent query/K/V shapes do not align")
        expected_mask = (query.size(2), key.size(2))
        if tuple(valid_mask.shape) != expected_mask:
            raise ValueError(
                f"recurrent valid mask {tuple(valid_mask.shape)} != {expected_mask}"
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_mask,
            is_causal=False,
        )
        weights = None
        elements = query.size(0) * query.size(1) * query.size(2) * key.size(2)
        if return_weights and elements <= ATTENTION_WEIGHT_ELEMENT_LIMIT:
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
            scores.mul_(query.size(-1) ** -0.5)
            mask = valid_mask.view(1, 1, query.size(2), key.size(2))
            row_valid = mask.any(dim=-1, keepdim=True)
            masked = scores.masked_fill(~mask, -torch.inf)
            safe = torch.where(row_valid, masked, torch.zeros_like(masked))
            weights = F.softmax(safe, dim=-1) * mask.to(safe.dtype)
        return output, weights

    def _parallel_block1(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ):
        block1 = self.base.transformer.h[0]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block1.ln_1(residual)
        query, local_key, local_value = block1.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_pre = F.scaled_dot_product_attention(
            query,
            local_key,
            local_value,
            attn_mask=self.local_mask(length, residual.device),
            is_causal=False,
        )

        recurrent_weights = None
        recurrent_bank = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self.build_recurrent_bank(recurrent_source)
            recurrent_key, recurrent_value = self.project_recurrent_kv(
                recurrent_bank.values
            )
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query,
                recurrent_key,
                recurrent_value,
                recurrent_bank.valid_mask,
                return_weights=return_diagnostics,
            )
            coefficient = self._gate_coefficient(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        attention_output = block1.attn.c_proj(combined)
        after_attention = residual + attention_output
        output = after_attention + block1.mlp(block1.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return output, {
            "bank_mode": self._active_bank_mode,
            "recurrent_attention_weights": recurrent_weights,
            "attention_weights_omitted_for_memory": bool(
                recurrent_source is not None and recurrent_weights is None
            ),
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
        }

    def forward_pass(self, *args, bank_mode: str = "full", **kwargs):
        mode = self._validate_bank_mode(bank_mode)
        previous = self._active_bank_mode
        self._active_bank_mode = mode
        try:
            return super().forward_pass(*args, **kwargs)
        finally:
            self._active_bank_mode = previous

    def forward_multi_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_passes: int = 2,
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        if int(num_passes) not in (2, 3):
            raise ValueError("2D2B multi-pass execution permits exactly two or three passes")
        mode = self._validate_bank_mode(bank_mode)
        results = [
            self.forward_pass(
                tokens,
                targets=targets,
                activation_checkpointing=activation_checkpointing,
                return_diagnostics=return_diagnostics,
                bank_mode=mode,
            )
        ]
        for _ in range(1, int(num_passes)):
            results.append(
                self.forward_pass(
                    tokens,
                    targets=targets,
                    recurrent_source=results[-1]["h12"],
                    recurrent_permutation=recurrent_permutation,
                    gate_override=gate_override,
                    activation_checkpointing=activation_checkpointing,
                    return_diagnostics=return_diagnostics,
                    bank_mode=mode,
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
            "h12": final["h12"],
            "raw_h12": final["h12"],
            "top": final["top"],
            "logits": final["logits"],
            "diagnostics": tuple(result["diagnostics"] for result in results),
        }

    def _validate_incremental_state(self, state: RecurrentKVIncrementalState) -> None:
        if not isinstance(state, RecurrentKVIncrementalState):
            raise TypeError("incremental_step requires RecurrentKVIncrementalState")
        if not 0 <= int(state.position) <= int(self.config.block_size):
            raise ValueError("incremental position is outside the model context")
        if len(state.caches) != int(self.config.n_layer):
            raise ValueError("incremental state has the wrong number of layer caches")
        lengths = self.incremental_cache_lengths(state)
        expected = (min(state.position, 1),) + (
            min(state.position, int(self.config.block_size) - 1),
        ) * (int(self.config.n_layer) - 1)
        if lengths != expected:
            raise ValueError(f"incremental cache lengths {lengths} != {expected}")
        ring_start = max(0, state.position - RECURRENT_RING_CAPACITY)
        expected_positions = tuple(range(ring_start, state.position))
        if state.h12_positions != expected_positions:
            raise ValueError(
                f"incremental h12 positions {state.h12_positions} != {expected_positions}"
            )
        expected_ring = (
            state.batch_size,
            len(expected_positions),
            int(self.config.n_embd),
        )
        if tuple(state.h12_ring.shape) != expected_ring:
            raise ValueError("incremental h12 ring has the wrong shape")
        for block_index, cache in enumerate(state.caches):
            if cache is None:
                continue
            if cache.key.shape != cache.value.shape or cache.key.ndim != 4:
                raise ValueError(f"B{block_index + 1} cache K/V shapes do not match")
            if cache.key.size(0) != state.batch_size:
                raise ValueError(f"B{block_index + 1} cache batch size mismatch")

    def _incremental_recurrent_bank(
        self,
        residual: torch.Tensor,
        state: RecurrentKVIncrementalState,
        bank_mode: str,
    ) -> FullRecurrentBank:
        mode = self._validate_bank_mode(bank_mode)
        absolute = torch.tensor(
            state.h12_positions, device=residual.device, dtype=torch.long
        )
        if absolute.numel() == 0:
            selected = absolute
        else:
            lag = state.position - absolute
            if mode == "full":
                valid = (lag >= 2) & (lag <= RECURRENT_MAX_LAG)
            elif mode == "two_slot":
                valid = (lag >= 2) & (lag <= 3)
            elif mode == "recent_only":
                valid = (lag >= 2) & (lag <= 31)
            else:
                valid = (lag >= 32) & (lag <= RECURRENT_MAX_LAG)
            selected = absolute[valid]
        if selected.numel() == 0:
            values = state.h12_ring[:, :0]
        else:
            first = state.h12_positions[0]
            indices = selected - first
            values = state.h12_ring.index_select(1, indices)
        valid_mask = torch.ones(
            (1, selected.numel()), device=residual.device, dtype=torch.bool
        )
        return FullRecurrentBank(
            values=values,
            valid_mask=valid_mask,
            positions=selected.view(1, -1),
        )

    def incremental_step(
        self,
        token: torch.Tensor,
        state: RecurrentKVIncrementalState,
        control: str = "real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = True,
    ):
        self._validate_incremental_state(state)
        mode = self._validate_bank_mode(bank_mode)
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context is exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")
        if control not in {"real", "shuffled", "plain"}:
            raise ValueError(f"unknown incremental control: {control}")
        if control == "shuffled":
            permutation = self._validate_permutation(
                recurrent_permutation, state.batch_size, token.device
            )
        elif recurrent_permutation is not None:
            raise ValueError("a recurrent permutation is valid only for shuffled control")
        else:
            permutation = None

        position = torch.tensor([state.position], dtype=torch.long, device=token.device)
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(position)
        block1 = self.base.transformer.h[0]
        channels = int(self.config.n_embd)
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block1.ln_1(residual)
        query, current_key, current_value = block1.attn.c_attn(normalized).split(
            channels, dim=-1
        )
        query = query.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
        current_key = current_key.view(
            state.batch_size, 1, heads, head_size
        ).transpose(1, 2)
        current_value = current_value.view(
            state.batch_size, 1, heads, head_size
        ).transpose(1, 2)
        b1_cache = state.caches[0]
        if b1_cache is None:
            local_keys, local_values = current_key, current_value
        else:
            local_keys = torch.cat((b1_cache.key, current_key), dim=2)
            local_values = torch.cat((b1_cache.value, current_value), dim=2)
        if local_keys.size(2) > LOCAL_WINDOW:
            raise RuntimeError("B1 materialized more than W2 local entries")
        local_pre = F.scaled_dot_product_attention(
            query, local_keys, local_values, is_causal=False
        )
        next_b1_cache = self._append_cache(
            current_key, current_value, b1_cache, LOCAL_WINDOW - 1
        )

        recurrent_weights = None
        recurrent_bank = None
        if control == "plain":
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self._incremental_recurrent_bank(residual, state, mode)
            if recurrent_bank.values.size(1) == 0:
                recurrent_pre = torch.zeros_like(local_pre)
                recurrent_weights = local_pre.new_empty(
                    (state.batch_size, heads, 1, 0), dtype=torch.float32
                )
            else:
                bank_values = recurrent_bank.values
                if permutation is not None:
                    bank_values = bank_values[permutation]
                recurrent_key, recurrent_value = self.project_recurrent_kv(bank_values)
                recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                    query,
                    recurrent_key,
                    recurrent_value,
                    recurrent_bank.valid_mask,
                    return_weights=(
                        return_diagnostics and diagnostic_attention_weights
                    ),
                )
            coefficient = self._gate_coefficient(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(
            state.batch_size, 1, channels
        )
        residual = residual + block1.attn.c_proj(combined)
        residual = residual + block1.mlp(block1.ln_2(residual))
        updated_caches = [next_b1_cache]

        upper_capacity = int(self.config.block_size) - 1
        for block_index, block in enumerate(self.base.transformer.h[1:], start=1):
            normalized = block.ln_1(residual)
            query_upper, key_upper, value_upper = block.attn.c_attn(normalized).split(
                channels, dim=-1
            )
            query_upper = query_upper.view(
                state.batch_size, 1, heads, head_size
            ).transpose(1, 2)
            key_upper = key_upper.view(
                state.batch_size, 1, heads, head_size
            ).transpose(1, 2)
            value_upper = value_upper.view(
                state.batch_size, 1, heads, head_size
            ).transpose(1, 2)
            cache = state.caches[block_index]
            if cache is None:
                keys, values = key_upper, value_upper
            else:
                keys = torch.cat((cache.key, key_upper), dim=2)
                values = torch.cat((cache.value, value_upper), dim=2)
            if keys.size(2) > int(self.config.block_size):
                raise RuntimeError(f"B{block_index + 1} materialized excess KV history")
            attention = F.scaled_dot_product_attention(
                query_upper, keys, values, is_causal=False
            )
            attention = attention.transpose(1, 2).contiguous().view(
                state.batch_size, 1, channels
            )
            residual = residual + block.attn.c_proj(attention)
            residual = residual + block.mlp(block.ln_2(residual))
            updated_caches.append(
                self._append_cache(key_upper, value_upper, cache, upper_capacity)
            )

        h12 = residual
        top = self.base.transformer.ln_f(h12)
        logits = self.base.lm_head(top)
        next_ring = torch.cat((state.h12_ring, h12.detach()), dim=1)
        next_positions: Sequence[int] = (*state.h12_positions, state.position)
        if next_ring.size(1) > RECURRENT_RING_CAPACITY:
            next_ring = next_ring[:, -RECURRENT_RING_CAPACITY:].detach().clone(
                memory_format=torch.contiguous_format
            )
            next_positions = next_positions[-RECURRENT_RING_CAPACITY:]
        next_state = RecurrentKVIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h12_ring=next_ring,
            h12_positions=tuple(int(value) for value in next_positions),
        )
        self._validate_incremental_state(next_state)

        if not return_diagnostics:
            return logits, next_state
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            state.batch_size, 1, channels
        )
        diagnostics = {
            "position": state.position,
            "bank_mode": mode,
            "recurrent_attention_weights": recurrent_weights,
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "recurrent_positions": (
                None if recurrent_bank is None else recurrent_bank.positions
            ),
            "gate_coefficient": coefficient,
            "cache_audit": self.incremental_cache_audit(next_state),
        }
        return logits, next_state, diagnostics

    def incremental_cache_audit(self, state: RecurrentKVIncrementalState) -> dict:
        self._validate_incremental_state(state)
        lengths = self.incremental_cache_lengths(state)
        upper_limit = min(int(self.config.block_size) - 1, 1023)

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
        ring_storage = storage_row(state.h12_ring)
        physical_storage_exact = ring_storage["exact"] and all(
            row["key"] is None
            or (row["key"]["exact"] and row["value"]["exact"])
            for row in cache_storage
        )
        passed = (
            lengths[0] <= 1
            and all(length <= upper_limit for length in lengths[1:])
            and state.h12_ring.size(1) <= RECURRENT_RING_CAPACITY
            and physical_storage_exact
        )
        return {
            "position": state.position,
            "cache_lengths": lengths,
            "b1_historical_kv": lengths[0],
            "b1_historical_kv_limit": 1,
            "b2_b12_historical_kv": lengths[1:],
            "b2_b12_historical_kv_limit": upper_limit,
            "h12_ring_length": int(state.h12_ring.size(1)),
            "h12_ring_limit": RECURRENT_RING_CAPACITY,
            "h12_ring_positions": state.h12_positions,
            "physical_storage_exact": bool(physical_storage_exact),
            "cache_physical_storage": cache_storage,
            "h12_ring_physical_storage": ring_storage,
            "passed": bool(passed),
        }

    def incremental_logits(
        self,
        tokens: torch.Tensor,
        control: str = "real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = True,
    ) -> dict:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        mode = self._validate_bank_mode(bank_mode)
        state = self.init_incremental_state(
            tokens.size(0),
            device=tokens.device,
            dtype=self.base.transformer.wte.weight.dtype,
        )
        logits = []
        diagnostics = []
        maxima = [0] * int(self.config.n_layer)
        maximum_ring = 0
        for position in range(tokens.size(1)):
            result = self.incremental_step(
                tokens[:, position],
                state,
                control=control,
                recurrent_permutation=recurrent_permutation,
                gate_override=gate_override,
                return_diagnostics=return_diagnostics,
                bank_mode=mode,
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
            maximum_ring = max(maximum_ring, int(state.h12_ring.size(1)))
        return {
            "logits": torch.cat(logits, dim=1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "max_h12_ring_length": maximum_ring,
            "cache_audit": self.incremental_cache_audit(state),
        }


RecurrentKVGPT = FullB12ToB1RecurrentKVGPT
Experiment2D2BModel = FullB12ToB1RecurrentKVGPT


__all__ = [
    "LOCAL_WINDOW",
    "MAX_RECURRENT_ENTRIES",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "BANK_MODES",
    "FullRecurrentBank",
    "FullB12ToB1RecurrentKVGPT",
    "RecurrentKVGPT",
    "Experiment2D2BModel",
]
