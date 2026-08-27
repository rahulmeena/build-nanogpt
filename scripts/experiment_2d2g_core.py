"""Architecture kernels for Experiment 2D2G.

Stage A is the frozen 2D2B architecture without modification.  Stage B keeps
B1 W2 plus B12 recurrence and an ordinary full-context B2, then replaces B3
with W64 local attention plus a B10 recurrent bank covering lags 64..1023.
There is deliberately no B11 gate, recurrent path, or raw-state ring.
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
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
)


TOTAL_LAYERS = 12
B1_INDEX = 0
B2_INDEX = 1
B3_INDEX = 2
B10_INDEX = 9
B12_INDEX = 11
B2_LOCAL_WINDOW = 1024
B3_LOCAL_WINDOW = 64
B3_RECURRENT_MIN_LAG = 64
B3_MAX_RECURRENT_ENTRIES = RECURRENT_MAX_LAG - B3_RECURRENT_MIN_LAG + 1

INCREMENTAL_CONTROLS = (
    "real",
    "b3_off",
    "b3_shuffled",
    "b3_full_counterfactual",
)


StageARecurrentKVGPT = FullB12ToB1RecurrentKVGPT


@dataclass(frozen=True)
class StageBIncrementalState:
    """Exact deployment state; notably contains no B11 recurrent ring."""

    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h10_ring: torch.Tensor
    h10_positions: Tuple[int, ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]
    b3_full_cache: bool = False


class StageBRecurrentKVGPT(FullB12ToB1RecurrentKVGPT):
    """2D2G-B: full B2 and one fresh B10-to-B3 scalar gate."""

    def __init__(self, base: nn.Module):
        super().__init__(base)
        self.g_rec_b3 = nn.Parameter(torch.zeros(()))
        self._b3_local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @property
    def g_rec_b1(self) -> torch.Tensor:
        return self.g_rec

    @property
    def recurrent_scale_b1(self) -> torch.Tensor:
        return self.g_rec.tanh()

    @property
    def recurrent_scale_b3(self) -> torch.Tensor:
        return self.g_rec_b3.tanh()

    def b3_local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._b3_local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (source >= query - (B3_LOCAL_WINDOW - 1))
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
        FullB12ToB1RecurrentKVGPT._validate_bank_mode(bank_mode)
        query = torch.arange(
            query_offset, query_offset + query_length, device=device
        ).view(query_length, 1)
        source = torch.arange(
            source_offset, source_offset + source_length, device=device
        ).view(1, source_length)
        lag = query - source
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
        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("B10 recurrent source must have shape [batch,time,channel]")
        _, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd) or length < 1:
            raise ValueError("invalid B10 recurrent source")
        mode = self._active_bank_mode if bank_mode is None else bank_mode
        mode = self._validate_bank_mode(mode)
        valid = self.b3_recurrent_mask(length, length, recurrent_source.device, mode)
        positions = torch.arange(
            length, device=recurrent_source.device, dtype=torch.long
        ).view(1, length)
        return FullRecurrentBank(
            values=recurrent_source,
            valid_mask=valid,
            positions=positions.expand(length, length),
        )

    def project_recurrent_kv_b3(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 3:
            raise ValueError("B10 recurrent values must have shape [batch,source,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels:
            raise ValueError("B10 recurrent values have wrong channel width")
        block = self.base.transformer.h[B3_INDEX]
        _, key, value = block.attn.c_attn(block.ln_1(bank_values)).split(
            channels, dim=-1
        )
        batch, source_length, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, source_length, heads, head_size).transpose(1, 2)
        value = value.view(batch, source_length, heads, head_size).transpose(1, 2)
        return key, value

    @staticmethod
    def _diagnostic_attention_weights(
        query: torch.Tensor, key: torch.Tensor, valid_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        elements = query.size(0) * query.size(1) * query.size(2) * key.size(2)
        if elements > ATTENTION_WEIGHT_ELEMENT_LIMIT:
            return None
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores.mul_(query.size(-1) ** -0.5)
        mask = valid_mask.view(1, 1, query.size(2), key.size(2))
        row_valid = mask.any(dim=-1, keepdim=True)
        safe = torch.where(
            row_valid, scores.masked_fill(~mask, -torch.inf), torch.zeros_like(scores)
        )
        return F.softmax(safe, dim=-1) * mask.to(safe.dtype)

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

    def _parallel_block3(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ):
        block = self.base.transformer.h[B3_INDEX]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        query, local_key, local_value = block.attn.c_attn(block.ln_1(residual)).split(
            channels, dim=-1
        )
        query = query.view(batch, length, heads, head_size).transpose(1, 2)
        local_key = local_key.view(batch, length, heads, head_size).transpose(1, 2)
        local_value = local_value.view(batch, length, heads, head_size).transpose(1, 2)
        local_mask = self.b3_local_mask(length, residual.device)
        local_pre = F.scaled_dot_product_attention(
            query, local_key, local_value, attn_mask=local_mask, is_causal=False
        )
        local_weights = (
            self._diagnostic_attention_weights(query, local_key, local_mask)
            if return_diagnostics
            else None
        )
        recurrent_bank = None
        recurrent_weights = None
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
        after_attention = residual + block.attn.c_proj(combined)
        output = after_attention + block.mlp(block.ln_2(after_attention))
        if not return_diagnostics:
            return output, None
        return output, {
            "recurrent_attention_weights": recurrent_weights,
            "local_attention_weights": local_weights,
            "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
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

    @staticmethod
    def _source(
        value: Optional[torch.Tensor],
        permutation: Optional[torch.Tensor],
        tokens: torch.Tensor,
        channels: int,
        validator,
    ) -> Optional[torch.Tensor]:
        if value is None:
            if permutation is not None:
                raise ValueError("permutation requires recurrent source")
            return None
        if tuple(value.shape) != (tokens.size(0), tokens.size(1), channels):
            raise ValueError("recurrent source has wrong shape")
        if value.device != tokens.device:
            raise ValueError("recurrent source and tokens must share device")
        if permutation is None:
            return value
        return value[validator(permutation, tokens.size(0), tokens.device)]

    def forward_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        b1_recurrent_source: Optional[torch.Tensor] = None,
        b3_recurrent_source: Optional[torch.Tensor] = None,
        b1_recurrent_permutation: Optional[torch.Tensor] = None,
        b3_recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b3_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("targets must match tokens")
        mode = self._validate_bank_mode(bank_mode)
        channels = int(self.config.n_embd)
        b1_source = self._source(
            b1_recurrent_source,
            b1_recurrent_permutation,
            tokens,
            channels,
            self._validate_permutation,
        )
        b3_source = self._source(
            b3_recurrent_source,
            b3_recurrent_permutation,
            tokens,
            channels,
            self._validate_permutation,
        )
        positions = torch.arange(tokens.size(1), device=tokens.device)
        residual = self.base.transformer.wte(tokens) + self.base.transformer.wpe(positions)
        use_checkpoint = bool(
            activation_checkpointing and self.training and torch.is_grad_enabled()
        )
        previous_mode = self._active_bank_mode
        self._active_bank_mode = mode
        try:
            b1_diag = None
            if use_checkpoint and not return_diagnostics:
                if b1_source is None:
                    residual = checkpoint(
                        lambda x: self._parallel_block1(x, None, b1_gate_override, False)[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda x, memory: self._parallel_block1(
                            x, memory, b1_gate_override, False
                        )[0],
                        residual,
                        b1_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b1_diag = self._parallel_block1(
                    residual, b1_source, b1_gate_override, return_diagnostics
                )

            block2 = self.base.transformer.h[B2_INDEX]
            residual = (
                checkpoint(
                    block2,
                    residual,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if use_checkpoint
                else block2(residual)
            )

            b3_diag = None
            block3 = self.base.transformer.h[B3_INDEX]
            if b3_full_counterfactual:
                residual = (
                    checkpoint(
                        block3,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if use_checkpoint
                    else block3(residual)
                )
            elif use_checkpoint and not return_diagnostics:
                if b3_source is None:
                    residual = checkpoint(
                        lambda x: self._parallel_block3(x, None, b3_gate_override, False)[0],
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = checkpoint(
                        lambda x, memory: self._parallel_block3(
                            x, memory, b3_gate_override, False
                        )[0],
                        residual,
                        b3_source,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                residual, b3_diag = self._parallel_block3(
                    residual, b3_source, b3_gate_override, return_diagnostics
                )

            h10 = None
            for block_index in range(3, TOTAL_LAYERS):
                block = self.base.transformer.h[block_index]
                residual = (
                    checkpoint(
                        block,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if use_checkpoint
                    else block(residual)
                )
                if block_index == B10_INDEX:
                    h10 = residual
            if h10 is None:
                raise RuntimeError("B10 source capture was not reached")
            h12 = residual
            top = self.base.transformer.ln_f(h12)
            logits = self.base.lm_head(top)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return {
                "h10": h10,
                "raw_h10": h10,
                "h12": h12,
                "raw_h12": h12,
                "top": top,
                "logits": logits,
                "loss": loss,
                "diagnostics": (
                    {"b1": b1_diag, "b3": b3_diag, "b3_full_counterfactual": bool(b3_full_counterfactual)}
                    if return_diagnostics
                    else None
                ),
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
        b3_recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        b3_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        if int(num_passes) not in (2, 3):
            raise ValueError("2D2G permits exactly two or three passes")
        rows = [
            self.forward_pass(
                tokens,
                targets=targets,
                b3_full_counterfactual=b3_full_counterfactual,
                activation_checkpointing=activation_checkpointing,
                return_diagnostics=return_diagnostics,
                bank_mode=bank_mode,
            )
        ]
        for _ in range(1, int(num_passes)):
            previous = rows[-1]
            rows.append(
                self.forward_pass(
                    tokens,
                    targets=targets,
                    b1_recurrent_source=previous["h12"],
                    b3_recurrent_source=previous["h10"],
                    b1_recurrent_permutation=b1_recurrent_permutation,
                    b3_recurrent_permutation=b3_recurrent_permutation,
                    b1_gate_override=b1_gate_override,
                    b3_gate_override=b3_gate_override,
                    b3_full_counterfactual=b3_full_counterfactual,
                    activation_checkpointing=activation_checkpointing,
                    return_diagnostics=return_diagnostics,
                    bank_mode=bank_mode,
                )
            )
        weights = (0.25, 0.75) if int(num_passes) == 2 else (0.20, 0.40, 0.40)
        loss = None
        if targets is not None:
            loss = sum(weight * row["loss"] for weight, row in zip(weights, rows))
        final = rows[-1]
        return {
            "passes": tuple(rows),
            "pass_weights": weights,
            "pass_losses": tuple(row["loss"] for row in rows),
            "loss": loss,
            "h10": final["h10"],
            "h12": final["h12"],
            "top": final["top"],
            "logits": final["logits"],
            "diagnostics": tuple(row["diagnostics"] for row in rows),
        }

    def init_incremental_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        b3_full_cache: bool = False,
    ) -> StageBIncrementalState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        empty = torch.empty((batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype)
        return StageBIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * int(self.config.n_layer),
            h10_ring=empty,
            h10_positions=(),
            h12_ring=empty.clone(),
            h12_positions=(),
            b3_full_cache=bool(b3_full_cache),
        )

    @staticmethod
    def incremental_cache_lengths(state: StageBIncrementalState) -> Tuple[int, ...]:
        if not isinstance(state, StageBIncrementalState):
            raise TypeError("state must be StageBIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

    def _validate_incremental_state(self, state: StageBIncrementalState) -> None:
        if not isinstance(state, StageBIncrementalState):
            raise TypeError("incremental_step requires StageBIncrementalState")
        if not 0 <= state.position <= int(self.config.block_size):
            raise ValueError("incremental position outside context")
        upper = min(state.position, int(self.config.block_size) - 1)
        b3 = upper if state.b3_full_cache else min(state.position, B3_LOCAL_WINDOW - 1)
        expected = (min(state.position, 1), upper, b3) + (upper,) * 9
        if self.incremental_cache_lengths(state) != expected:
            raise ValueError("incremental cache geometry mismatch")
        positions = tuple(range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position))
        for name, ring, observed in (
            ("h10", state.h10_ring, state.h10_positions),
            ("h12", state.h12_ring, state.h12_positions),
        ):
            if observed != positions:
                raise ValueError(f"{name} ring positions mismatch")
            if tuple(ring.shape) != (state.batch_size, len(positions), int(self.config.n_embd)):
                raise ValueError(f"{name} ring shape mismatch")

    def _bank_from_ring(
        self,
        ring: torch.Tensor,
        positions: Tuple[int, ...],
        position: int,
        minimum_lag: int,
        bank_mode: str,
    ) -> FullRecurrentBank:
        mode = self._validate_bank_mode(bank_mode)
        absolute = torch.tensor(positions, device=ring.device, dtype=torch.long)
        if absolute.numel() == 0:
            selected = absolute
        else:
            lag = int(position) - absolute
            if mode == "full":
                valid = (lag >= minimum_lag) & (lag <= RECURRENT_MAX_LAG)
            elif mode == "two_slot":
                valid = (lag >= minimum_lag) & (lag < minimum_lag + 2)
            elif mode == "recent_only":
                valid = (lag >= 2) & (lag <= 31) if minimum_lag == 2 else torch.zeros_like(lag, dtype=torch.bool)
            else:
                valid = (lag >= max(minimum_lag, 32)) & (lag <= RECURRENT_MAX_LAG)
            selected = absolute[valid]
        values = ring[:, :0] if selected.numel() == 0 else ring.index_select(1, selected - positions[0])
        return FullRecurrentBank(
            values=values,
            valid_mask=torch.ones((1, selected.numel()), device=ring.device, dtype=torch.bool),
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
            updated = updated[:, -RECURRENT_RING_CAPACITY:].detach().clone(memory_format=torch.contiguous_format)
            updated_positions = updated_positions[-RECURRENT_RING_CAPACITY:]
        return updated, tuple(int(item) for item in updated_positions)

    def _incremental_ordinary_block(
        self,
        residual: torch.Tensor,
        block_index: int,
        cache: Optional[LayerKVCache],
        capacity: int,
    ):
        block = self.base.transformer.h[block_index]
        batch, _, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        query, key, value = block.attn.c_attn(block.ln_1(residual)).split(channels, -1)
        query = query.view(batch, 1, heads, head_size).transpose(1, 2)
        key = key.view(batch, 1, heads, head_size).transpose(1, 2)
        value = value.view(batch, 1, heads, head_size).transpose(1, 2)
        keys = key if cache is None else torch.cat((cache.key, key), 2)
        values = value if cache is None else torch.cat((cache.value, value), 2)
        if keys.size(2) > capacity + 1:
            raise RuntimeError(f"B{block_index + 1} materialized excess KV")
        attention = F.scaled_dot_product_attention(query, keys, values, is_causal=False)
        attention = attention.transpose(1, 2).contiguous().view(batch, 1, channels)
        after_attention = residual + block.attn.c_proj(attention)
        output = after_attention + block.mlp(block.ln_2(after_attention))
        return output, self._append_cache(key, value, cache, capacity)

    def _incremental_recurrent_block(
        self,
        residual: torch.Tensor,
        block_index: int,
        cache: Optional[LayerKVCache],
        bank: FullRecurrentBank,
        permutation: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        local_capacity: int,
        return_diagnostics: bool,
    ):
        block = self.base.transformer.h[block_index]
        batch, _, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        query, key, value = block.attn.c_attn(block.ln_1(residual)).split(channels, -1)
        query = query.view(batch, 1, heads, head_size).transpose(1, 2)
        key = key.view(batch, 1, heads, head_size).transpose(1, 2)
        value = value.view(batch, 1, heads, head_size).transpose(1, 2)
        local_keys = key if cache is None else torch.cat((cache.key, key), 2)
        local_values = value if cache is None else torch.cat((cache.value, value), 2)
        local_pre = F.scaled_dot_product_attention(query, local_keys, local_values, is_causal=False)
        next_cache = self._append_cache(key, value, cache, local_capacity)
        if bank.values.size(1) == 0:
            recurrent_pre = torch.zeros_like(local_pre)
            weights = local_pre.new_empty((batch, heads, 1, 0))
        else:
            bank_values = bank.values if permutation is None else bank.values[permutation]
            if block_index == B1_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv(bank_values)
            elif block_index == B3_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv_b3(bank_values)
            else:
                raise ValueError("recurrent block must be B1 or B3")
            recurrent_pre, weights = self._masked_recurrent_attention(
                query, recurrent_key, recurrent_value, bank.valid_mask, return_weights=return_diagnostics
            )
        coefficient = (
            self._gate_coefficient(local_pre, gate_override)
            if block_index == B1_INDEX
            else self._gate_coefficient_b3(local_pre, gate_override)
        )
        combined = (local_pre + coefficient * recurrent_pre).transpose(1, 2).contiguous().view(batch, 1, channels)
        after_attention = residual + block.attn.c_proj(combined)
        output = after_attention + block.mlp(block.ln_2(after_attention))
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "recurrent_attention_weights": weights,
                "recurrent_output_rms": recurrent_pre.float().square().mean().sqrt(),
                "gate_coefficient": coefficient,
                "recurrent_positions": bank.positions,
            }
        return output, next_cache, diagnostics

    def incremental_step(
        self,
        token: torch.Tensor,
        state: StageBIncrementalState,
        control: str = "real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = False,
    ):
        del diagnostic_attention_weights
        self._validate_incremental_state(state)
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown control {control}")
        if state.b3_full_cache != (control == "b3_full_counterfactual"):
            raise ValueError("B3 cache geometry does not match control")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("token must have shape [batch,1]")
        if control == "b3_shuffled":
            permutation = self._validate_permutation(recurrent_permutation, state.batch_size, token.device)
        elif recurrent_permutation is not None:
            raise ValueError("permutation is only valid for shuffled control")
        else:
            permutation = None
        position_tensor = torch.tensor([state.position], device=token.device)
        residual = self.base.transformer.wte(token) + self.base.transformer.wpe(position_tensor)
        b1_bank = self._bank_from_ring(state.h12_ring, state.h12_positions, state.position, 2, bank_mode)
        residual, b1_cache, b1_diag = self._incremental_recurrent_block(
            residual, B1_INDEX, state.caches[B1_INDEX], b1_bank, None,
            b1_gate_override, B1_LOCAL_WINDOW - 1, return_diagnostics
        )
        upper = int(self.config.block_size) - 1
        residual, b2_cache = self._incremental_ordinary_block(residual, B2_INDEX, state.caches[B2_INDEX], upper)
        b3_diag = None
        if control == "b3_full_counterfactual":
            residual, b3_cache = self._incremental_ordinary_block(residual, B3_INDEX, state.caches[B3_INDEX], upper)
        else:
            b3_bank = self._bank_from_ring(
                state.h10_ring, state.h10_positions, state.position,
                B3_RECURRENT_MIN_LAG, bank_mode
            )
            override = 0.0 if control == "b3_off" else b3_gate_override
            residual, b3_cache, b3_diag = self._incremental_recurrent_block(
                residual, B3_INDEX, state.caches[B3_INDEX], b3_bank,
                permutation, override, B3_LOCAL_WINDOW - 1, return_diagnostics
            )
        caches = [b1_cache, b2_cache, b3_cache]
        h10 = None
        for block_index in range(3, TOTAL_LAYERS):
            residual, cache = self._incremental_ordinary_block(
                residual, block_index, state.caches[block_index], upper
            )
            caches.append(cache)
            if block_index == B10_INDEX:
                h10 = residual
        if h10 is None:
            raise RuntimeError("B10 capture missing")
        h12 = residual
        logits = self.base.lm_head(self.base.transformer.ln_f(h12))
        h10_ring, h10_positions = self._append_ring(state.h10_ring, state.h10_positions, h10, state.position)
        h12_ring, h12_positions = self._append_ring(state.h12_ring, state.h12_positions, h12, state.position)
        next_state = StageBIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(caches),
            h10_ring=h10_ring,
            h10_positions=h10_positions,
            h12_ring=h12_ring,
            h12_positions=h12_positions,
            b3_full_cache=state.b3_full_cache,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics:
            return logits, next_state
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "b1": b1_diag,
            "b3": b3_diag,
            "cache_audit": self.incremental_cache_audit(next_state),
        }

    def incremental_cache_audit(self, state: StageBIncrementalState) -> dict:
        self._validate_incremental_state(state)
        lengths = self.incremental_cache_lengths(state)

        def storage(tensor: torch.Tensor) -> dict:
            expected = tensor.numel() * tensor.element_size()
            actual = tensor.untyped_storage().nbytes()
            return {
                "shape": tuple(tensor.shape),
                "expected_bytes": expected,
                "actual_bytes": actual,
                "exact": bool(tensor.storage_offset() == 0 and tensor.is_contiguous() and actual == expected),
            }

        cache_storage = []
        for block, cache in enumerate(state.caches, 1):
            cache_storage.append({
                "block": block,
                "key": None if cache is None else storage(cache.key),
                "value": None if cache is None else storage(cache.value),
            })
        h10 = storage(state.h10_ring)
        h12 = storage(state.h12_ring)
        physical = h10["exact"] and h12["exact"] and all(
            row["key"] is None or (row["key"]["exact"] and row["value"]["exact"])
            for row in cache_storage
        )
        upper = int(self.config.block_size) - 1
        b3_limit = upper if state.b3_full_cache else B3_LOCAL_WINDOW - 1
        passed = (
            lengths[0] <= 1
            and lengths[1] <= upper
            and lengths[2] <= b3_limit
            and all(length <= upper for length in lengths[3:])
            and state.h10_ring.size(1) <= RECURRENT_RING_CAPACITY
            and state.h12_ring.size(1) <= RECURRENT_RING_CAPACITY
            and physical
        )
        return {
            "position": state.position,
            "cache_lengths": lengths,
            "b1_historical_kv": lengths[0],
            "b1_historical_kv_limit": 1,
            "b2_historical_kv": lengths[1],
            "b2_historical_kv_limit": upper,
            "b3_historical_kv": lengths[2],
            "b3_historical_kv_limit": b3_limit,
            "b4_b12_historical_kv": lengths[3:],
            "h10_ring_length": int(state.h10_ring.size(1)),
            "h12_ring_length": int(state.h12_ring.size(1)),
            "h10_ring_positions": state.h10_positions,
            "h12_ring_positions": state.h12_positions,
            "has_b11_ring": False,
            "physical_storage_exact": bool(physical),
            "cache_physical_storage": cache_storage,
            "h10_ring_physical_storage": h10,
            "h12_ring_physical_storage": h12,
            "passed": bool(passed),
        }

    def incremental_logits(
        self,
        tokens: torch.Tensor,
        control: str = "real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b3_gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        state = self.init_incremental_state(
            tokens.size(0), device=tokens.device,
            dtype=self.base.transformer.wte.weight.dtype,
            b3_full_cache=control == "b3_full_counterfactual",
        )
        logits = []
        diagnostics = []
        maxima = [0] * TOTAL_LAYERS
        max_h10 = max_h12 = 0
        for position in range(tokens.size(1)):
            result = self.incremental_step(
                tokens[:, position], state, control=control,
                recurrent_permutation=recurrent_permutation,
                b1_gate_override=b1_gate_override,
                b3_gate_override=b3_gate_override,
                return_diagnostics=return_diagnostics,
                bank_mode=bank_mode,
            )
            if return_diagnostics:
                current, state, row = result
                diagnostics.append(row)
            else:
                current, state = result
            logits.append(current)
            maxima = [max(a, b) for a, b in zip(maxima, self.incremental_cache_lengths(state))]
            max_h10 = max(max_h10, int(state.h10_ring.size(1)))
            max_h12 = max(max_h12, int(state.h12_ring.size(1)))
        return {
            "logits": torch.cat(logits, 1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "max_h10_ring_length": max_h10,
            "max_h12_ring_length": max_h12,
            "cache_audit": self.incremental_cache_audit(state),
        }


RecurrentKVGPT = StageBRecurrentKVGPT
Experiment2D2GModel = StageBRecurrentKVGPT

__all__ = [
    "BANK_MODES",
    "B1_LOCAL_WINDOW",
    "B2_LOCAL_WINDOW",
    "B3_LOCAL_WINDOW",
    "B3_RECURRENT_MIN_LAG",
    "B3_MAX_RECURRENT_ENTRIES",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "INCREMENTAL_CONTROLS",
    "StageARecurrentKVGPT",
    "StageBIncrementalState",
    "StageBRecurrentKVGPT",
    "RecurrentKVGPT",
    "Experiment2D2GModel",
]
