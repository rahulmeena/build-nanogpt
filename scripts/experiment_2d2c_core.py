"""Architecture kernel for Experiment 2D2C.

2D2C preserves the finalized full-bank B12->B1 path from 2D2B and adds
exactly one mirrored path: full-bank B11->B2 recurrent K/V.  Blocks 1 and 2
retain only W2 ordinary same-layer context, while Blocks 3-12 remain W1024.
Both recurrent paths reuse the destination block's existing LayerNorm and
fused K/V projection slices, use separate recurrent softmaxes, and apply the
ordinary destination ``c_proj`` exactly once.

The parallel path stores one ``[B,T,C]`` source tensor per recurrent link and
never constructs ``[B,T,T,C]`` state copies.  The incremental path stores at
most 1023 raw B11 states, 1023 raw B12 states, one historical B1 K/V entry,
one historical B2 K/V entry, and ordinary upper-layer caches.
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
    LOCAL_WINDOW,
    MAX_RECURRENT_ENTRIES,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
)


TOTAL_LAYERS = 12
B1_INDEX = 0
B2_INDEX = 1
B11_INDEX = 10
B12_INDEX = 11

INCREMENTAL_CONTROLS = (
    "both_real",
    "b2_recurrence_off",
    "b2_shuffled",
    "b2_full_counterfactual",
    "both_shuffled",
    "b1_off_b2_real",
    "b1_real_b2_off",
)


@dataclass(frozen=True)
class MirroredIncrementalState:
    """Deployment state for the two mirrored recurrent links."""

    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h11_ring: torch.Tensor
    h11_positions: Tuple[int, ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]
    b2_full_cache: bool = False


class MirroredFullRecurrentKVGPT(FullB12ToB1RecurrentKVGPT):
    """Final 2D2B model plus one B11->B2 gate and no other parameters."""

    def __init__(self, base: nn.Module):
        super().__init__(base)
        # The sole new learnable tensor versus 2D2B.
        self.g_rec_b2 = nn.Parameter(torch.zeros(()))
        self._b2_local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

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

    def b2_local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the exact B2 W2 causal mask."""

        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._b2_local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (source >= query - (LOCAL_WINDOW - 1))
            self._b2_local_mask_cache[key] = mask
        return mask

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
        local_pre = F.scaled_dot_product_attention(
            query,
            local_key,
            local_value,
            attn_mask=self.b2_local_mask(length, residual.device),
            is_causal=False,
        )

        recurrent_weights = None
        recurrent_bank = None
        if recurrent_source is None:
            recurrent_pre = torch.zeros_like(local_pre)
            coefficient = local_pre.new_zeros(())
        else:
            recurrent_bank = self.build_recurrent_bank(recurrent_source)
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
            "attention_weights_omitted_for_memory": bool(
                recurrent_source is not None and recurrent_weights is None
            ),
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec_b2,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
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
        b1_recurrent_permutation: Optional[torch.Tensor] = None,
        b2_recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b2_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        """Run one parallel pass and expose exact post-MLP B11/B12 streams."""

        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        if targets is not None and tuple(targets.shape) != tuple(tokens.shape):
            raise ValueError("targets must match tokens")
        mode = self._validate_bank_mode(bank_mode)
        channels = int(self.config.n_embd)
        self._validate_source("B12 source", b1_recurrent_source, tokens, channels)
        self._validate_source("B11 source", b2_recurrent_source, tokens, channels)

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
            block2 = self.base.transformer.h[B2_INDEX]
            if b2_full_counterfactual:
                if use_checkpoint:
                    residual = checkpoint(
                        block2,
                        residual,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    residual = block2(residual)
            elif use_checkpoint and not return_diagnostics:
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

            h11 = None
            for block_index in range(2, TOTAL_LAYERS):
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
                if block_index == B11_INDEX:
                    h11 = residual
            if h11 is None:
                raise RuntimeError("B11 source capture was not reached")
            h12 = residual
            top = self.base.transformer.ln_f(h12)
            logits = self.base.lm_head(top)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            return {
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
                    "b2_full_counterfactual": bool(b2_full_counterfactual),
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
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        b2_full_counterfactual: bool = False,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
    ) -> dict:
        """Run attached two/three-pass recurrence with unchanged CE weighting."""

        if int(num_passes) not in (2, 3):
            raise ValueError("2D2C multi-pass execution permits exactly two or three passes")
        results = [
            self.forward_pass(
                tokens,
                targets=targets,
                b2_full_counterfactual=b2_full_counterfactual,
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
                    b1_recurrent_permutation=b1_recurrent_permutation,
                    b2_recurrent_permutation=b2_recurrent_permutation,
                    b1_gate_override=b1_gate_override,
                    b2_gate_override=b2_gate_override,
                    b2_full_counterfactual=b2_full_counterfactual,
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
        b2_full_cache: bool = False,
    ) -> MirroredIncrementalState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        empty = torch.empty(
            (batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype
        )
        return MirroredIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * int(self.config.n_layer),
            h11_ring=empty,
            h11_positions=(),
            h12_ring=empty.clone(),
            h12_positions=(),
            b2_full_cache=bool(b2_full_cache),
        )

    @staticmethod
    def incremental_cache_lengths(
        state: MirroredIncrementalState,
    ) -> Tuple[int, ...]:
        if not isinstance(state, MirroredIncrementalState):
            raise TypeError("state must be MirroredIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

    def _validate_incremental_state(self, state: MirroredIncrementalState) -> None:
        if not isinstance(state, MirroredIncrementalState):
            raise TypeError("incremental_step requires MirroredIncrementalState")
        if not 0 <= int(state.position) <= int(self.config.block_size):
            raise ValueError("incremental position is outside the model context")
        if len(state.caches) != int(self.config.n_layer):
            raise ValueError("incremental state has the wrong number of caches")
        upper = min(state.position, int(self.config.block_size) - 1)
        b2 = upper if state.b2_full_cache else min(state.position, 1)
        expected = (min(state.position, 1), b2) + (upper,) * 10
        lengths = self.incremental_cache_lengths(state)
        if lengths != expected:
            raise ValueError(f"incremental cache lengths {lengths} != {expected}")
        expected_positions = tuple(
            range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position)
        )
        for name, ring, positions in (
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
    ) -> FullRecurrentBank:
        mode = self._validate_bank_mode(bank_mode)
        absolute = torch.tensor(positions, device=ring.device, dtype=torch.long)
        if absolute.numel() == 0:
            selected = absolute
        else:
            lag = int(state_position) - absolute
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
            coefficient = (
                self._gate_coefficient(local_pre, gate_override)
                if block_index == B1_INDEX
                else self._gate_coefficient_b2(local_pre, gate_override)
            )
        else:
            bank_values = recurrent_bank.values
            if permutation is not None:
                bank_values = bank_values[permutation]
            if block_index == B1_INDEX:
                recurrent_key, recurrent_value = self.project_recurrent_kv(bank_values)
                coefficient = self._gate_coefficient(local_pre, gate_override)
            else:
                recurrent_key, recurrent_value = self.project_recurrent_kv_b2(bank_values)
                coefficient = self._gate_coefficient_b2(local_pre, gate_override)
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

    def incremental_step(
        self,
        token: torch.Tensor,
        state: MirroredIncrementalState,
        control: str = "both_real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        bank_mode: str = "full",
        diagnostic_attention_weights: bool = True,
    ):
        self._validate_incremental_state(state)
        mode = self._validate_bank_mode(bank_mode)
        if control not in INCREMENTAL_CONTROLS:
            raise ValueError(f"unknown incremental control: {control}")
        wants_full_b2 = control == "b2_full_counterfactual"
        if state.b2_full_cache != wants_full_b2:
            raise ValueError("incremental state B2 cache geometry does not match control")
        if state.position >= int(self.config.block_size):
            raise ValueError("incremental context is exhausted")
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if tuple(token.shape) != (state.batch_size, 1):
            raise ValueError("incremental token must have shape [batch] or [batch,1]")
        shuffled_b1 = control == "both_shuffled"
        shuffled_b2 = control in {"b2_shuffled", "both_shuffled"}
        if shuffled_b1 or shuffled_b2:
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
            state.h12_ring, state.h12_positions, state.position, mode
        )
        b1_off = control == "b1_off_b2_real"
        residual, next_b1_cache, b1_diag = self._incremental_special_block(
            residual,
            B1_INDEX,
            state.caches[B1_INDEX],
            None if b1_off else b1_bank,
            permutation if shuffled_b1 else None,
            0.0 if b1_off else b1_gate_override,
            LOCAL_WINDOW - 1,
            return_diagnostics,
            diagnostic_attention_weights,
        )

        updated_caches = [next_b1_cache]
        if wants_full_b2:
            block2 = self.base.transformer.h[B2_INDEX]
            channels = int(self.config.n_embd)
            heads = int(self.config.n_head)
            head_size = channels // heads
            normalized = block2.ln_1(residual)
            query, key, value = block2.attn.c_attn(normalized).split(channels, dim=-1)
            query = query.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            key = key.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            value = value.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            cache = state.caches[B2_INDEX]
            keys = key if cache is None else torch.cat((cache.key, key), dim=2)
            values = value if cache is None else torch.cat((cache.value, value), dim=2)
            attention = F.scaled_dot_product_attention(query, keys, values, is_causal=False)
            attention = attention.transpose(1, 2).contiguous().view(
                state.batch_size, 1, channels
            )
            residual = residual + block2.attn.c_proj(attention)
            residual = residual + block2.mlp(block2.ln_2(residual))
            next_b2_cache = self._append_cache(
                key, value, cache, int(self.config.block_size) - 1
            )
            b2_diag = None
        else:
            b2_bank = self._incremental_bank_from_ring(
                state.h11_ring, state.h11_positions, state.position, mode
            )
            b2_off = control in {"b2_recurrence_off", "b1_real_b2_off"}
            residual, next_b2_cache, b2_diag = self._incremental_special_block(
                residual,
                B2_INDEX,
                state.caches[B2_INDEX],
                None if b2_off else b2_bank,
                permutation if shuffled_b2 else None,
                0.0 if b2_off else b2_gate_override,
                LOCAL_WINDOW - 1,
                return_diagnostics,
                diagnostic_attention_weights,
            )
        updated_caches.append(next_b2_cache)

        channels = int(self.config.n_embd)
        heads = int(self.config.n_head)
        head_size = channels // heads
        upper_capacity = int(self.config.block_size) - 1
        h11 = None
        for block_index in range(2, TOTAL_LAYERS):
            block = self.base.transformer.h[block_index]
            normalized = block.ln_1(residual)
            query, key, value = block.attn.c_attn(normalized).split(channels, dim=-1)
            query = query.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            key = key.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            value = value.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            cache = state.caches[block_index]
            keys = key if cache is None else torch.cat((cache.key, key), dim=2)
            values = value if cache is None else torch.cat((cache.value, value), dim=2)
            if keys.size(2) > int(self.config.block_size):
                raise RuntimeError(f"B{block_index + 1} materialized excess KV")
            attention = F.scaled_dot_product_attention(query, keys, values, is_causal=False)
            attention = attention.transpose(1, 2).contiguous().view(
                state.batch_size, 1, channels
            )
            residual = residual + block.attn.c_proj(attention)
            residual = residual + block.mlp(block.ln_2(residual))
            updated_caches.append(
                self._append_cache(key, value, cache, upper_capacity)
            )
            if block_index == B11_INDEX:
                h11 = residual
        if h11 is None:
            raise RuntimeError("incremental B11 capture was not reached")
        h12 = residual
        top = self.base.transformer.ln_f(h12)
        logits = self.base.lm_head(top)

        next_h11, next_h11_positions = self._append_ring(
            state.h11_ring, state.h11_positions, h11, state.position
        )
        next_h12, next_h12_positions = self._append_ring(
            state.h12_ring, state.h12_positions, h12, state.position
        )
        next_state = MirroredIncrementalState(
            position=state.position + 1,
            batch_size=state.batch_size,
            caches=tuple(updated_caches),
            h11_ring=next_h11,
            h11_positions=next_h11_positions,
            h12_ring=next_h12,
            h12_positions=next_h12_positions,
            b2_full_cache=state.b2_full_cache,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics:
            return logits, next_state
        return logits, next_state, {
            "position": state.position,
            "control": control,
            "b1": b1_diag,
            "b2": b2_diag,
            "cache_audit": self.incremental_cache_audit(next_state),
        }

    def incremental_cache_audit(self, state: MirroredIncrementalState) -> dict:
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
        h11_storage = storage_row(state.h11_ring)
        h12_storage = storage_row(state.h12_ring)
        physical = h11_storage["exact"] and h12_storage["exact"] and all(
            row["key"] is None
            or (row["key"]["exact"] and row["value"]["exact"])
            for row in cache_storage
        )
        upper_limit = int(self.config.block_size) - 1
        b2_limit = upper_limit if state.b2_full_cache else 1
        passed = (
            lengths[0] <= 1
            and lengths[1] <= b2_limit
            and all(length <= upper_limit for length in lengths[2:])
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
            "b2_historical_kv_limit": b2_limit,
            "b3_b12_historical_kv": lengths[2:],
            "b3_b12_historical_kv_limit": upper_limit,
            "h11_ring_length": int(state.h11_ring.size(1)),
            "h11_ring_limit": RECURRENT_RING_CAPACITY,
            "h11_ring_positions": state.h11_positions,
            "h12_ring_length": int(state.h12_ring.size(1)),
            "h12_ring_limit": RECURRENT_RING_CAPACITY,
            "h12_ring_positions": state.h12_positions,
            "b2_full_counterfactual": state.b2_full_cache,
            "physical_storage_exact": bool(physical),
            "cache_physical_storage": cache_storage,
            "h11_ring_physical_storage": h11_storage,
            "h12_ring_physical_storage": h12_storage,
            "passed": bool(passed),
        }

    def incremental_logits(
        self,
        tokens: torch.Tensor,
        control: str = "both_real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        b1_gate_override: Optional[torch.Tensor] = None,
        b2_gate_override: Optional[torch.Tensor] = None,
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
            b2_full_cache=control == "b2_full_counterfactual",
        )
        logits = []
        diagnostics = []
        maxima = [0] * int(self.config.n_layer)
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
            maximum_h11 = max(maximum_h11, int(state.h11_ring.size(1)))
            maximum_h12 = max(maximum_h12, int(state.h12_ring.size(1)))
        return {
            "logits": torch.cat(logits, dim=1),
            "state": state,
            "diagnostics": tuple(diagnostics) if return_diagnostics else None,
            "max_cache_lengths": tuple(maxima),
            "max_h11_ring_length": maximum_h11,
            "max_h12_ring_length": maximum_h12,
            "cache_audit": self.incremental_cache_audit(state),
        }


RecurrentKVGPT = MirroredFullRecurrentKVGPT
Experiment2D2CModel = MirroredFullRecurrentKVGPT


__all__ = [
    "BANK_MODES",
    "LOCAL_WINDOW",
    "MAX_RECURRENT_ENTRIES",
    "RECURRENT_MAX_LAG",
    "RECURRENT_RING_CAPACITY",
    "INCREMENTAL_CONTROLS",
    "MirroredIncrementalState",
    "MirroredFullRecurrentKVGPT",
    "RecurrentKVGPT",
    "Experiment2D2CModel",
]
