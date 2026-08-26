"""Architecture kernel for Experiment 2D2A.

The wrapper in this module deliberately owns only one parameter, ``g_rec``.
Every projection and normalization used by the recurrent branch is borrowed
from Block 1 of the wrapped Standard GPT model.

The parallel path implements the training recurrence (a previous pass supplies
raw Block-12 states).  The incremental path implements deployment recurrence
(the model supplies its own recent raw Block-12 states).  Both paths use:

* Block-1 local attention over ``[t-1, t]``;
* a separate recurrent softmax over ``[h12[t-3], h12[t-2]]``; and
* exactly one application of Block 1's existing attention ``c_proj``.

No experiment orchestration, checkpoint loading, or optimizer policy lives
here.  Keeping this file architecture-only makes the scientific driver able to
audit the single added scalar independently of training infrastructure.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


LOCAL_WINDOW = 2
RECURRENT_WINDOW = 2
RECURRENT_LAG = 2
RECURRENT_OFFSETS = (3, 2)
RECURRENT_RING_CAPACITY = 3


@dataclass(frozen=True)
class RecurrentBank:
    """A fixed two-slot bank plus the mask that makes early slots absent.

    ``values`` has shape ``[batch, time, 2, channel]``.  Invalid values are
    zeroed for diagnostics, but they are also masked before softmax and hence
    receive exactly zero probability.  ``positions`` and ``valid_mask`` have
    shape ``[time, 2]`` and use slots ``[t-3, t-2]`` in that order.
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class LayerKVCache:
    """A compact, already-projected historical K/V cache for one layer."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.key.size(2))


@dataclass(frozen=True)
class RecurrentKVIncrementalState:
    """Deployment state for token-by-token 2D2A evaluation."""

    position: int
    batch_size: int
    caches: Tuple[Optional[LayerKVCache], ...]
    h12_ring: torch.Tensor
    h12_positions: Tuple[int, ...]


class B12ToB1RecurrentKVGPT(nn.Module):
    """Wrap a 12-layer Standard GPT with one scalar recurrent K/V gate."""

    def __init__(self, base: nn.Module):
        super().__init__()
        if not isinstance(base, nn.Module) or not hasattr(base, "config"):
            raise TypeError("base must be a GPT module with a config")
        config = base.config
        if getattr(config, "residual_mode", None) != "standard":
            raise ValueError("Experiment 2D2A requires Standard GPT residuals")
        if int(getattr(config, "n_layer", -1)) != 12:
            raise ValueError("Experiment 2D2A requires exactly 12 Transformer blocks")
        if int(getattr(config, "block_size", 0)) < LOCAL_WINDOW:
            raise ValueError("Experiment 2D2A requires block_size >= 2")
        if int(config.block_size) > 1024:
            raise ValueError("Experiment 2D2A permits at most the frozen 1024-token context")
        if int(config.n_embd) % int(config.n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if len(base.transformer.h) != 12:
            raise ValueError("base.transformer.h must contain exactly 12 blocks")

        self.base = base
        # The sole architectural addition.  Do not add modules or learned
        # buffers to this wrapper without changing the experiment identity.
        self.g_rec = nn.Parameter(torch.zeros(()))
        self._local_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @property
    def config(self):
        return self.base.config

    @property
    def recurrent_scale(self) -> torch.Tensor:
        return self.g_rec.tanh()

    def local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the exact Block-1 W2 causal mask."""
        length = int(length)
        if length < 1:
            raise ValueError("attention length must be positive")
        key = (length, str(device))
        mask = self._local_mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(length, 1)
            source = torch.arange(length, device=device).view(1, length)
            mask = (source <= query) & (source >= query - (LOCAL_WINDOW - 1))
            self._local_mask_cache[key] = mask
        return mask

    def build_recurrent_bank(self, recurrent_source: torch.Tensor) -> RecurrentBank:
        """Build ``[h12[t-3], h12[t-2]]`` without wraparound."""
        if not isinstance(recurrent_source, torch.Tensor) or recurrent_source.ndim != 3:
            raise ValueError("recurrent source must have shape [batch,time,channel]")
        batch, length, channels = recurrent_source.shape
        if channels != int(self.config.n_embd):
            raise ValueError("recurrent source channel width does not match the model")
        if length < 1:
            raise ValueError("recurrent source time dimension must be nonempty")

        receivers = torch.arange(length, device=recurrent_source.device).view(length, 1)
        offsets = torch.tensor(RECURRENT_OFFSETS, device=recurrent_source.device).view(1, 2)
        positions = receivers - offsets
        valid = positions.ge(0)
        safe_positions = positions.clamp_min(0)
        # Advanced indexing keeps the temporal edge attached to recurrent_source.
        values = recurrent_source[:, safe_positions, :]
        values = values * valid.view(1, length, RECURRENT_WINDOW, 1).to(values.dtype)
        expected = (batch, length, RECURRENT_WINDOW, channels)
        if tuple(values.shape) != expected:
            raise RuntimeError(f"internal recurrent bank shape {tuple(values.shape)} != {expected}")
        return RecurrentBank(values=values, valid_mask=valid, positions=positions)

    def project_recurrent_kv(
        self, bank_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Block 1's existing LN and fused ``c_attn`` K/V slices."""
        if not isinstance(bank_values, torch.Tensor) or bank_values.ndim != 4:
            raise ValueError("recurrent bank values must have shape [batch,time,slot,channel]")
        channels = int(self.config.n_embd)
        if bank_values.size(-1) != channels or bank_values.size(2) != RECURRENT_WINDOW:
            raise ValueError("recurrent bank has the wrong slot or channel dimension")
        block1 = self.base.transformer.h[0]
        normalized = block1.ln_1(bank_values)
        # Use the exact existing fused B1 projection, then discard its query
        # slice.  This introduces no recurrent weights and makes the reuse
        # identity directly auditable without comparing differently shaped
        # GEMM reduction kernels.
        _, key, value = block1.attn.c_attn(normalized).split(channels, dim=-1)
        batch, length, slots, _ = key.shape
        heads = int(self.config.n_head)
        head_size = channels // heads
        key = key.view(batch, length, slots, heads, head_size).permute(0, 3, 1, 2, 4)
        value = value.view(batch, length, slots, heads, head_size).permute(0, 3, 1, 2, 4)
        return key, value

    @staticmethod
    def _masked_recurrent_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate recurrent softmax with exact all-invalid-row handling."""
        if query.ndim != 4 or key.ndim != 5 or value.ndim != 5:
            raise ValueError("invalid recurrent attention tensor ranks")
        if key.shape != value.shape:
            raise ValueError("recurrent keys and values must have identical shapes")
        if key.shape[:3] != query.shape[:3] or key.size(-1) != query.size(-1):
            raise ValueError("recurrent query/K/V shapes do not align")
        if tuple(valid_mask.shape) != (query.size(2), key.size(3)):
            raise ValueError("recurrent valid mask has the wrong shape")

        scale = query.size(-1) ** -0.5
        scores = torch.einsum("bhtd,bhtsd->bhts", query.float(), key.float()) * scale
        mask = valid_mask.view(1, 1, query.size(2), key.size(3))
        row_valid = mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        # Softmax of an all-masked row is undefined.  Replace only those rows
        # before softmax, then multiply by the mask so their final probability
        # is exactly zero in every slot.
        safe_scores = torch.where(row_valid, masked_scores, torch.zeros_like(masked_scores))
        weights = F.softmax(safe_scores, dim=-1) * mask.to(safe_scores.dtype)
        output = torch.einsum("bhts,bhtsd->bhtd", weights.to(value.dtype), value)
        return output, weights

    def _gate_coefficient(
        self, reference: torch.Tensor, gate_override: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_override is None:
            coefficient = self.recurrent_scale
        elif isinstance(gate_override, torch.Tensor):
            if gate_override.numel() != 1:
                raise ValueError("gate override must be scalar")
            coefficient = gate_override.reshape(())
        else:
            coefficient = reference.new_tensor(float(gate_override))
        return coefficient.to(device=reference.device, dtype=reference.dtype)

    def _parallel_block1(
        self,
        residual: torch.Tensor,
        recurrent_source: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
        return_diagnostics: bool,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        block1 = self.base.transformer.h[0]
        batch, length, channels = residual.shape
        heads = int(self.config.n_head)
        head_size = channels // heads

        normalized = block1.ln_1(residual)
        qkv = block1.attn.c_attn(normalized)
        query, local_key, local_value = qkv.split(channels, dim=-1)
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
            recurrent_key, recurrent_value = self.project_recurrent_kv(recurrent_bank.values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query, recurrent_key, recurrent_value, recurrent_bank.valid_mask
            )
            coefficient = self._gate_coefficient(local_pre, gate_override)

        combined = local_pre + coefficient * recurrent_pre
        combined = combined.transpose(1, 2).contiguous().view(batch, length, channels)
        # This is intentionally the only Block-1 c_proj call.
        attention_output = block1.attn.c_proj(combined)
        after_attention = residual + attention_output
        output = after_attention + block1.mlp(block1.ln_2(after_attention))

        if not return_diagnostics:
            return output, None
        recurrent_output = recurrent_pre.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        diagnostics = {
            "recurrent_attention_weights": recurrent_weights,
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "local_output_rms": local_pre.float().square().mean().sqrt(),
            "gate_raw": self.g_rec,
            "gate_coefficient": coefficient,
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "recurrent_positions": None if recurrent_bank is None else recurrent_bank.positions,
        }
        return output, diagnostics

    def _validate_parallel_inputs(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor],
        recurrent_source: Optional[torch.Tensor],
        recurrent_permutation: Optional[torch.Tensor],
        gate_override: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        batch, length = tokens.shape
        if not 1 <= length <= int(self.config.block_size):
            raise ValueError("token length exceeds learned position embeddings")
        if targets is not None and tuple(targets.shape) != (batch, length):
            raise ValueError("targets must have the same [batch,time] shape as tokens")
        if recurrent_source is None:
            if recurrent_permutation is not None:
                raise ValueError("a recurrent permutation requires a recurrent source")
            if gate_override is not None and float(torch.as_tensor(gate_override).detach()) != 0.0:
                raise ValueError("a nonzero gate override requires a recurrent source")
            return None
        expected = (batch, length, int(self.config.n_embd))
        if tuple(recurrent_source.shape) != expected:
            raise ValueError(f"recurrent source shape {tuple(recurrent_source.shape)} != {expected}")
        if recurrent_source.device != tokens.device:
            raise ValueError("tokens and recurrent source must be on the same device")
        if recurrent_permutation is None:
            return recurrent_source
        permutation = self._validate_permutation(recurrent_permutation, batch, tokens.device)
        return recurrent_source[permutation]

    @staticmethod
    def _validate_permutation(
        permutation: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(permutation, torch.Tensor) or tuple(permutation.shape) != (batch_size,):
            raise ValueError("recurrent permutation must have shape [batch]")
        permutation = permutation.to(device=device, dtype=torch.long)
        expected = torch.arange(batch_size, device=device)
        if batch_size < 2 or torch.any(permutation == expected):
            raise ValueError("shuffled recurrence requires a fixed-point-free permutation")
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("recurrent permutation must contain each batch row exactly once")
        return permutation

    def forward_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        recurrent_source: Optional[torch.Tensor] = None,
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
    ) -> dict:
        """Run one sequence-parallel pass and expose raw post-MLP ``h12``.

        ``gate_override`` is an effective recurrent coefficient (it is not
        passed through tanh).  Supplying zero is the plain-control override.
        """
        source = self._validate_parallel_inputs(
            tokens, targets, recurrent_source, recurrent_permutation, gate_override
        )
        _, length = tokens.shape
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        residual = self.base.transformer.wte(tokens) + self.base.transformer.wpe(positions)
        use_checkpoint = bool(
            activation_checkpointing and self.training and torch.is_grad_enabled()
        )

        diagnostics = None
        if use_checkpoint and not return_diagnostics:
            if source is None:
                residual = checkpoint(
                    lambda value: self._parallel_block1(value, None, gate_override, False)[0],
                    residual,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                residual = checkpoint(
                    lambda value, memory: self._parallel_block1(
                        value, memory, gate_override, False
                    )[0],
                    residual,
                    source,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
        else:
            residual, diagnostics = self._parallel_block1(
                residual, source, gate_override, return_diagnostics
            )

        for block in self.base.transformer.h[1:]:
            if use_checkpoint:
                residual = checkpoint(
                    block,
                    residual,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                residual = block(residual)

        # residual is the exact Block-12 post-MLP stream before final LN.
        h12 = residual
        top = self.base.transformer.ln_f(h12)
        logits = self.base.lm_head(top)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return {
            "h12": h12,
            "raw_h12": h12,
            "top": top,
            "logits": logits,
            "loss": loss,
            "diagnostics": diagnostics,
        }

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs):
        return self.forward_pass(tokens, targets=targets, **kwargs)

    def forward_multi_pass(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_passes: int = 2,
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        activation_checkpointing: bool = False,
        return_diagnostics: bool = False,
    ) -> dict:
        """Run attached two- or three-pass recurrence with CE-only weighting."""
        if int(num_passes) not in (2, 3):
            raise ValueError("2D2A multi-pass execution permits exactly two or three passes")
        results = [
            self.forward_pass(
                tokens,
                targets=targets,
                activation_checkpointing=activation_checkpointing,
                return_diagnostics=return_diagnostics,
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
                )
            )
        weights = (0.25, 0.75) if int(num_passes) == 2 else (0.20, 0.40, 0.40)
        weighted_loss = None
        if targets is not None:
            weighted_loss = sum(weight * result["loss"] for weight, result in zip(weights, results))
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

    def init_incremental_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> RecurrentKVIncrementalState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        reference = self.base.transformer.wte.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        ring = torch.empty(
            (batch_size, 0, int(self.config.n_embd)), device=device, dtype=dtype
        )
        return RecurrentKVIncrementalState(
            position=0,
            batch_size=batch_size,
            caches=(None,) * int(self.config.n_layer),
            h12_ring=ring,
            h12_positions=(),
        )

    @staticmethod
    def incremental_cache_lengths(
        state: RecurrentKVIncrementalState,
    ) -> Tuple[int, ...]:
        if not isinstance(state, RecurrentKVIncrementalState):
            raise TypeError("state must be RecurrentKVIncrementalState")
        return tuple(0 if cache is None else cache.length for cache in state.caches)

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
        expected_positions = tuple(
            range(max(0, state.position - RECURRENT_RING_CAPACITY), state.position)
        )
        if state.h12_positions != expected_positions:
            raise ValueError(
                f"incremental h12 ring positions {state.h12_positions} != {expected_positions}"
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

    @staticmethod
    def _append_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[LayerKVCache],
        capacity: int,
    ) -> LayerKVCache:
        if cache is None:
            keys = key
            values = value
        else:
            if cache.key.device != key.device or cache.value.device != value.device:
                raise ValueError("incremental cache device mismatch")
            if cache.key.shape[:2] != key.shape[:2] or cache.key.size(-1) != key.size(-1):
                raise ValueError("incremental cache geometry mismatch")
            keys = torch.cat((cache.key, key), dim=2)
            values = torch.cat((cache.value, value), dim=2)
        if keys.size(2) > capacity + 1:
            raise RuntimeError("hidden incremental history exceeded the physical window")
        # A slice alone would keep the just-evicted/full-history backing
        # storage alive.  Clone into an exact contiguous allocation so the
        # physical cache is genuinely bounded, not merely logically bounded.
        retained_key = keys[:, :, -capacity:].detach().clone(
            memory_format=torch.contiguous_format
        )
        retained_value = values[:, :, -capacity:].detach().clone(
            memory_format=torch.contiguous_format
        )
        return LayerKVCache(retained_key, retained_value)

    def _incremental_recurrent_bank(
        self, residual: torch.Tensor, state: RecurrentKVIncrementalState
    ) -> RecurrentBank:
        positions = torch.tensor(
            [state.position - RECURRENT_OFFSETS[0], state.position - RECURRENT_OFFSETS[1]],
            device=residual.device,
            dtype=torch.long,
        ).view(1, RECURRENT_WINDOW)
        lookup = {position: index for index, position in enumerate(state.h12_positions)}
        slots = []
        valid_values = []
        for position in positions.view(-1).tolist():
            index = lookup.get(int(position))
            if index is None:
                slots.append(torch.zeros_like(residual[:, 0]))
                valid_values.append(False)
            else:
                slots.append(state.h12_ring[:, index])
                valid_values.append(True)
        values = torch.stack(slots, dim=1).unsqueeze(1)
        valid = torch.tensor(valid_values, device=residual.device, dtype=torch.bool).view(1, 2)
        return RecurrentBank(values=values, valid_mask=valid, positions=positions)

    def incremental_step(
        self,
        token: torch.Tensor,
        state: RecurrentKVIncrementalState,
        control: str = "real",
        recurrent_permutation: Optional[torch.Tensor] = None,
        gate_override: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        """Consume one token using bounded deployment-equivalent state.

        Controls are ``real``, ``shuffled``, and ``plain``.  The returned
        logits have shape ``[batch, 1, vocab]``.
        """
        self._validate_incremental_state(state)
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
        qkv = block1.attn.c_attn(normalized)
        query, current_key, current_value = qkv.split(channels, dim=-1)
        query = query.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
        current_key = current_key.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
        current_value = current_value.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
        b1_cache = state.caches[0]
        if b1_cache is None:
            local_keys = current_key
            local_values = current_value
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
            recurrent_bank = self._incremental_recurrent_bank(residual, state)
            bank_values = recurrent_bank.values
            if permutation is not None:
                bank_values = bank_values[permutation]
            recurrent_key, recurrent_value = self.project_recurrent_kv(bank_values)
            recurrent_pre, recurrent_weights = self._masked_recurrent_attention(
                query, recurrent_key, recurrent_value, recurrent_bank.valid_mask
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
            qkv = block.attn.c_attn(normalized)
            query, current_key, current_value = qkv.split(channels, dim=-1)
            query = query.view(state.batch_size, 1, heads, head_size).transpose(1, 2)
            current_key = current_key.view(
                state.batch_size, 1, heads, head_size
            ).transpose(1, 2)
            current_value = current_value.view(
                state.batch_size, 1, heads, head_size
            ).transpose(1, 2)
            cache = state.caches[block_index]
            if cache is None:
                keys = current_key
                values = current_value
            else:
                keys = torch.cat((cache.key, current_key), dim=2)
                values = torch.cat((cache.value, current_value), dim=2)
            if keys.size(2) > int(self.config.block_size):
                raise RuntimeError(f"B{block_index + 1} materialized excess KV history")
            attention = F.scaled_dot_product_attention(
                query, keys, values, is_causal=False
            )
            attention = attention.transpose(1, 2).contiguous().view(
                state.batch_size, 1, channels
            )
            residual = residual + block.attn.c_proj(attention)
            residual = residual + block.mlp(block.ln_2(residual))
            updated_caches.append(
                self._append_cache(
                    current_key, current_value, cache, upper_capacity
                )
            )

        h12 = residual
        top = self.base.transformer.ln_f(h12)
        logits = self.base.lm_head(top)
        next_ring = torch.cat((state.h12_ring, h12.detach()), dim=1)
        next_positions: Sequence[int] = (*state.h12_positions, state.position)
        if next_ring.size(1) > RECURRENT_RING_CAPACITY:
            # As with K/V eviction, cloning is required: a narrow view would
            # retain the four-state concatenation's backing allocation.
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
            "recurrent_attention_weights": recurrent_weights,
            "recurrent_output_rms": recurrent_output.float().square().mean().sqrt(),
            "recurrent_valid_mask": (
                None if recurrent_bank is None else recurrent_bank.valid_mask
            ),
            "recurrent_positions": None if recurrent_bank is None else recurrent_bank.positions,
            "gate_coefficient": coefficient,
            "cache_audit": self.incremental_cache_audit(next_state),
        }
        return logits, next_state, diagnostics

    def incremental_cache_audit(self, state: RecurrentKVIncrementalState) -> dict:
        """Return explicit physical-cache and recurrent-ring bounds."""
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
            cache_storage.append({
                "block": block_index + 1,
                "key": None if cache is None else storage_row(cache.key),
                "value": None if cache is None else storage_row(cache.value),
            })
        ring_storage = storage_row(state.h12_ring)
        physical_storage_exact = (
            ring_storage["exact"]
            and all(
                row["key"] is None
                or (row["key"]["exact"] and row["value"]["exact"])
                for row in cache_storage
            )
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
    ) -> dict:
        """Teacher-force a complete sequence through the true incremental kernel."""
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,time]")
        state = self.init_incremental_state(
            tokens.size(0), device=tokens.device, dtype=self.base.transformer.wte.weight.dtype
        )
        logits = []
        diagnostics = []
        maxima = [0] * int(self.config.n_layer)
        maximum_ring = 0
        for position in range(tokens.size(1)):
            step_result = self.incremental_step(
                tokens[:, position],
                state,
                control=control,
                recurrent_permutation=recurrent_permutation,
                gate_override=gate_override,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                current_logits, state, current_diagnostics = step_result
                diagnostics.append(current_diagnostics)
            else:
                current_logits, state = step_result
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


# Short aliases for orchestration code and tests.
RecurrentKVGPT = B12ToB1RecurrentKVGPT
Experiment2D2AModel = B12ToB1RecurrentKVGPT


__all__ = [
    "LOCAL_WINDOW",
    "RECURRENT_WINDOW",
    "RECURRENT_LAG",
    "RECURRENT_OFFSETS",
    "RECURRENT_RING_CAPACITY",
    "RecurrentBank",
    "LayerKVCache",
    "RecurrentKVIncrementalState",
    "B12ToB1RecurrentKVGPT",
    "RecurrentKVGPT",
    "Experiment2D2AModel",
]
