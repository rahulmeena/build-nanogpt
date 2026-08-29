"""Experiment 2D4A write-time recurrent source-depth routing kernel.

The fixed arm is the sealed 2D3A kernel.  The routed arm changes only the
vector written into each of the four existing recurrent rings.  It keeps the
known-good source as the exact zero point and learns a gated mixture over
higher complete-block post-MLP residual states.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import experiment_2d3a_core as fixed


DESTINATION_BLOCKS = (0, 2, 4, 5)  # B1, B3, B5, B6
DESTINATION_NAMES = {0: "b1", 2: "b3", 4: "b5", 5: "b6"}
CANDIDATE_BLOCKS = {
    0: tuple(range(1, 12)),  # B2..B12
    2: tuple(range(3, 12)),  # B4..B12
    4: tuple(range(5, 12)),  # B6..B12
    5: tuple(range(6, 12)),  # B7..B12
}
BASELINE_BLOCKS = {0: 11, 2: 9, 4: 7, 5: 6}
EXPECTED_CANDIDATE_COUNTS = {0: 11, 2: 9, 4: 7, 5: 6}
ROUTER_PARAMETER_COUNT = 4 * (768 + 768 + 1)
ROUTED_MODEL_PARAMETERS = 124_482_056
ROUTE_MODES = ("learned", "uniform", "off")


class RouteRMSNorm(nn.Module):
    """Project-exact affine RMSNorm without importing the training CLI."""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_dtype = value.dtype
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class RecurrentSourceDepthRouter(nn.Module):
    """Single-query AttnRes-style router around one fixed source."""

    def __init__(self, n_embd: int, source_blocks: Sequence[int], baseline_block: int,
                 eps: float = 1e-5):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_embd))
        self.norm = RouteRMSNorm(n_embd, eps=eps)
        self.gate = nn.Parameter(torch.zeros(()))
        self.source_blocks = tuple(int(value) for value in source_blocks)
        self.baseline_block = int(baseline_block)
        if self.baseline_block not in self.source_blocks:
            raise ValueError("baseline source must be a router candidate")

    def _score(self, value: torch.Tensor) -> torch.Tensor:
        key = self.norm(value)
        # Frozen project convention: no 1/sqrt(d) scaling.
        return F.linear(key, self.query.unsqueeze(0)).squeeze(-1)

    def forward(self, values: Sequence[torch.Tensor], mode: str = "learned",
                return_diagnostics: bool = False):
        if mode not in ROUTE_MODES:
            raise ValueError(f"unknown route mode: {mode}")
        if len(values) != len(self.source_blocks):
            raise ValueError(
                f"expected {len(self.source_blocks)} sources, got {len(values)}"
            )
        if not values:
            raise ValueError("source-depth router requires candidates")
        shape = tuple(values[0].shape)
        if any(tuple(value.shape) != shape for value in values):
            raise ValueError("all source-depth candidates must share shape")

        if mode == "uniform":
            logits = torch.zeros(
                (len(values), *shape[:-1]), device=values[0].device,
                dtype=torch.float32,
            )
        else:
            use_checkpoint = self.training and torch.is_grad_enabled()
            scores = []
            for value in values:
                if use_checkpoint:
                    score = checkpoint(
                        self._score, value, use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    score = self._score(value)
                scores.append(score.float())
            logits = torch.stack(scores, dim=0)

        beta = F.softmax(logits, dim=0)
        mixture = torch.zeros_like(values[0])
        for weight, value in zip(beta.unbind(dim=0), values):
            contribution = weight.to(value.dtype).unsqueeze(-1) * value
            mixture = mixture + contribution.to(mixture.dtype)

        baseline_index = self.source_blocks.index(self.baseline_block)
        baseline = values[baseline_index]
        gamma = self.gate.tanh()
        if mode == "off":
            gamma = gamma.new_zeros(())
        gamma_value = gamma.to(device=baseline.device, dtype=baseline.dtype)
        memory = baseline + gamma_value * (mixture - baseline)

        if not return_diagnostics:
            return memory
        effective = beta * gamma.float()
        effective = effective.clone()
        effective[baseline_index] += 1.0 - gamma.float()
        safe = beta.clamp_min(torch.finfo(beta.dtype).tiny)
        entropy = -(beta * safe.log()).sum(dim=0)
        return memory, {
            "candidate_blocks": tuple(block + 1 for block in self.source_blocks),
            "baseline_block": self.baseline_block + 1,
            "baseline_index": baseline_index,
            "mode": mode,
            "raw_gate": self.gate,
            "gamma": gamma,
            "beta": beta,
            "effective_coefficients": effective,
            "entropy": entropy,
            "normalized_entropy": entropy / math.log(len(values)),
            "mixture": mixture,
            "baseline": baseline,
            "memory": memory,
            "candidates": tuple(values),
        }


class RoutedRecurrentPyramidGPT(fixed.AlternatingIntegrationRecurrentPyramidGPT):
    """2D3A with four gated write-time source-depth routers."""

    def __init__(self, base: nn.Module):
        super().__init__(base)
        channels = int(self.config.n_embd)
        eps = float(getattr(self.config, "attnres_rms_eps", 1e-5))
        self.source_routers = nn.ModuleDict({
            DESTINATION_NAMES[block]: RecurrentSourceDepthRouter(
                channels, CANDIDATE_BLOCKS[block], BASELINE_BLOCKS[block], eps=eps
            )
            for block in DESTINATION_BLOCKS
        })

    def source_router(self, block_index: int) -> RecurrentSourceDepthRouter:
        if block_index not in DESTINATION_NAMES:
            raise ValueError(f"B{block_index + 1} has no source router")
        return self.source_routers[DESTINATION_NAMES[block_index]]

    def route_memories(self, block_states: Dict[int, torch.Tensor],
                       mode: str = "learned",
                       off_destinations: Iterable[str] = (),
                       return_diagnostics: bool = False):
        if mode not in ROUTE_MODES:
            raise ValueError(f"unknown route mode: {mode}")
        off = {str(value).lower() for value in off_destinations}
        unknown = off - set(DESTINATION_NAMES.values())
        if unknown:
            raise ValueError(f"unknown route-off destinations: {sorted(unknown)}")
        memories = {}
        diagnostics = {}
        for block in DESTINATION_BLOCKS:
            name = DESTINATION_NAMES[block]
            candidates = [block_states[source] for source in CANDIDATE_BLOCKS[block]]
            current_mode = "off" if name in off else mode
            result = self.source_router(block)(
                candidates, mode=current_mode,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                memories[name], diagnostics[name] = result
            else:
                memories[name] = result
        return (memories, diagnostics) if return_diagnostics else memories

    def forward_pass(self, tokens, targets=None, route_mode="learned",
                     route_off_destinations=(), return_route_diagnostics=False,
                     **kwargs):
        if "capture_all_block_states" in kwargs:
            raise ValueError("capture_all_block_states is managed by 2D4A")
        result = super().forward_pass(
            tokens, targets=targets, capture_all_block_states=True, **kwargs
        )
        block_states = result.pop("all_block_states")
        memories, route_diagnostics = self.route_memories(
            block_states,
            mode=route_mode,
            off_destinations=route_off_destinations,
            return_diagnostics=True,
        )
        result.update({
            "m_b1": memories["b1"],
            "m_b3": memories["b3"],
            "m_b5": memories["b5"],
            "m_b6": memories["b6"],
            "route_diagnostics": (
                route_diagnostics if return_route_diagnostics else None
            ),
        })
        return result

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
        return_route_diagnostics=False,
        bank_mode="full",
        route_mode="learned",
        route_off_destinations=(),
    ):
        if int(num_passes) not in (2, 3):
            raise ValueError("2D4A permits exactly two or three training passes")
        shared = {
            "b1_recurrent_permutation": b1_recurrent_permutation,
            "b3_recurrent_permutation": b3_recurrent_permutation,
            "b5_recurrent_permutation": b5_recurrent_permutation,
            "b6_recurrent_permutation": b6_recurrent_permutation,
            "b1_gate_override": b1_gate_override,
            "b3_gate_override": b3_gate_override,
            "b5_gate_override": b5_gate_override,
            "b6_gate_override": b6_gate_override,
            "full_counterfactual_blocks": full_counterfactual_blocks,
            "activation_checkpointing": activation_checkpointing,
            "return_diagnostics": return_diagnostics,
            "return_route_diagnostics": return_route_diagnostics,
            "bank_mode": bank_mode,
            "route_mode": route_mode,
            "route_off_destinations": route_off_destinations,
        }
        first_shared = dict(shared)
        for key in tuple(first_shared):
            if key.endswith("_recurrent_permutation") or key.endswith("_gate_override"):
                first_shared.pop(key)
        results = [self.forward_pass(tokens, targets=targets, **first_shared)]
        for _ in range(1, int(num_passes)):
            previous = results[-1]
            results.append(self.forward_pass(
                tokens,
                targets=targets,
                b1_recurrent_source=previous["m_b1"],
                b3_recurrent_source=previous["m_b3"],
                b5_recurrent_source=previous["m_b5"],
                b6_recurrent_source=previous["m_b6"],
                **shared,
            ))
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
            **{
                name: final[name]
                for name in (
                    "h7", "h8", "h10", "h12", "m_b1", "m_b3", "m_b5",
                    "m_b6", "top", "logits",
                )
            },
            "diagnostics": tuple(row["diagnostics"] for row in results),
            "route_diagnostics": tuple(
                row["route_diagnostics"] for row in results
            ),
        }

    def incremental_step(
        self,
        token,
        state,
        control="all_real",
        recurrent_permutation=None,
        return_diagnostics=False,
        bank_mode="full",
        diagnostic_attention_weights=True,
        route_mode="learned",
        route_off_destinations=(),
    ):
        logits, provisional, diagnostics = super().incremental_step(
            token,
            state,
            control=control,
            recurrent_permutation=recurrent_permutation,
            return_diagnostics=return_diagnostics,
            bank_mode=bank_mode,
            diagnostic_attention_weights=diagnostic_attention_weights,
            return_block_states=True,
        )
        block_states = diagnostics.pop("block_states")
        memories, route_diagnostics = self.route_memories(
            block_states,
            mode=route_mode,
            off_destinations=route_off_destinations,
            return_diagnostics=True,
        )
        next_h12, next_h12_pos = self._append_ring(
            state.h12_ring, state.h12_positions, memories["b1"], state.position
        )
        next_h10, next_h10_pos = self._append_ring(
            state.h10_ring, state.h10_positions, memories["b3"], state.position
        )
        next_h8, next_h8_pos = self._append_ring(
            state.h8_ring, state.h8_positions, memories["b5"], state.position
        )
        next_h7, next_h7_pos = self._append_ring(
            state.h7_ring, state.h7_positions, memories["b6"], state.position
        )
        next_state = replace(
            provisional,
            h12_ring=next_h12,
            h12_positions=next_h12_pos,
            h10_ring=next_h10,
            h10_positions=next_h10_pos,
            h8_ring=next_h8,
            h8_positions=next_h8_pos,
            h7_ring=next_h7,
            h7_positions=next_h7_pos,
        )
        self._validate_incremental_state(next_state)
        if not return_diagnostics:
            return logits, next_state
        diagnostics["route"] = route_diagnostics
        diagnostics["cache_audit"] = self.incremental_cache_audit(next_state)
        diagnostics["transient_router"] = {
            "unique_current_token_states": 11,
            "logical_candidate_references": sum(EXPECTED_CANDIDATE_COUNTS.values()),
            "persistent_candidate_history_states": 0,
        }
        return logits, next_state, diagnostics


FixedRecurrentPyramidGPT = fixed.AlternatingIntegrationRecurrentPyramidGPT


def expected_router_parameter_names(prefix="source_routers") -> Tuple[str, ...]:
    names = []
    for destination in ("b1", "b3", "b5", "b6"):
        names.extend((
            f"{prefix}.{destination}.query",
            f"{prefix}.{destination}.norm.weight",
            f"{prefix}.{destination}.gate",
        ))
    return tuple(names)


def router_parameter_manifest(model: RoutedRecurrentPyramidGPT):
    named = dict(model.named_parameters())
    expected = expected_router_parameter_names()
    rows = {
        name: {
            "shape": list(named[name].shape),
            "numel": named[name].numel(),
            "initialization": (
                "zero" if name.endswith((".query", ".gate")) else "one"
            ),
        }
        for name in expected
    }
    count = sum(named[name].numel() for name in expected)
    checks = {
        "exact_names": sorted(name for name in named if name.startswith("source_routers."))
        == sorted(expected),
        "exact_tensor_count": len(expected) == 12,
        "exact_new_parameter_count": count == ROUTER_PARAMETER_COUNT,
        "queries_zero": all(
            torch.count_nonzero(named[name]).item() == 0
            for name in expected if name.endswith(".query")
        ),
        "norms_one": all(
            torch.equal(named[name], torch.ones_like(named[name]))
            for name in expected if name.endswith(".norm.weight")
        ),
        "gates_zero": all(
            torch.count_nonzero(named[name]).item() == 0
            for name in expected if name.endswith(".gate")
        ),
    }
    return {
        "parameters": rows,
        "new_tensor_count": len(expected),
        "new_parameter_count": count,
        "checks": checks,
        "passed": all(checks.values()),
    }
