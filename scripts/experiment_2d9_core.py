"""2D9: O1 attention with an optional token-conditioned scalar gate.

The inherited kernels still own masking, attention, projection, and caches.
An explicit immutable gate argument carries each invocation's residual-derived
coefficient through activation checkpoint recomputation; no residual is cached.
"""
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn

import experiment_2d7_core as prior

W_NAMES = {0: "w_B1", 2: "w_B3", 4: "w_B5"}
PARAMETER_COUNTS = {"S": 124_475_908, "D": 124_478_212}
CONDITIONS = {"S": "STATIC_REAL", "D": "DYNAMIC_REAL"}
RMS_EPSILON = 1e-5


def architecture_manifest(arm):
    if arm not in PARAMETER_COUNTS:
        raise ValueError(arm)
    value = prior.architecture_manifest("O")
    value.update(experiment="2D9", arm=arm, condition=CONDITIONS[arm],
                 description="token-conditioned-dynamic-recurrent-gating",
                 parameter_count=PARAMETER_COUNTS[arm],
                 new_parameters=2304 if arm == "D" else 0,
                 gate_formula="tanh(g0 + sum(RMS(h_pre_ln1) * w))" if arm == "D" else "tanh(g0)",
                 rms_epsilon=RMS_EPSILON, gate_precision="FP32 then cast to attention dtype",
                 dynamic_vector_names=list(W_NAMES.values()) if arm == "D" else [],
                 gate_shared_across_heads=True, persistent_gate_cache=False)
    return value


def architecture_fingerprint(arm):
    return hashlib.sha256(json.dumps(architecture_manifest(arm), sort_keys=True,
                                    separators=(",", ":")).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class TokenGate:
    preactivation: torch.Tensor
    coefficient: torch.Tensor


class DynamicGatingGPT(prior.BoundaryAlignmentGPT):
    def __init__(self, base, arm="S"):
        super().__init__(base, "O")
        self.arm = "S"
        self.gate_mode = "real"
        self.gate_collector = None
        if arm == "D":
            self.enable_dynamic()
        elif arm != "S":
            raise ValueError(arm)

    def enable_dynamic(self):
        """Call after restoring the old optimizer, then append these vectors."""
        if self.arm != "S" or any(hasattr(self, n) for n in W_NAMES.values()):
            raise ValueError("dynamic vectors already registered")
        reference = self.g_rec
        for name in W_NAMES.values():
            self.register_parameter(name, nn.Parameter(torch.zeros(
                int(self.config.n_embd), device=reference.device, dtype=torch.float32)))
        self.arm = "D"

    def architecture_fingerprint(self):
        return architecture_fingerprint(self.arm)

    def set_gate_mode(self, mode):
        if mode not in ("real", "staticized") or (self.arm == "S" and mode != "real"):
            raise ValueError(mode)
        self.gate_mode = mode

    def intrinsic_gate(self, residual, block_index):
        with torch.autocast(device_type=residual.device.type, enabled=False):
            h = residual.float()
            r = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + RMS_EPSILON)
            u = self.gate_parameter(block_index).float() + (
                r * getattr(self, W_NAMES[block_index]).float()).sum(-1, keepdim=True)
            return TokenGate(u, torch.tanh(u))

    def _explicit_gate(self, residual, block_index, gate_override):
        if gate_override is not None:
            raise ValueError("2D9 controls use model-wide gate mode only")
        if self.arm == "D" and self.gate_mode == "real":
            return self.intrinsic_gate(residual, block_index)
        return None

    def _gate_coefficient(self, block_index, reference, gate_override):
        if isinstance(gate_override, TokenGate):
            coefficient = gate_override.coefficient.to(reference.dtype).unsqueeze(1)
            if self.gate_collector is not None:
                self.gate_collector.record(block_index, gate_override, coefficient)
            return coefficient
        return super()._gate_coefficient(block_index, reference, gate_override)

    def _parallel_special_block(self, residual, block_index, recurrent_source,
                                recurrent_permutation, gate_override, return_diagnostics):
        gate = self._explicit_gate(residual, block_index, gate_override)
        return super()._parallel_special_block(residual, block_index, recurrent_source,
                                               recurrent_permutation, gate, return_diagnostics)

    def _incremental_special_block(self, residual, block_index, cache, recurrent_bank,
                                   permutation, gate_override, local_capacity,
                                   return_diagnostics, diagnostic_attention_weights):
        gate = self._explicit_gate(residual, block_index, gate_override)
        return super()._incremental_special_block(
            residual, block_index, cache, recurrent_bank, permutation, gate,
            local_capacity, return_diagnostics, diagnostic_attention_weights)
