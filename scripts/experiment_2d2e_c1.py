#!/usr/bin/env python3
"""Frozen large true-self confirmation for finalized Experiment 2D2E.

This driver performs no training and never constructs an optimizer.  It
strictly loads the finalized 2D2E model, reconstructs the original four-batch
incremental subset, selects a token-disjoint sixteen-batch continuation, and
evaluates deployment-equivalent incremental Real, B3-off, and B3-shuffled
controls.  The confirmatory bootstrap is preregistered in constants below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d2a as legacy  # noqa: E402
from experiment_2d2e_core import RecurrentKVGPT  # noqa: E402


EXPERIMENT = "2D2E-C1"
CHECKPOINT_SHA256 = "dea5e76b55d1ad7281fe3cf3893713392343b875182ba186a0049904e61de790"
CHECKPOINT_BYTES = 1_493_942_757
CHECKPOINT_SCHEMA = "exp2d2e_b3_w64_b10_recurrent_960_checkpoint_v1"
VALIDATION_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
EXPECTED_PARAMETERS = 124_475_907
T = 1024
B = 64
ORIGINAL_BATCHES = 4
CONFIRM_BATCHES = 16
TARGETS_PER_CONTROL = CONFIRM_BATCHES * B * T
EXPECTED_SEQUENCES = CONFIRM_BATCHES * B
# The old subset read token indices 0..262144 inclusive.  Starting at 262145
# makes both input and target token intervals literally disjoint.
CONFIRM_START_TOKEN_OFFSET = ORIGINAL_BATCHES * B * T + 1
BOOTSTRAP_SEED = 20_260_221
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
CONTROLS = ("all_real", "b3_off", "b3_shuffled")


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def durable_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def paired_stats(real, comparator) -> dict:
    left = np.asarray(real, dtype=np.float64)
    right = np.asarray(comparator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired inputs must be equal nonempty vectors")
    # Positive means the comparator is worse and therefore recurrence helps.
    effect = right - left
    return {
        "count": int(effect.size),
        "mean_comparator_minus_real": float(effect.mean()),
        "median_comparator_minus_real": float(np.median(effect)),
        "sample_std": float(effect.std(ddof=1)) if effect.size > 1 else 0.0,
        "wins": int(np.count_nonzero(effect > 0)),
        "losses": int(np.count_nonzero(effect < 0)),
        "ties": int(np.count_nonzero(effect == 0)),
        "differences_comparator_minus_real": effect.tolist(),
    }


def bootstrap_mean_ci(effects, *, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES) -> dict:
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap requires a nonempty vector")
    if int(resamples) < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    chunk = 100
    for start in range(0, int(resamples), chunk):
        stop = min(start + chunk, int(resamples))
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, BOOTSTRAP_PERCENTILES, method="linear")
    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "sampling_unit": "paired validation sequence",
        "statistic": "mean(comparator_loss - real_loss)",
        "percentile_method": "numpy linear",
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_mean": float(means.mean()),
    }


def classify_confirmation(off_effect: float, shuffled_effect: float, off_ci, shuffled_ci) -> str:
    if off_effect <= 0 or shuffled_effect <= 0:
        return "NOT CONFIRMED"
    if off_ci["lower"] > 0 and shuffled_ci["lower"] > 0:
        return "STRONG CONFIRMATION"
    return "DIRECTIONAL CONFIRMATION"


def sequence_identity(x: torch.Tensor, y: torch.Tensor) -> dict:
    if tuple(x.shape) != (T,) or tuple(y.shape) != (T,):
        raise ValueError("sequence identity requires one T-token row")
    input_bytes = x.contiguous().numpy().tobytes()
    target_bytes = y.contiguous().numpy().tobytes()
    return {
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "combined_sha256": hashlib.sha256(input_bytes + target_bytes).hexdigest(),
    }


def require_visible_a100() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("C1 requires exactly one CUDA-visible physical lane")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name or properties.total_memory < 79 * 1024**3:
        raise SystemExit(f"C1 requires an 80-GB A100, observed {properties.name}")
    torch.cuda.set_device(0)
    return {
        "visible_device_count": 1,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "uuid": getattr(properties, "uuid", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def load_model(checkpoint: Path, device: torch.device):
    checkpoint = checkpoint.resolve()
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit("2D2E checkpoint byte size mismatch")
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("2D2E checkpoint SHA-256 mismatch")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    verification = checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != CHECKPOINT_SHA256:
        raise SystemExit("2D2E checkpoint SHA sidecar missing or mismatched")
    if not verification.is_file() or not json.loads(verification.read_text()).get("passed"):
        raise SystemExit("2D2E checkpoint strict verification sidecar did not pass")
    payload = legacy.d0.torch_load(checkpoint, mmap=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("2D2E checkpoint schema mismatch")
    symbols = legacy.d0.support.load_training_symbols()
    base = symbols["GPT"](legacy.d0.model_config(symbols))
    model = RecurrentKVGPT(base).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict frozen model load failed: {missing}, {unexpected}")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise SystemExit("frozen 2D2E parameter count mismatch")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise SystemExit("frozen 2D2E model contains nonfinite parameters")
    model.eval()
    return model, {
        "schema": payload["schema"],
        "git_commit": payload.get("git_commit"),
        "completed_2d2e_updates": payload.get("completed_2d2e_updates"),
        "processed_2d2e_targets": payload.get("processed_2d2e_targets"),
        "parameters": EXPECTED_PARAMETERS,
        "gates": {
            "b1": model.recurrent_scale_b1.detach().float().item(),
            "b2": model.recurrent_scale_b2.detach().float().item(),
            "b3": model.recurrent_scale_b3.detach().float().item(),
        },
    }


def reconstruct_subsets(validation: Path, prior_incremental: Path):
    prior = json.loads(prior_incremental.read_text())
    if prior.get("batch_count") != ORIGINAL_BATCHES or prior.get("batch_size") != B:
        raise SystemExit("prior 2D2E incremental provenance geometry mismatch")
    if prior.get("sequence_length") != T or prior.get("targets_per_control") != ORIGINAL_BATCHES * B * T:
        raise SystemExit("prior 2D2E incremental target provenance mismatch")

    old_loader = legacy.d1.ExplicitShardLoader([validation], B, T)
    old_batches = []
    old_sequences = []
    for batch_index in range(ORIGINAL_BATCHES):
        x, y = old_loader.next_batch()
        identity = legacy.d0d.batch_identity(x, y)
        old_batches.append(identity)
        for row in range(B):
            old_sequences.append({"batch": batch_index, "row": row, **sequence_identity(x[row], y[row])})
    if old_batches != prior.get("batch_identities"):
        raise SystemExit("could not exactly reconstruct original 2D2E incremental subset")
    if old_loader.current_position != ORIGINAL_BATCHES * B * T:
        raise SystemExit("unexpected old validation loader cursor")

    state = old_loader.state_dict()
    state["current_position"] = CONFIRM_START_TOKEN_OFFSET
    new_loader = legacy.d1.ExplicitShardLoader([validation], B, T, state=state)
    new_batches = []
    new_sequences = []
    tensors = []
    for batch_index in range(CONFIRM_BATCHES):
        x, y = new_loader.next_batch()
        new_batches.append(legacy.d0d.batch_identity(x, y))
        for row in range(B):
            new_sequences.append({"batch": batch_index, "row": row, **sequence_identity(x[row], y[row])})
        tensors.append((x, y))

    old_hashes = {row["combined_sha256"] for row in old_sequences}
    new_hashes = {row["combined_sha256"] for row in new_sequences}
    if len(old_hashes) != len(old_sequences) or len(new_hashes) != len(new_sequences):
        raise SystemExit("duplicate validation sequence discovered")
    if old_hashes & new_hashes:
        raise SystemExit("C1 sequence hashes overlap original 2D2E subset")
    old_token_interval = [0, ORIGINAL_BATCHES * B * T]
    new_token_interval = [CONFIRM_START_TOKEN_OFFSET, CONFIRM_START_TOKEN_OFFSET + CONFIRM_BATCHES * B * T]
    if old_token_interval[1] >= new_token_interval[0]:
        raise SystemExit("C1 raw token intervals overlap")
    manifest = {
        "schema": "exp2d2e_c1_disjoint_subset_v1",
        "validation_shard": str(validation.resolve()),
        "validation_shard_sha256": VALIDATION_SHA256,
        "old_subset": {
            "batch_count": ORIGINAL_BATCHES,
            "sequence_count": len(old_sequences),
            "targets": ORIGINAL_BATCHES * B * T,
            "raw_token_interval_inclusive": old_token_interval,
            "batch_identities": old_batches,
            "sequence_identities": old_sequences,
            "canonical_subset_sha256": prior.get("canonical_subset_sha256"),
        },
        "c1_subset": {
            "batch_count": CONFIRM_BATCHES,
            "sequence_count": len(new_sequences),
            "targets": TARGETS_PER_CONTROL,
            "start_cursor": {"current_shard": 0, "current_position": CONFIRM_START_TOKEN_OFFSET},
            "end_cursor": new_loader.state_dict(),
            "raw_token_interval_inclusive": new_token_interval,
            "batch_identities": new_batches,
            "sequence_identities": new_sequences,
            "batch_collection_sha256": legacy.d0.aggregate_hashes(
                [row["combined_sha256"] for row in new_batches]
            ),
        },
        "checks": {
            "original_batches_reconstructed_exactly": True,
            "raw_token_intervals_disjoint": True,
            "sequence_hash_sets_disjoint": True,
            "new_sequence_hashes_unique": True,
            "target_count_exact": TARGETS_PER_CONTROL == 1_048_576,
            "sequence_count_exact": len(new_sequences) == EXPECTED_SEQUENCES,
        },
    }
    manifest["passed"] = all(manifest["checks"].values())
    return manifest, tensors


@torch.no_grad()
def evaluate_controls(model, batches) -> dict:
    device = next(model.parameters()).device
    rows = {
        name: {"loss_sum": 0.0, "targets": 0, "per_batch_losses": [], "per_sequence_losses": [], "cache_audits": []}
        for name in CONTROLS
    }
    permutation = torch.arange(B, device=device).roll(1)
    if bool((permutation == torch.arange(B, device=device)).any()):
        raise SystemExit("C1 row permutation is not fixed-point-free")
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (cpu_x, cpu_y) in enumerate(batches, 1):
        x, y = cpu_x.to(device), cpu_y.to(device)
        for control in CONTROLS:
            state = model.init_incremental_state(B, device=device)
            sequence_sum = torch.zeros(B, dtype=torch.float64)
            total = 0.0
            for position in range(T):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, state, _ = model.incremental_step(
                        x[:, position],
                        state,
                        control=control,
                        recurrent_permutation=permutation if control == "b3_shuffled" else None,
                        return_diagnostics=False,
                        diagnostic_attention_weights=False,
                    )
                losses = F.cross_entropy(
                    logits[:, 0].float(), y[:, position], reduction="none"
                ).double().cpu()
                sequence_sum += losses
                total += losses.sum().item()
            audit = model.incremental_cache_audit(state)
            row = rows[control]
            row["loss_sum"] += total
            row["targets"] += B * T
            row["per_batch_losses"].append(total / (B * T))
            row["per_sequence_losses"].extend((sequence_sum / T).tolist())
            row["cache_audits"].append(audit)
            del state, losses, logits, sequence_sum
            torch.cuda.empty_cache()
        print(f"2D2E-C1 incremental batch={batch_index:02d}/{CONFIRM_BATCHES}", flush=True)
        del x, y
    controls = {}
    for name, row in rows.items():
        controls[name] = {
            "validation_loss": row["loss_sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["per_batch_losses"],
            "per_sequence_losses": row["per_sequence_losses"],
            "cache_audits": row["cache_audits"],
        }
    real = controls["all_real"]
    off = controls["b3_off"]
    shuffled = controls["b3_shuffled"]
    off_stats = paired_stats(real["per_sequence_losses"], off["per_sequence_losses"])
    shuffled_stats = paired_stats(real["per_sequence_losses"], shuffled["per_sequence_losses"])
    return {
        "controls": controls,
        "confirm_gain": off["validation_loss"] - real["validation_loss"],
        "confirm_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "off_minus_real": off_stats,
        "shuffled_minus_real": shuffled_stats,
        "fixed_point_free_row_permutation": permutation.cpu().tolist(),
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }


def render_report(summary: dict) -> str:
    return "\n".join(
        [
            "# Experiment 2D2E-C1 Frozen Large True-Self Confirmation",
            "",
            f"Classification: **{summary['classification']}**",
            "",
            f"- Confirm gain (Off - Real): `{summary['confirm_gain']}`",
            f"- Confirm sequence gap (Shuffled - Real): `{summary['confirm_sequence_gap']}`",
            f"- Off - Real 95% bootstrap CI: `{summary['bootstrap']['off_minus_real']['lower']}` to `{summary['bootstrap']['off_minus_real']['upper']}`",
            f"- Shuffled - Real 95% bootstrap CI: `{summary['bootstrap']['shuffled_minus_real']['lower']}` to `{summary['bootstrap']['shuffled_minus_real']['upper']}`",
            f"- Paired wins vs Off: `{summary['paired']['off_minus_real']['wins']}/{EXPECTED_SEQUENCES}`",
            f"- Paired wins vs Shuffled: `{summary['paired']['shuffled_minus_real']['wins']}/{EXPECTED_SEQUENCES}`",
            f"- Targets/control: `{TARGETS_PER_CONTROL}`",
            f"- Sequences/control: `{EXPECTED_SEQUENCES}`",
            "",
            "This frozen confirmation does not alter the official 2D2E classification.",
            "",
        ]
    )


def run(args) -> dict:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    validation = Path(args.validation_shard).resolve()
    prior = Path(args.prior_incremental).resolve()
    stop_audit = json.loads(Path(args.stop_audit).read_text())
    if not stop_audit.get("passed") or stop_audit.get("pod_id") != args.pod_id:
        raise SystemExit("authenticated exact-pod stop preflight is missing")
    hardware = require_visible_a100()
    if file_sha256(validation) != VALIDATION_SHA256:
        raise SystemExit("validation shard SHA-256 mismatch")
    checkpoint_before = file_sha256(checkpoint)
    model, model_manifest = load_model(checkpoint, torch.device("cuda", 0))
    subset, batches = reconstruct_subsets(validation, prior)
    durable_json(output / "subset_manifest.json", subset)
    evaluation = evaluate_controls(model, batches)
    off_effects = evaluation["off_minus_real"]["differences_comparator_minus_real"]
    shuffled_effects = evaluation["shuffled_minus_real"]["differences_comparator_minus_real"]
    bootstrap = {
        "off_minus_real": bootstrap_mean_ci(off_effects),
        "shuffled_minus_real": bootstrap_mean_ci(shuffled_effects),
    }
    classification = classify_confirmation(
        evaluation["confirm_gain"], evaluation["confirm_sequence_gap"],
        bootstrap["off_minus_real"], bootstrap["shuffled_minus_real"],
    )
    cache_passed = all(
        audit.get("passed")
        for control in evaluation["controls"].values()
        for audit in control["cache_audits"]
    )
    checkpoint_after = file_sha256(checkpoint)
    checks = {
        "exact frozen checkpoint SHA before evaluation": checkpoint_before == CHECKPOINT_SHA256,
        "exact frozen checkpoint SHA after evaluation": checkpoint_after == CHECKPOINT_SHA256,
        "checkpoint not mutated": checkpoint_before == checkpoint_after,
        "no optimizer constructed": True,
        "no backward or parameter update": True,
        "source model parameter count exact": model_manifest["parameters"] == EXPECTED_PARAMETERS,
        "old subset reconstructed": subset["checks"]["original_batches_reconstructed_exactly"],
        "new subset token-disjoint": subset["checks"]["raw_token_intervals_disjoint"],
        "new subset sequence-disjoint": subset["checks"]["sequence_hash_sets_disjoint"],
        "1,048,576 targets per control": all(
            row["validation_targets"] == TARGETS_PER_CONTROL
            for row in evaluation["controls"].values()
        ),
        "1,024 paired sequences": evaluation["off_minus_real"]["count"] == EXPECTED_SEQUENCES,
        "fixed-point-free shuffled rows": all(
            index != value for index, value in enumerate(evaluation["fixed_point_free_row_permutation"])
        ),
        "physical incremental cache audit": cache_passed,
        "authenticated stop capability preregistered": True,
    }
    audit = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not audit["passed"]:
        raise SystemExit(f"2D2E-C1 final audit failed: {checks}")
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "official_2d2e_classification_unchanged": "B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY",
        "confirm_gain": evaluation["confirm_gain"],
        "confirm_sequence_gap": evaluation["confirm_sequence_gap"],
        "paired": {
            "off_minus_real": evaluation["off_minus_real"],
            "shuffled_minus_real": evaluation["shuffled_minus_real"],
        },
        "bootstrap": bootstrap,
        "controls": evaluation["controls"],
        "performance": evaluation["performance"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_after,
        "model": model_manifest,
        "hardware": hardware,
        "pod": {"id": args.pod_id, "name": args.pod_name, "gpu_lane": 0},
        "subset_manifest": str(output / "subset_manifest.json"),
    }
    durable_json(output / "paired_results.json", {
        "controls": evaluation["controls"],
        "off_minus_real": evaluation["off_minus_real"],
        "shuffled_minus_real": evaluation["shuffled_minus_real"],
    })
    durable_json(output / "bootstrap_results.json", bootstrap)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "HEARTBEAT.json", {
        "experiment": EXPERIMENT, "status": "SUCCESS", "timestamp": time.time(),
        "classification": classification,
    })
    durable_text(output / "C1_FINAL_REPORT.md", render_report(summary))
    print(f"EXPERIMENT_2D2E_C1_COMPLETE classification={classification}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-shard", required=True)
    parser.add_argument("--prior-incremental", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--stop-audit", required=True)
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    main()
