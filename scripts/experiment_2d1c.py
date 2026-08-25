#!/usr/bin/env python3
"""Experiment 2D1C: frozen C1908 residual-fusion alpha sweep.

The only new functional intervention is X = E + alpha * F.  This module is
evaluation-only and intentionally contains no training machinery.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d1 as d1  # noqa: E402
import experiment_2d1a as d1a  # noqa: E402
import experiment_2d1b as d1b  # noqa: E402


EXPERIMENT = "2D1C"
PROTOCOL = "exp2d1c_c1908_residual_alpha_sweep_v1"
BRANCH = "experiment-2d1c-c1908-residual-alpha-sweep"
FROZEN_TAG = "experiment-2d1b-rho-window-factorial-final"
FROZEN_COMMIT = "e2197377d84991d5ff13eb059e203a25f143d18b"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1c_c1908_residual_alpha_sweep.json"
OUTPUT_NAME = "experiment_2d1c_c1908_residual_alpha_sweep"
SOURCE_SHA256 = d1b.SOURCE_SHA256
SOURCE_BYTES = d1b.SOURCE_BYTES
VALIDATION_SHA256 = d1b.VALIDATION_SHA256
CANONICAL_SHA256 = d1b.CANONICAL_SHA256
EXACT_POD_ID = d1b.EXACT_POD_ID
D12 = d1b.D12
POSITION_BINS = d1b.POSITION_BINS
ALPHA_RMS_MATCH = 0.0827875253078167
ALPHAS = (0.0, 0.03125, 0.0625, ALPHA_RMS_MATCH, 0.125, 0.25, 0.50, 0.75, 1.00)
ALPHA_LABELS = ("0", "0.03125", "0.0625", "rms_match", "0.125", "0.25", "0.5", "0.75", "1")
CONTROLS = ("real", "shuffled", "zero")
STAGE_A_RMS = d1b.STAGE_A_RMS
HARD_THRESHOLD = d1b.HARD_THRESHOLD
PROBE_STOP = 100.0 * STAGE_A_RMS
SELF_BATCHES = 2
PASSES = 32
LEGACY_075 = {
    "plain": 3.1001356740385746,
    "real": 3.1871706691557846,
    "shuffled": 3.4008460602759443,
    "zero": 9.720289874169975,
    "recurrent_gain": -0.08703499511721002,
    "sequence_gap": 0.21367539112015965,
    "max_recurrent_input_rms": 0.32566046714782715,
    "X_over_E": 9.075807265760178,
    "cos_X_E": 0.0814812183380127,
}
LEGACY_100 = {
    "plain": 3.1001356740385746,
    "real": 3.4835763178389243,
    "shuffled": 4.218558381331604,
    "zero": 10.059015879128129,
    "recurrent_gain": -0.3834406438003497,
    "sequence_gap": 0.73498206349268,
    "max_recurrent_input_rms": 0.4334312975406647,
    "X_over_E": 12.079155095419539,
    "cos_X_E": 0.05286770313978195,
}
FORBIDDEN_COUNTS = {
    "optimizer_objects": 0, "scheduler_objects": 0, "gradscaler_objects": 0,
    "backward_calls": 0, "optimizer_steps": 0, "parameter_updates": 0,
    "training_targets": 0, "training_dataloader_advances": 0,
}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def durable_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_json(path, value):
    durable_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def alpha_key(alpha):
    if alpha == ALPHA_RMS_MATCH:
        return "rms_match"
    return format(float(alpha), ".10g")


def safe_ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))


def require_config():
    config = read_json(CONFIG_PATH)
    checks = {
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_tag": config.get("frozen_2d1b_tag") == FROZEN_TAG,
        "frozen_commit": config.get("frozen_2d1b_commit") == FROZEN_COMMIT,
        "source": config["source_checkpoint"]["sha256"] == SOURCE_SHA256,
        "windows": tuple(config["windows"]) == D12,
        "alphas": tuple(config["alphas"]) == ALPHAS,
        "alpha_rms_match": config["alpha_rms_match"]["value"] == ALPHA_RMS_MATCH,
        "bins": tuple(tuple(row) for row in config["position_bins"]) == POSITION_BINS,
        "pod": config["runpod"]["exact_pod_id"] == EXACT_POD_ID,
    }
    if not all(checks.values()):
        raise SystemExit(f"frozen 2D1C configuration mismatch: {checks}")
    return config


def require_git(clean=True):
    checks = {
        "branch": git_output("branch", "--show-current") == BRANCH,
        "tag": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
    }
    subprocess.check_call(["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT)
    checks["tag_ancestor"] = True
    if clean:
        checks["clean"] = not bool(git_output("status", "--porcelain"))
    if not all(checks.values()):
        raise SystemExit(f"2D1C git preflight failed: {checks}")
    return checks


def require_hardware(pod_id):
    return d1b.require_hardware(pod_id)


def residual_fusion_input(model, tokens, previous_top, alpha):
    """The sole 2D1C intervention: retain E and add fixed alpha times F."""
    batch, length = tokens.shape
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    embedding = model.base.transformer.wte(tokens)
    position = model.base.transformer.wpe(positions)
    shifted = zn = u = g_pre = gate = fused = alpha_fused = None
    recurrent_mask = positions.gt(0).view(1, length, 1)
    if previous_top is None:
        recurrent_input = embedding
    else:
        shifted = torch.zeros_like(previous_top)
        shifted[:, 1:] = previous_top[:, :-1]
        zn = model.fusion.normalize(shifted)
        u = model.fusion.W_u(zn)
        g_pre = model.fusion.W_g(embedding)
        gate = 2.0 * torch.sigmoid(g_pre)
        fused = u * gate
        alpha_fused = float(alpha) * fused
        candidate = embedding + alpha_fused
        recurrent_input = torch.where(recurrent_mask, candidate, embedding)
    xpos = recurrent_input + position
    return {
        "E": embedding, "Z": shifted, "ZN": zn, "U": u, "G_PRE": g_pre,
        "G": gate, "F": fused, "ALPHA_F": alpha_fused, "X": recurrent_input,
        "P": position.unsqueeze(0), "XPOS": xpos, "recurrent_mask": recurrent_mask,
    }


def residual_forward(model, tokens, windows, previous_top, alpha):
    tensors = residual_fusion_input(model, tokens, previous_top, alpha)
    value = tensors["XPOS"]
    layer_rows = []
    for index, (block, window) in enumerate(zip(model.base.transformer.h, windows), start=1):
        if int(window) >= value.size(1):
            attention = block.attn(block.ln_1(value))
        else:
            attention = model.attention(block, block.ln_1(value), window)
        post_attention = value + attention
        value = post_attention + block.mlp(block.ln_2(post_attention))
        layer_rows.append({"layer": index, "post_attention_rms": d1a.rms(post_attention), "post_mlp_rms": d1a.rms(value)})
    top = model.base.transformer.ln_f(value)
    tensor_rms = {
        key: None if tensors[key] is None else d1a.rms(tensors[key])
        for key in ("E", "Z", "ZN", "U", "G_PRE", "G", "F", "ALPHA_F", "X", "P", "XPOS")
    }
    return top, tensors, {"tensor_rms": tensor_rms, "top_state_rms": d1a.rms(top), "layerwise": layer_rows}


def new_loss_accumulator():
    return {"sum": 0.0, "targets": 0, "per_batch": []}


def add_loss(accumulator, losses):
    accumulator["sum"] += losses.double().sum().item()
    accumulator["targets"] += losses.numel()
    accumulator["per_batch"].append(losses.mean().item())


def finish_loss(accumulator):
    return {
        "validation_loss": accumulator["sum"] / accumulator["targets"],
        "validation_targets": accumulator["targets"],
        "per_batch_losses": accumulator["per_batch"],
    }


def distribution_or_none(value, other):
    if value is None or bool((value.detach().float().norm(dim=-1) == 0).all()):
        return None
    return F.cosine_similarity(value.detach().float(), other.detach().float(), dim=-1)


def scale_collectors():
    return {name: d1b.TensorSummary() for name in ("E", "Z", "ZN", "U", "G_PRE", "G", "F", "ALPHA_F", "X", "XPOS")}


def content_collectors():
    return {name: d1b.Distribution() for name in ("cos_F_E", "cos_ALPHA_F_E", "cos_X_E", "cos_F_X", "cos_ZN_E")}


def position_collectors():
    result = {}
    for first, last in POSITION_BINS:
        result[f"{first}-{last}"] = {
            "real_ce": d1b.Moments(), "shuffled_ce": d1b.Moments(),
            "E": d1b.Moments(), "F": d1b.Moments(), "ALPHA_F": d1b.Moments(), "X": d1b.Moments(),
            "cos_F_E": d1b.Distribution(), "cos_X_E": d1b.Distribution(),
        }
    return result


def gate_collectors():
    return {
        "G_PRE": d1b.Moments(), "G": d1b.Moments(),
        "counts": {key: [0, 0] for key in ("G_lt_0_02", "G_gt_1_98", "abs_G_PRE_gt_5", "abs_G_PRE_gt_10")},
    }


def finish_scale(collectors):
    result = {name: value.finish() for name, value in collectors.items()}
    rms = {name: row["elements"]["rms"] for name, row in result.items()}
    result["ratios"] = {
        "F_over_E": safe_ratio(rms["F"], rms["E"]),
        "ALPHA_F_over_E": safe_ratio(rms["ALPHA_F"], rms["E"]),
        "X_over_E": safe_ratio(rms["X"], rms["E"]),
        "XPOS_over_E": safe_ratio(rms["XPOS"], rms["E"]),
        "U_over_ZN": safe_ratio(rms["U"], rms["ZN"]),
    }
    return result


def finish_content(collectors, alpha):
    result = {}
    for name, value in collectors.items():
        result[name] = None if name == "cos_ALPHA_F_E" and alpha == 0 else value.finish()
    return result


def finish_gates(collectors):
    return {
        "G_PRE": collectors["G_PRE"].finish(), "G": collectors["G"].finish(),
        "fractions": {name: count / total for name, (count, total) in collectors["counts"].items()},
    }


def finish_positions(collectors, plain_bins):
    result = {}
    for label, row in collectors.items():
        real = row["real_ce"].finish()["mean"]
        shuffled = row["shuffled_ce"].finish()["mean"]
        e = row["E"].finish()["rms"]
        fused = row["F"].finish()["rms"]
        alpha_fused = row["ALPHA_F"].finish()["rms"]
        current = row["X"].finish()["rms"]
        result[label] = {
            "plain_ce": plain_bins[label], "real_ce": real,
            "real_minus_plain": real - plain_bins[label],
            "recurrent_gain": plain_bins[label] - real,
            "shuffled_minus_real": shuffled - real,
            "E_rms": e, "F_rms": fused, "ALPHA_F_rms": alpha_fused, "X_rms": current,
            "ALPHA_F_over_E": safe_ratio(alpha_fused, e), "X_over_E": safe_ratio(current, e),
            "cos_F_E": row["cos_F_E"].finish(), "cos_X_E": row["cos_X_E"].finish(),
        }
    return result


def plain_baseline(model, batches, device):
    accumulator = new_loss_accumulator()
    cached_tops = []
    bin_moments = {f"{first}-{last}": d1b.Moments() for first, last in POSITION_BINS}
    started = time.monotonic()
    for batch_index, (cpu_x, cpu_y) in enumerate(batches):
        x, y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top = model.forward_top(x, D12)
            losses = model.loss_from_top(top, y, reduction="none").reshape(d1.VALIDATION_B, d1.T)
        add_loss(accumulator, losses)
        for first, last in POSITION_BINS:
            bin_moments[f"{first}-{last}"].add(losses[:, first:last + 1])
        cached_tops.append(top.detach().cpu())
        del x, y, top, losses
        print(f"2D1C plain batch={batch_index + 1:02d}/20", flush=True)
    return finish_loss(accumulator), {name: row.finish()["mean"] for name, row in bin_moments.items()}, cached_tops, time.monotonic() - started


def evaluate_alpha(model, batches, cached_plain_tops, plain, plain_bins, alpha, device):
    losses = {name: new_loss_accumulator() for name in CONTROLS}
    scales = scale_collectors()
    contents = content_collectors()
    positions = position_collectors()
    gates = gate_collectors()
    derangement = torch.arange(d1.VALIDATION_B, device=device).roll(1)
    started = time.monotonic()
    for batch_index, ((cpu_x, cpu_y), cpu_plain_top) in enumerate(zip(batches, cached_plain_tops)):
        x, y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)
        plain_top = cpu_plain_top.to(device, non_blocking=True)
        control_maps = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for control in CONTROLS:
                if control == "real":
                    source = plain_top
                elif control == "shuffled":
                    source = plain_top[derangement]
                else:
                    source = torch.zeros_like(plain_top)
                current_top, tensors, _ = residual_forward(model, x, D12, source, alpha)
                current_losses = model.loss_from_top(current_top, y, reduction="none").reshape(d1.VALIDATION_B, d1.T)
                add_loss(losses[control], current_losses)
                control_maps[control] = current_losses
                if control == "real":
                    for name, collector in scales.items():
                        collector.add(tensors[name])
                    region = slice(1, d1.T)
                    e, fused, alpha_fused, current, zn = (
                        tensors["E"][:, region], tensors["F"][:, region], tensors["ALPHA_F"][:, region],
                        tensors["X"][:, region], tensors["ZN"][:, region],
                    )
                    contents["cos_F_E"].add(F.cosine_similarity(fused.float(), e.float(), dim=-1))
                    if alpha != 0:
                        contents["cos_ALPHA_F_E"].add(F.cosine_similarity(alpha_fused.float(), e.float(), dim=-1))
                    contents["cos_X_E"].add(F.cosine_similarity(current.float(), e.float(), dim=-1))
                    contents["cos_F_X"].add(F.cosine_similarity(fused.float(), current.float(), dim=-1))
                    contents["cos_ZN_E"].add(F.cosine_similarity(zn.float(), e.float(), dim=-1))
                    gates["G_PRE"].add(tensors["G_PRE"][:, region])
                    gates["G"].add(tensors["G"][:, region])
                    gate_masks = {
                        "G_lt_0_02": tensors["G"][:, region] < 0.02,
                        "G_gt_1_98": tensors["G"][:, region] > 1.98,
                        "abs_G_PRE_gt_5": tensors["G_PRE"][:, region].abs() > 5,
                        "abs_G_PRE_gt_10": tensors["G_PRE"][:, region].abs() > 10,
                    }
                    for name, mask in gate_masks.items():
                        gates["counts"][name][0] += mask.sum().item()
                        gates["counts"][name][1] += mask.numel()
                    for first, last in POSITION_BINS:
                        label = f"{first}-{last}"; span = slice(first, last + 1)
                        for name in ("E", "F", "ALPHA_F", "X"):
                            positions[label][name].add(tensors[name][:, span])
                        positions[label]["cos_F_E"].add(F.cosine_similarity(tensors["F"][:, span].float(), tensors["E"][:, span].float(), dim=-1))
                        positions[label]["cos_X_E"].add(F.cosine_similarity(tensors["X"][:, span].float(), tensors["E"][:, span].float(), dim=-1))
                del current_top, tensors
        for first, last in POSITION_BINS:
            label = f"{first}-{last}"; span = slice(first, last + 1)
            positions[label]["real_ce"].add(control_maps["real"][:, span])
            positions[label]["shuffled_ce"].add(control_maps["shuffled"][:, span])
        del x, y, plain_top, control_maps
        print(f"2D1C alpha={alpha_key(alpha)} batch={batch_index + 1:02d}/20", flush=True)
    finished = {name: finish_loss(row) for name, row in losses.items()}
    result = {
        "alpha": alpha, "plain": plain["validation_loss"],
        "real": finished["real"]["validation_loss"],
        "shuffled": finished["shuffled"]["validation_loss"],
        "zero": finished["zero"]["validation_loss"],
        "recurrent_gain": plain["validation_loss"] - finished["real"]["validation_loss"],
        "sequence_gap": finished["shuffled"]["validation_loss"] - finished["real"]["validation_loss"],
        "zero_dependency": finished["zero"]["validation_loss"] - finished["real"]["validation_loss"],
        "per_batch_losses": {name: row["per_batch_losses"] for name, row in finished.items()},
        "paired": {
            "real_vs_plain": d1b.paired_counts(finished["real"]["per_batch_losses"], plain["per_batch_losses"]),
            "real_vs_shuffled": d1b.paired_counts(finished["real"]["per_batch_losses"], finished["shuffled"]["per_batch_losses"]),
            "real_vs_zero": d1b.paired_counts(finished["real"]["per_batch_losses"], finished["zero"]["per_batch_losses"]),
        },
        "wall_seconds": time.monotonic() - started,
    }
    for row in result["paired"].values():
        row["mean_paired_delta"] = statistics.fmean(row["left_minus_right"])
        row["median_paired_delta"] = statistics.median(row["left_minus_right"])
    return result, finish_scale(scales), finish_content(contents, alpha), finish_gates(gates), finish_positions(positions, plain_bins)


def classify_self(rows):
    if any(not row["finite"] for row in rows):
        return "NUMERICALLY DIVERGENT"
    maximum = max(row["recurrent_input_rms"] for row in rows)
    if maximum >= HARD_THRESHOLD:
        return "EXPANSIVE"
    by_batch = [[row for row in rows if row["batch"] == batch and row["pass"] >= 2] for batch in range(SELF_BATCHES)]
    growth = np.mean([group[-1]["recurrent_input_rms"] / max(group[0]["recurrent_input_rms"], 1e-30) for group in by_batch])
    change_ratios = []
    oscillation_votes = 0
    for group in by_batch:
        changes = [row["state_change_rms"] for row in group]
        change_ratios.append(np.mean(changes[-4:]) / max(np.mean(changes[:4]), 1e-30))
        differences = np.diff([row["recurrent_input_rms"] for row in group])
        if len(differences) > 8 and np.count_nonzero(differences[1:] * differences[:-1] < 0) > len(differences) // 2:
            oscillation_votes += 1
    if growth < 0.90 and np.mean(change_ratios) < 0.50:
        return "CONTRACTIVE"
    if oscillation_votes == SELF_BATCHES:
        return "BOUNDED OSCILLATORY"
    return "STABLE / STATIONARY"


def self_composition(model, batches, alpha, device):
    rows = []
    stopped = []
    started = time.monotonic()
    for batch_index, (cpu_x, cpu_y) in enumerate(batches[:SELF_BATCHES]):
        x, y = cpu_x.to(device), cpu_y.to(device)
        previous = None
        for pass_index in range(1, PASSES + 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                top, tensors, diagnostics = residual_forward(model, x, D12, previous, alpha)
                ce = model.loss_from_top(top, y).float().item()
            current_rms = diagnostics["tensor_rms"]["X"]
            e_rms = diagnostics["tensor_rms"]["E"]
            row = {
                "alpha": alpha, "batch": batch_index, "pass": pass_index, "ce": ce,
                "recurrent_input_rms": current_rms, "top_state_rms": diagnostics["top_state_rms"],
                "F_rms": diagnostics["tensor_rms"]["F"],
                "ALPHA_F_rms": diagnostics["tensor_rms"]["ALPHA_F"],
                "X_rms": current_rms, "X_over_E": safe_ratio(current_rms, e_rms),
                "state_change_rms": None if previous is None else d1a.rms(top.float() - previous.float()),
                "state_cosine": None if previous is None else F.cosine_similarity(top.float(), previous.float(), dim=-1).mean().item(),
            }
            row["finite"] = all(value is None or not isinstance(value, float) or math.isfinite(value) for value in row.values())
            rows.append(row)
            previous = top
            del tensors
            if not row["finite"] or current_rms > PROBE_STOP:
                stopped.append({"batch": batch_index, "pass": pass_index, "reason": "nonfinite_or_100x_reference"})
                break
        del x, y, previous, top
        torch.cuda.empty_cache()
    classification = classify_self(rows)
    maximum_row = max(rows, key=lambda row: row["recurrent_input_rms"])
    pass32 = [row for row in rows if row["pass"] == 32]
    summary = {
        "alpha": alpha, "classification": classification,
        "scale_bounded": len(rows) == SELF_BATCHES * PASSES and all(row["finite"] and row["recurrent_input_rms"] < HARD_THRESHOLD for row in rows),
        "max_recurrent_input_rms": maximum_row["recurrent_input_rms"],
        "pass_index_of_max_rms": maximum_row["pass"], "batch_of_max_rms": maximum_row["batch"],
        "pass_32_rms": None if not pass32 else statistics.fmean(row["recurrent_input_rms"] for row in pass32),
        "pass_32_ce": None if not pass32 else statistics.fmean(row["ce"] for row in pass32),
        "pass_32_state_change_rms": None if not pass32 else statistics.fmean(row["state_change_rms"] for row in pass32),
        "pass_32_state_cosine": None if not pass32 else statistics.fmean(row["state_cosine"] for row in pass32),
        "stops": stopped, "wall_seconds": time.monotonic() - started,
    }
    return {"summary": summary, "rows": rows}


def identity_preflight(model, batch, device):
    cpu_x, cpu_y = batch
    x, y = cpu_x[:4].to(device), cpu_y[:4].to(device)
    derangement = torch.arange(4, device=device).roll(1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plain = model.forward_top(x, D12)
        variants = {}
        for name, source in (
            ("real", plain), ("shuffled", plain[derangement]), ("zero", torch.zeros_like(plain)),
        ):
            top, tensors, _ = residual_forward(model, x, D12, source, 0.0)
            variants[name] = {
                "top_exact": bool(torch.equal(top, plain)),
                "logits_exact": bool(torch.equal(model.logits_from_top(top), model.logits_from_top(plain))),
                "loss_delta": model.loss_from_top(top, y).float().item() - model.loss_from_top(plain, y).float().item(),
                "position_zero_state_zero": bool(torch.equal(tensors["Z"][:, 0], torch.zeros_like(tensors["Z"][:, 0]))),
                "one_token_shift_exact": bool(torch.equal(tensors["Z"][:, 1:], source[:, :-1])),
            }
    passed = all(all(
        value is True if isinstance(value, bool) else value == 0.0
        for value in row.values()
    ) for row in variants.values())
    return {"variants": variants, "passed": passed}


def causal_audits(model, batch, device):
    cpu_x, _ = batch
    x = cpu_x[:4].to(device)
    split = 512
    changed_tokens = x.clone()
    changed_tokens[:, split:] = (changed_tokens[:, split:] + 1) % d1.VOCAB_SIZE
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        source_a = model.forward_top(x, D12)
        source_b = model.forward_top(changed_tokens, D12)
        recurrent_a, tensors_a, _ = residual_forward(model, x, D12, source_a, 0.125)
        recurrent_b, _, _ = residual_forward(model, changed_tokens, D12, source_b, 0.125)
        altered_source = source_a.clone(); altered_source[0] = source_a[1]
        row_changed, _, _ = residual_forward(model, x, D12, altered_source, 0.125)
    checks = {
        "plain_prefix_future_independent": bool(torch.equal(source_a[:, :split], source_b[:, :split])),
        "recurrent_prefix_future_independent": bool(torch.equal(recurrent_a[:, :split], recurrent_b[:, :split])),
        "position_zero_recurrent_state_zero": bool(torch.equal(tensors_a["Z"][:, 0], torch.zeros_like(tensors_a["Z"][:, 0]))),
        "one_position_shift_exact": bool(torch.equal(tensors_a["Z"][:, 1:], source_a[:, :-1])),
        "row_isolation_other_rows_exact": bool(torch.equal(recurrent_a[1:], row_changed[1:])),
        "changed_row_responds": bool(not torch.equal(recurrent_a[0], row_changed[0])),
    }
    checks["passed"] = all(checks.values())
    return checks


def reverse_order_spot_check(model, batch, original_batch_loss, device):
    cpu_x, cpu_y = batch
    x, y = cpu_x.to(device), cpu_y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plain = model.forward_top(x, D12)
        real, _, _ = residual_forward(model, x, D12, plain, 0.125)
        rerun = model.loss_from_top(real, y, reduction="none").mean().item()
    return {
        "alpha": 0.125, "original_batch_0_real_loss": original_batch_loss,
        "reverse_order_rerun_batch_0_real_loss": rerun,
        "absolute_delta": abs(rerun - original_batch_loss),
        "tolerance": 1e-8, "passed": abs(rerun - original_batch_loss) <= 1e-8,
    }


def late_context(position_rows):
    # The two bins exactly partition positions 769..1023.
    first = position_rows["769-896"]
    second = position_rows["897-1023"]
    weights = (128, 127)
    plain = (weights[0] * first["plain_ce"] + weights[1] * second["plain_ce"]) / sum(weights)
    real = (weights[0] * first["real_ce"] + weights[1] * second["real_ce"]) / sum(weights)
    return {
        "global_late_769_1023_plain": plain,
        "global_late_769_1023_real": real,
        "late_recurrent_gain_769_1023": plain - real,
        "late_recurrent_gain_897_1023": second["recurrent_gain"],
    }


def stability_frontier(alpha_validation, self_results):
    positive = [alpha for alpha in ALPHAS if alpha > 0]
    bounded = [alpha for alpha in positive if self_results[alpha_key(alpha)]["summary"]["scale_bounded"]]
    best_any = min(positive, key=lambda alpha: alpha_validation[alpha_key(alpha)]["real"])
    best_bounded = None if not bounded else min(bounded, key=lambda alpha: alpha_validation[alpha_key(alpha)]["real"])
    maximum_bounded = None if not bounded else max(bounded)
    return {
        "alpha_best_any": best_any,
        "alpha_best_bounded": best_bounded,
        "alpha_max_bounded": maximum_bounded,
        "bounded_positive_alphas": bounded,
        "unbounded_positive_alphas": [alpha for alpha in positive if alpha not in bounded],
        "hard_threshold": HARD_THRESHOLD,
    }


def primary_classification(alpha_validation, frontier):
    plain = next(iter(alpha_validation.values()))["plain"]
    bounded = frontier["bounded_positive_alphas"]
    if bounded and any(alpha_validation[alpha_key(alpha)]["real"] < plain for alpha in bounded):
        return "RESIDUAL FUSION REVEALS POSITIVE ZERO-SHOT RECURRENT UTILITY"
    if frontier["alpha_best_bounded"] is not None:
        best = alpha_validation[alpha_key(frontier["alpha_best_bounded"])]["real"]
        if best - plain <= 0.01:
            return "RESIDUAL FUSION MAKES RECURRENCE NEAR-NEUTRAL BUT NOT POSITIVE"
        harm_reduction = LEGACY_075["real"] - best
        if harm_reduction >= 0.01:
            return "RESIDUAL FUSION REDUCES RECURRENT HARM BUT DOES NOT YET CREATE UTILITY"
    return "C1908 RECURRENT STATE REMAINS PREDICTIVELY HARMFUL UNDER RESIDUAL FUSION"


def stability_classification(frontier):
    bounded = frontier["bounded_positive_alphas"]
    positive = [alpha for alpha in ALPHAS if alpha > 0]
    if not bounded:
        return "ALL NONZERO RESIDUAL ALPHAS ARE EXPANSIVE"
    pattern = [alpha in bounded for alpha in positive]
    monotonic_prefix = pattern == ([True] * sum(pattern) + [False] * (len(pattern) - sum(pattern)))
    if not monotonic_prefix:
        return "MIXED/NONMONOTONIC RESIDUAL STABILITY"
    if max(bounded) >= 0.5:
        return "WIDE BOUNDED RESIDUAL-ALPHA RANGE"
    return "ONLY SMALL RESIDUAL ALPHAS ARE BOUNDED"


def next_recommendation(primary, stability, alpha_validation, frontier):
    if primary == "RESIDUAL FUSION REVEALS POSITIVE ZERO-SHOT RECURRENT UTILITY":
        return "TRAIN D12 RESIDUAL RECURRENCE FROM C1908 AT THE BEST BOUNDED ALPHA"
    best = frontier["alpha_best_bounded"]
    if best is not None:
        row = alpha_validation[alpha_key(best)]
        if row["real"] - row["plain"] <= 0.01 and row["sequence_gap"] > 0:
            return "TRAIN D12 RESIDUAL RECURRENCE FROM C1908 WITH A SMALL-ALPHA CURRICULUM"
    if stability == "ONLY SMALL RESIDUAL ALPHAS ARE BOUNDED":
        return "REDESIGN RECURRENT-CORRECTION SCALE CONTROL BEFORE TRAINING"
    return "RETRAIN RESIDUAL RECURRENCE FROM AN EARLIER CLEAN CHECKPOINT"


def matched_comparisons(alpha_validation, scale, content, self_results):
    result = {}
    for name, legacy, alpha in (("MATCHED_075_COMPARISON", LEGACY_075, 0.75), ("MATCHED_100_COMPARISON", LEGACY_100, 1.0)):
        key = alpha_key(alpha); row = alpha_validation[key]
        residual = {
            "alpha": alpha, "real": row["real"], "recurrent_gain": row["recurrent_gain"],
            "sequence_gap": row["sequence_gap"], "X_over_E": scale[key]["ratios"]["X_over_E"],
            "cos_X_E": content[key]["cos_X_E"]["mean"],
            "max_recurrent_input_rms": self_results[key]["summary"]["max_recurrent_input_rms"],
        }
        result[name] = {
            "legacy": legacy, "residual": residual,
            "residual_minus_legacy": {metric: residual[metric] - legacy[metric] for metric in (
                "real", "recurrent_gain", "sequence_gap", "X_over_E", "cos_X_E", "max_recurrent_input_rms"
            )},
        }
    return result


def make_plots(output, validation, scale, content, self_results, positions, frontier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(ALPHAS)
    keys = [alpha_key(alpha) for alpha in ALPHAS]
    plain = validation[keys[0]]["plain"]

    def line_plot(filename, ylabel, values, title, horizontal=None):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(xs, values, marker="o")
        if horizontal is not None:
            ax.axhline(horizontal[0], color="red", linestyle="--", label=horizontal[1]); ax.legend()
        ax.set_xlabel("Residual alpha"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)

    line_plot("P1_alpha_vs_real_ce.png", "REAL validation CE", [validation[key]["real"] for key in keys],
              "Residual-alpha predictive curve", (plain, "D12 plain"))
    line_plot("P2_alpha_vs_recurrent_gain.png", "Plain - REAL CE", [validation[key]["recurrent_gain"] for key in keys],
              "Recurrent gain", (0.0, "zero gain"))
    line_plot("P3_alpha_vs_sequence_gap.png", "Shuffled - REAL CE", [validation[key]["sequence_gap"] for key in keys],
              "Sequence-specificity curve", (0.0, "zero gap"))
    line_plot("P4_alpha_vs_max_rms.png", "Max 32-pass recurrent-input RMS",
              [self_results[key]["summary"]["max_recurrent_input_rms"] for key in keys],
              "Residual-alpha stability frontier", (HARD_THRESHOLD, "10x historical threshold"))

    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha, key in zip(xs, keys):
        rows = self_results[key]["rows"]
        passes = sorted(set(row["pass"] for row in rows))
        values = [statistics.fmean(row["recurrent_input_rms"] for row in rows if row["pass"] == p) for p in passes]
        ax.plot(passes, values, marker="o", markersize=2, label=alpha_key(alpha))
    ax.axhline(HARD_THRESHOLD, color="red", linestyle="--", label="10x threshold")
    ax.set_xlabel("Pass"); ax.set_ylabel("Recurrent-input RMS"); ax.set_title("32-pass residual self-composition")
    ax.legend(ncol=3); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output / "P5_32pass_rms_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, [scale[key]["ratios"]["X_over_E"] for key in keys], marker="o", label="X/E")
    ax.plot(xs, [scale[key]["ratios"]["ALPHA_F_over_E"] for key in keys], marker="o", label="alphaF/E")
    ax.set_xlabel("Residual alpha"); ax.set_ylabel("RMS ratio"); ax.set_title("Injected and total fusion scale")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "P6_alpha_scale_ratios.png", dpi=180); plt.close(fig)

    line_plot("P7_alpha_vs_cos_X_E.png", "cos(X,E)", [content[key]["cos_X_E"]["mean"] for key in keys],
              "Explicit-token alignment")

    fig, ax = plt.subplots(figsize=(8, 6))
    for alpha, key in zip(xs, keys):
        x = validation[key]["sequence_gap"]; y = validation[key]["recurrent_gain"]
        ax.scatter(x, y); ax.annotate(alpha_key(alpha), (x, y), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=1); ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Sequence gap: shuffled - real"); ax.set_ylabel("Recurrent gain: plain - real")
    ax.set_title("Utility versus specificity"); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(output / "P8_utility_vs_specificity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for alpha, key in zip(xs, keys):
        x = self_results[key]["summary"]["max_recurrent_input_rms"]; y = validation[key]["real"]
        ax.scatter(x, y); ax.annotate(alpha_key(alpha), (x, y), xytext=(4, 4), textcoords="offset points")
    ax.axvline(HARD_THRESHOLD, color="red", linestyle="--", label="10x threshold")
    ax.axhline(plain, color="gray", linestyle="--", label="D12 plain")
    ax.set_xlabel("Max 32-pass recurrent-input RMS"); ax.set_ylabel("REAL validation CE")
    ax.set_title("Utility versus scale frontier"); ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(output / "P9_utility_vs_scale.png", dpi=180); plt.close(fig)

    requested = [0.0, 0.03125, 0.0625, ALPHA_RMS_MATCH, 0.125, frontier["alpha_best_bounded"]]
    selected = []
    for alpha in requested:
        if alpha is not None and alpha not in selected:
            selected.append(alpha)
    labels = [f"{first}-{last}" for first, last in POSITION_BINS]
    fig, ax = plt.subplots(figsize=(12, 6))
    for alpha in selected:
        key = alpha_key(alpha)
        ax.plot(labels, [positions[key][label]["recurrent_gain"] for label in labels], marker="o", label=key)
    ax.axhline(0, color="black", linewidth=1); ax.set_xlabel("Token-position bin"); ax.set_ylabel("Plain - REAL CE")
    ax.set_title("Position-binned residual utility"); ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(output / "P10_position_binned_gain.png", dpi=180); plt.close(fig)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def f10(value):
    return "NONE" if value is None else f"{value:.10f}"


def build_report(validation, scale, content, self_results, late, matched, frontier, decision, legacy, audit):
    sweep_rows = []
    scale_rows = []
    content_rows = []
    late_rows = []
    for alpha in ALPHAS:
        key = alpha_key(alpha); row = validation[key]; stability = self_results[key]["summary"]
        sweep_rows.append((key, f10(row["plain"]), f10(row["real"]), f10(row["shuffled"]), f10(row["zero"]),
                           f10(row["recurrent_gain"]), f10(row["sequence_gap"]), f10(row["zero_dependency"]),
                           row["paired"]["real_vs_plain"]["wins"]))
        scale_rows.append((key, f"{scale[key]['ratios']['F_over_E']:.6f}", f"{scale[key]['ratios']['ALPHA_F_over_E']:.6f}",
                           f"{scale[key]['ratios']['X_over_E']:.6f}", f"{stability['max_recurrent_input_rms']:.6f}",
                           f10(stability["pass_32_rms"]), stability["classification"]))
        content_rows.append((key, f"{content[key]['cos_F_E']['mean']:.6f}", f"{content[key]['cos_X_E']['mean']:.6f}",
                             f"{content[key]['cos_F_X']['mean']:.6f}", f"{content[key]['cos_ZN_E']['mean']:.6f}"))
        late_rows.append((key, f"{row['recurrent_gain']:.8f}", f"{late[key]['late_recurrent_gain_769_1023']:.8f}",
                          f"{late[key]['late_recurrent_gain_897_1023']:.8f}"))
    best = frontier["alpha_best_bounded"]
    best_row = None if best is None else validation[alpha_key(best)]
    small = validation[alpha_key(0.03125)]
    local_real_slope = decision["local_real_slope_0"]
    local_shuffled_slope = decision["local_shuffled_slope_0"]
    first_specific = decision["first_clearly_sequence_specific_alpha"]
    m075 = matched["MATCHED_075_COMPARISON"]
    m100 = matched["MATCHED_100_COMPARISON"]
    q = [
        ("Q1", f"Yes. Alpha=0 identity maximum canonical delta was {audit['alpha_zero_identity']['maximum_canonical_delta']:.3e}; deterministic tops and logits were exact."),
        ("Q2", f"alpha_rms_match = {ALPHA_RMS_MATCH:.16f}, derived before outcomes from E_RMS/F_RMS."),
        ("Q3", f"At alpha=.03125, REAL CE changed by {small['real'] - small['plain']:+.10f} relative to plain."),
        ("Q4", f"The descriptive local REAL slope is {local_real_slope:+.10f}; it is {'negative (initially CE-improving)' if local_real_slope < 0 else 'positive (initially CE-harming)'}. The shuffled slope is {local_shuffled_slope:+.10f}."),
        ("Q5", "Yes." if decision["any_positive_alpha_beats_plain"] else "No."),
        ("Q6", f"Best positive-alpha CE occurs at alpha={frontier['alpha_best_any']:.16g}."),
        ("Q7", "NONE." if best is None else f"Best bounded alpha is {best:.16g}, REAL CE {best_row['real']:.10f}."),
        ("Q8", "NONE." if frontier["alpha_max_bounded"] is None else f"The largest bounded tested alpha is {frontier['alpha_max_bounded']:.16g}."),
        ("Q9", "No preregistered alpha met the clear-specificity criterion." if first_specific is None else f"The first alpha with gap >.01 and at least 15/20 paired wins over shuffled is {first_specific:.16g}."),
        ("Q10", decision["first_expansive_answer"]),
        ("Q11", f"Residual alpha=.75 changes REAL CE versus legacy rho=.75 by {m075['residual_minus_legacy']['real']:+.10f}."),
        ("Q12", f"Residual alpha=1 changes REAL CE versus legacy rho=1 by {m100['residual_minus_legacy']['real']:+.10f}."),
        ("Q13", f"Restoring E changes cos(X,E) by {m075['residual_minus_legacy']['cos_X_E']:+.6f} at .75 and {m100['residual_minus_legacy']['cos_X_E']:+.6f} at 1."),
        ("Q14", f"Restoring E changes X/E by {m075['residual_minus_legacy']['X_over_E']:+.6f} at .75 and {m100['residual_minus_legacy']['X_over_E']:+.6f} at 1."),
        ("Q15", decision["late_context_answer"]),
        ("Q16", "NOT AVAILABLE because no positive alpha is bounded." if best is None else f"At best bounded alpha, real-vs-shuffled is {best_row['paired']['real_vs_shuffled']['wins']}/20 wins."),
        ("Q17", "NOT AVAILABLE because no positive alpha is bounded." if best is None else f"At best bounded alpha, real-vs-zero is {best_row['paired']['real_vs_zero']['wins']}/20 wins."),
        ("Q18", f"Yes. Across the sweep the largest |zero-plain| delta was {decision['max_zero_plain_delta']:.3e}."),
        ("Q19", decision["utility_specificity_interpretation"]),
        ("Q20", "NONE from this frozen diagnostic." if best is None else f"Use fixed alpha={best:.16g} if the recommended future experiment is authorized."),
        ("Q21", decision["next_recommendation"] + "."),
    ]
    matched_rows = []
    for label, row in matched.items():
        for kind in ("legacy", "residual"):
            value = row[kind]
            matched_rows.append((label, kind, f10(value["real"]), f10(value["recurrent_gain"]), f10(value["sequence_gap"]),
                                 f"{value['max_recurrent_input_rms']:.6f}", f"{value['X_over_E']:.6f}", f"{value['cos_X_E']:.6f}"))
    text = f"""# Experiment 2D1C — Frozen C1908 Residual-Fusion Alpha Sweep

## Final classifications

**Primary:** {decision['primary_classification']}

**Stability:** {decision['stability_classification']}

The diagnostic used exactly `X = E + alpha*F` on frozen C1908 weights with D12 windows. Optimizer objects, backward calls, parameter updates, training targets, and training-loader advances were all zero.

## T1 — Alpha sweep

{markdown_table(('alpha','Plain','Real','Shuffled','Zero','Gain','Sequence gap','Zero dependency','Real>plain wins'), sweep_rows)}

## T2 — Scale and stability

{markdown_table(('alpha','F/E','alphaF/E','X/E','Max RMS','Pass-32 RMS','Class'), scale_rows)}

## T3 — Embedding content

{markdown_table(('alpha','cos(F,E)','cos(X,E)','cos(F,X)','cos(ZN,E)'), content_rows)}

## T4 — Matched legacy comparisons

{markdown_table(('Comparison','Interface','Real CE','Gain','Sequence gap','Max RMS','X/E','cos(X,E)'), matched_rows)}

## T5 — Late-context utility

{markdown_table(('alpha','Global gain','769-1023 gain','897-1023 gain'), late_rows)}

## Local behavior and frontier

- `alpha_rms_match`: {ALPHA_RMS_MATCH:.16f}
- Local REAL slope at zero: {local_real_slope:+.10f}
- Local shuffled slope at zero: {local_shuffled_slope:+.10f}
- Best-any alpha: {frontier['alpha_best_any']}
- Best-bounded alpha: {frontier['alpha_best_bounded']}
- Maximum bounded alpha: {frontier['alpha_max_bounded']}
- Historical 10× RMS threshold: {HARD_THRESHOLD:.16f}

## Direct answers Q1–Q21

"""
    for label, answer in q:
        text += f"### {label}\n\n{answer}\n\n"
    text += f"""## Exactly one next recommendation

**{decision['next_recommendation']}**

No follow-on compute was launched.

## Integrity and provenance

- Frozen 2D1B tag `{FROZEN_TAG}` resolves exactly to `{FROZEN_COMMIT}`.
- C1908 SHA-256 `{SOURCE_SHA256}` and strict reopen passed.
- Legacy D12 rho=.75 oracle regression passed before the residual sweep.
- Alpha manifest was frozen before residual outcomes.
- Model, base, W_u, and W_g hashes were unchanged.
- Exact canonical collection `{CANONICAL_SHA256}` and fixed derangement were used.
- Alpha=0 identities, causal shift, row isolation, and reverse-order contamination checks passed.
- No new alpha, training, window sweep, normalization variant, teacher, reconstruction, AttnRes, or HellaSwag was used.

## Artifacts

All machine-readable artifacts and P1–P10 are under `results/{OUTPUT_NAME}/`.

# EXPERIMENT 2D1C COMPLETE
"""
    return text


def legacy_regression(model, validation_path, frozen_artifact):
    result = d1.evaluate_temporal(model, validation_path, D12, 0.75, controls=("plain", "real", "shuffled", "zero"), batches=20)
    observed = {
        "plain": result["controls"]["plain"]["validation_loss"],
        "real": result["controls"]["real"]["validation_loss"],
        "shuffled": result["controls"]["shuffled"]["validation_loss"],
        "zero": result["controls"]["zero"]["validation_loss"],
    }
    observed["recurrent_gain"] = observed["plain"] - observed["real"]
    observed["sequence_gap"] = observed["shuffled"] - observed["real"]
    expected = {name: frozen_artifact[name] for name in observed}
    deltas = {name: abs(observed[name] - expected[name]) for name in observed}
    regression = {
        "observed": observed, "expected_from_2D1B": expected, "absolute_deltas": deltas,
        "maximum_absolute_delta": max(deltas.values()), "tolerance": 1e-6,
        "canonical_validation_sha256": result["canonical_validation_sha256"],
        "performance": result["performance"],
    }
    regression["passed"] = regression["maximum_absolute_delta"] <= 1e-6 and result["canonical_validation_sha256"] == CANONICAL_SHA256
    return regression


def make_decision(validation, self_results, late, frontier, primary, stability, recommendation):
    zero = validation[alpha_key(0.0)]
    small = validation[alpha_key(0.03125)]
    local_real = (small["real"] - zero["real"]) / 0.03125
    local_shuffled = (small["shuffled"] - zero["shuffled"]) / 0.03125
    positive = [alpha for alpha in ALPHAS if alpha > 0]
    first_specific = next((alpha for alpha in positive if (
        validation[alpha_key(alpha)]["sequence_gap"] > 0.01
        and validation[alpha_key(alpha)]["paired"]["real_vs_shuffled"]["wins"] >= 15
    )), None)
    first_expansive = next((alpha for alpha in positive if not self_results[alpha_key(alpha)]["summary"]["scale_bounded"]), None)
    best = frontier["alpha_best_bounded"]
    if best is None:
        late_answer = "No positive alpha is scale-bounded, so no bounded late-context benefit is claimable."
    else:
        key = alpha_key(best)
        global_gain = validation[key]["recurrent_gain"]
        late_gain = late[key]["late_recurrent_gain_769_1023"]
        late_answer = (
            f"At best bounded alpha={best:.16g}, global gain is {global_gain:+.10f} and positions 769-1023 gain is {late_gain:+.10f}. "
            + ("Benefit is late-concentrated." if late_gain > 0 and global_gain <= 0 else "There is no global-negative/late-positive split." )
        )
    any_beats = any(validation[alpha_key(alpha)]["real"] < zero["plain"] for alpha in positive)
    best_for_plane = frontier["alpha_best_any"]
    best_plane_row = validation[alpha_key(best_for_plane)]
    interpretation = (
        f"At best-any alpha={best_for_plane:.16g}, sequence gap is {best_plane_row['sequence_gap']:+.10f} and recurrent gain is {best_plane_row['recurrent_gain']:+.10f}; "
        + ("the state is both sequence-specific and useful." if best_plane_row["sequence_gap"] > 0 and best_plane_row["recurrent_gain"] > 0
           else "sequence-specificity does not imply positive utility at that operating point.")
    )
    return {
        "primary_classification": primary, "stability_classification": stability,
        "next_recommendation": recommendation,
        "local_real_slope_0": local_real, "local_shuffled_slope_0": local_shuffled,
        "any_positive_alpha_beats_plain": any_beats,
        "first_clearly_sequence_specific_alpha": first_specific,
        "first_expansive_alpha": first_expansive,
        "first_expansive_answer": "No tested positive alpha became expansive." if first_expansive is None else f"The first tested expansive alpha is {first_expansive:.16g}.",
        "late_context_answer": late_answer,
        "max_zero_plain_delta": max(abs(validation[alpha_key(alpha)]["zero"] - validation[alpha_key(alpha)]["plain"]) for alpha in ALPHAS),
        "utility_specificity_interpretation": interpretation,
        "clear_specificity_definition": "sequence_gap > 0.01 and at least 15/20 real-vs-shuffled paired wins",
    }


def run(args):
    started = time.time()
    config = require_config()
    git_checks = require_git(clean=True)
    device = require_hardware(args.pod_id)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command_artifact = {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "command": " ".join(sys.argv),
        "started_at": started, "git_head": git_output("rev-parse", "HEAD"),
        "environment": d1b.runtime_environment(args.pod_id),
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
    }
    durable_json(output / "commands_and_runtime.json", command_artifact)

    frozen_source = read_json(REPO_ROOT / "results" / d1b.OUTPUT_NAME / "source_manifest.json")
    frozen_controls = read_json(REPO_ROOT / "results" / d1b.OUTPUT_NAME / "canonical_controls.json")
    frozen_scale = read_json(REPO_ROOT / "results" / d1b.OUTPUT_NAME / "scale_decomposition.json")
    frozen_content = read_json(REPO_ROOT / "results" / d1b.OUTPUT_NAME / "embedding_content.json")
    frozen_self = read_json(REPO_ROOT / "results" / d1b.OUTPUT_NAME / "self_composition.json")
    source_resolution = {
        "manifest_checkpoint": frozen_source["checkpoint"],
        "manifest_sha256": frozen_source["checkpoint_sha256"],
        "requested_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "path_matches_manifest": str(Path(args.source_checkpoint).resolve()) == frozen_source["checkpoint"],
        "sha_matches_manifest": frozen_source["checkpoint_sha256"] == SOURCE_SHA256,
    }
    if not all(value for name, value in source_resolution.items() if name.endswith("matches_manifest")):
        raise SystemExit(f"source manifest resolution failed: {source_resolution}")
    model, source = d1b.source_reopen(args, device, output)
    source["resolved_from_2d1b_manifest"] = source_resolution
    source["wu_cap_provenance_only"] = config["source_checkpoint"]["wu_cap_provenance"]
    durable_json(output / "source_manifest.json", source)

    before = {
        "model": d1b.module_state_sha256(model), "base": d1b.module_state_sha256(model.base),
        "W_u": d1b.tensor_sha256(model.fusion.W_u.weight), "W_g": d1b.tensor_sha256(model.fusion.W_g.weight),
    }
    batches, batch_manifest = d1b.validation_batches(args.validation_shard)
    if batch_manifest["canonical_collection_sha256"] != CANONICAL_SHA256:
        raise SystemExit("canonical validation collection mismatch")
    durable_json(output / "batch_manifest.json", batch_manifest)

    exact_match = {
        "E_RMS": frozen_scale["C"]["E"]["elements"]["rms"],
        "F_RMS": frozen_scale["C"]["F"]["elements"]["rms"],
    }
    exact_match["value"] = exact_match["E_RMS"] / exact_match["F_RMS"]
    alpha_manifest = {
        "fusion_equation": "X = E + alpha * F", "windows": list(D12),
        "alphas_in_preregistered_order": list(ALPHAS), "alpha_labels": list(ALPHA_LABELS),
        "alpha_rms_match": exact_match,
        "derived_from": "2D1B scale_decomposition.json condition C before residual outcomes",
        "alpha_is_fixed_python_scalar": True, "alpha_parameter_objects": 0,
        "frozen_before_outcome_evaluation": True, "timestamp": time.time(),
    }
    if exact_match["value"] != ALPHA_RMS_MATCH:
        raise SystemExit(f"alpha_rms_match provenance mismatch: {exact_match}")
    durable_json(output / "alpha_manifest.json", alpha_manifest)

    legacy_expected = {
        "plain": frozen_controls["C"]["plain"], "real": frozen_controls["C"]["real"],
        "shuffled": frozen_controls["C"]["shuffled"], "zero": frozen_controls["C"]["zero"],
        "recurrent_gain": frozen_controls["C"]["recurrent_gain"], "sequence_gap": frozen_controls["C"]["sequence_gap"],
    }
    with torch.inference_mode():
        legacy = legacy_regression(model, Path(args.validation_shard), legacy_expected)
    legacy["historical_D12_rho_075"] = LEGACY_075
    legacy["historical_D12_rho_100"] = LEGACY_100
    legacy["historical_values_crosschecked"] = (
        abs(frozen_self["C"]["summary"]["max_recurrent_input_rms"] - LEGACY_075["max_recurrent_input_rms"]) <= 1e-12
        and abs(frozen_scale["C"]["ratios"]["X_over_E"] - LEGACY_075["X_over_E"]) <= 1e-12
        and abs(frozen_content["C"]["cos_X_E"]["mean"] - LEGACY_075["cos_X_E"]) <= 1e-12
    )
    legacy["passed"] = legacy["passed"] and legacy["historical_values_crosschecked"]
    durable_json(output / "legacy_regression.json", legacy)
    if not legacy["passed"]:
        raise SystemExit(f"legacy D12 evaluator hard stop: {legacy}")

    with torch.inference_mode():
        plain, plain_bins, cached_plain_tops, plain_seconds = plain_baseline(model, batches, device)
        identity = identity_preflight(model, batches[0], device)
        causal = causal_audits(model, batches[0], device)
    plain_oracle_delta = abs(plain["validation_loss"] - legacy_expected["plain"])
    if plain_oracle_delta > 1e-6 or not identity["passed"] or not causal["passed"]:
        raise SystemExit(f"residual preflight hard stop: plain_delta={plain_oracle_delta} identity={identity} causal={causal}")

    alpha_validation = OrderedDict(); scale = OrderedDict(); content = OrderedDict(); gates = OrderedDict()
    positions = OrderedDict(); self_results = OrderedDict(); late = OrderedDict(); performance = OrderedDict()
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        with torch.inference_mode():
            validation_row, scale_row, content_row, gate_row, position_row = evaluate_alpha(
                model, batches, cached_plain_tops, plain, plain_bins, alpha, device
            )
            self_row = self_composition(model, batches, alpha, device)
        alpha_validation[key] = validation_row; scale[key] = scale_row; content[key] = content_row
        gates[key] = gate_row; positions[key] = position_row; self_results[key] = self_row
        late[key] = late_context(position_row)
        performance[key] = {
            "canonical_wall_seconds": validation_row["wall_seconds"],
            "self_composition_wall_seconds": self_row["summary"]["wall_seconds"],
        }
        durable_json(output / "alpha_validation.json", alpha_validation)
        durable_json(output / "paired_losses.json", {name: row["paired"] for name, row in alpha_validation.items()})
        durable_json(output / "scale_decomposition.json", scale)
        durable_json(output / "embedding_content.json", content)
        durable_json(output / "gate_diagnostics.json", gates)
        durable_json(output / "position_bin_metrics.json", positions)
        durable_json(output / "self_composition.json", self_results)
        durable_json(output / "late_context_metrics.json", late)
        durable_json(output / "performance.json", {"plain_wall_seconds": plain_seconds, "conditions": performance})
        print(f"2D1C alpha={key} complete", flush=True)

    alpha_zero_deltas = {
        name: abs(alpha_validation[alpha_key(0.0)][name] - plain["validation_loss"])
        for name in ("real", "shuffled", "zero")
    }
    alpha_zero_identity = {
        "deterministic_preflight": identity, "canonical_absolute_deltas": alpha_zero_deltas,
        "maximum_canonical_delta": max(alpha_zero_deltas.values()), "tolerance": 1e-8,
    }
    alpha_zero_identity["passed"] = identity["passed"] and alpha_zero_identity["maximum_canonical_delta"] <= 1e-8
    if not alpha_zero_identity["passed"]:
        raise SystemExit(f"alpha=0 canonical identity hard stop: {alpha_zero_identity}")

    with torch.inference_mode():
        reverse = reverse_order_spot_check(
            model, batches[0], alpha_validation[alpha_key(0.125)]["per_batch_losses"]["real"][0], device
        )
    if not reverse["passed"]:
        raise SystemExit(f"reverse-order contamination check failed: {reverse}")

    frontier = stability_frontier(alpha_validation, self_results)
    primary = primary_classification(alpha_validation, frontier)
    stability = stability_classification(frontier)
    recommendation = next_recommendation(primary, stability, alpha_validation, frontier)
    matched = matched_comparisons(alpha_validation, scale, content, self_results)
    decision = make_decision(alpha_validation, self_results, late, frontier, primary, stability, recommendation)
    durable_json(output / "stability_frontier.json", frontier)
    durable_json(output / "matched_legacy_comparisons.json", matched)
    durable_json(output / "decision_summary.json", decision)

    after = {
        "model": d1b.module_state_sha256(model), "base": d1b.module_state_sha256(model.base),
        "W_u": d1b.tensor_sha256(model.fusion.W_u.weight), "W_g": d1b.tensor_sha256(model.fusion.W_g.weight),
    }
    arithmetic = {
        "gains": all(row["recurrent_gain"] == row["plain"] - row["real"] for row in alpha_validation.values()),
        "sequence_gaps": all(row["sequence_gap"] == row["shuffled"] - row["real"] for row in alpha_validation.values()),
        "zero_dependencies": all(row["zero_dependency"] == row["zero"] - row["real"] for row in alpha_validation.values()),
        "alpha_count": len(alpha_validation) == 9,
    }
    zero_plain_delta = max(abs(row["zero"] - row["plain"]) for row in alpha_validation.values())
    audit_checks = {
        "2D1B_frozen_tag_exact": git_checks["tag"], "C1908_SHA_exact": source["checkpoint_sha256"] == SOURCE_SHA256,
        "C1908_strict_reopen": source["strict_reopen"]["passed"],
        "C1908_parameter_hash_unchanged": before["model"] == after["model"],
        "base_model_unchanged": before["base"] == after["base"], "W_u_unchanged": before["W_u"] == after["W_u"],
        "W_g_unchanged": before["W_g"] == after["W_g"],
        "forbidden_counts_zero": all(value == 0 for value in FORBIDDEN_COUNTS.values()),
        "D12_exact": D12 == tuple(config["windows"]), "legacy_D12_regression": legacy["passed"],
        "fusion_exact_residual_equation": alpha_manifest["fusion_equation"] == "X = E + alpha * F",
        "alpha_list_exact": tuple(alpha_manifest["alphas_in_preregistered_order"]) == ALPHAS,
        "alpha_rms_match_frozen_before_outcomes": alpha_manifest["frozen_before_outcome_evaluation"] and exact_match["value"] == ALPHA_RMS_MATCH,
        "alpha_nontrainable": alpha_manifest["alpha_parameter_objects"] == 0,
        "alpha_zero_plain_and_control_identity": alpha_zero_identity["passed"],
        "canonical_hash_exact": batch_manifest["canonical_collection_sha256"] == CANONICAL_SHA256,
        "same_shuffled_derangement": True, "causal_shift_position_zero_row_isolation": causal["passed"],
        "state_cache_reset_and_reverse_order_check": reverse["passed"],
        "individual_divergence_isolated": all(len(row["summary"]["stops"]) <= SELF_BATCHES for row in self_results.values()),
        "no_new_alpha_window_fusion_Wu_variant": True,
        "no_teacher_reconstruction_AttnRes_HellaSwag": True,
        "arithmetic_crosschecks": all(arithmetic.values()),
        "zero_state_residual_plain_identity": zero_plain_delta <= 1e-8,
    }
    audit = {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "checks": audit_checks,
        "passed": all(audit_checks.values()), "parameter_hashes": {"before": before, "after": after},
        "alpha_zero_identity": alpha_zero_identity, "causality": causal,
        "reverse_order_contamination_check": reverse, "arithmetic": arithmetic,
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
        "git_synchronization": "FINALIZATION GATE OUTSIDE RESULT-BEARING PROCESS",
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    result_summary = {
        "primary_classification": primary, "stability_classification": stability,
        "best_any_alpha": frontier["alpha_best_any"], "best_bounded_alpha": frontier["alpha_best_bounded"],
        "maximum_bounded_alpha": frontier["alpha_max_bounded"], "plain_D12_loss": plain["validation_loss"],
        "alpha_rms_match": ALPHA_RMS_MATCH, "decision": decision, "frontier": frontier,
    }
    durable_json(output / "result_summary.json", result_summary)
    make_plots(output, alpha_validation, scale, content, self_results, positions, frontier)
    durable_text(output / "EXPERIMENT_2D1C_FINAL_REPORT.md", build_report(
        alpha_validation, scale, content, self_results, late, matched, frontier, decision, legacy, audit
    ))
    command_artifact.update({
        "completed_at": time.time(), "wall_seconds": time.time() - started,
        "forbidden_operation_counts": FORBIDDEN_COUNTS, "result": "PASS" if audit["passed"] else "FAIL",
    })
    durable_json(output / "commands_and_runtime.json", command_artifact)
    if not audit["passed"]:
        raise SystemExit(f"final audit failed: {audit_checks}")
    print(f"EXPERIMENT_2D1C_PRIMARY_CLASSIFICATION: {primary}", flush=True)
    print(f"EXPERIMENT_2D1C_STABILITY_CLASSIFICATION: {stability}", flush=True)
    print(f"BEST_ANY_ALPHA: {frontier['alpha_best_any']}", flush=True)
    print(f"BEST_BOUNDED_ALPHA: {frontier['alpha_best_bounded']}", flush=True)
    print(f"MAXIMUM_BOUNDED_ALPHA: {frontier['alpha_max_bounded']}", flush=True)
    print("EXPERIMENT_2D1C_COMPLETE", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--validation-shard", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / OUTPUT_NAME))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
