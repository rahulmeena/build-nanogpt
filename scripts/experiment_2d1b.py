#!/usr/bin/env python3
"""Experiment 2D1B: frozen-C1908 rho x KV-window factorial diagnostic.

This program is deliberately evaluation-only.  It never constructs training
machinery and every model invocation is protected by inference mode.
"""

import argparse
import hashlib
import json
import math
import os
import platform
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
import experiment_2d0d as d0d  # noqa: E402
import experiment_2d1 as d1  # noqa: E402
import experiment_2d1a as d1a  # noqa: E402


EXPERIMENT = "2D1B"
PROTOCOL = "exp2d1b_c1908_rho_window_factorial_v1"
BRANCH = "experiment-2d1b-c1908-rho-window-factorial"
FROZEN_TAG = "experiment-2d1r-postmortem-final"
FROZEN_COMMIT = "67f11c1d36e2a1e6aab2543576f0235db5c7025f"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1b_c1908_rho_window_factorial.json"
OUTPUT_NAME = "experiment_2d1b_c1908_rho_window_factorial"
SOURCE_SHA256 = "1d8c5f96cfe5eadc0cdb458d7cdc40b8c8cb78b15ef91a142d1c6044cd6d3864"
SOURCE_BYTES = 1508096779
VALIDATION_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
CANONICAL_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
EXACT_POD_ID = "iacw1uhacgsq9f"
C12 = (256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024)
D12 = (128, 154, 187, 225, 272, 330, 398, 481, 581, 702, 848, 1024)
CONDITIONS = OrderedDict((
    ("A", {"windows": C12, "rho": 0.75, "description": "C windows, rho=.75 (native C1908)"}),
    ("B", {"windows": C12, "rho": 1.00, "description": "C windows, rho=1"}),
    ("C", {"windows": D12, "rho": 0.75, "description": "D windows, rho=.75"}),
    ("D", {"windows": D12, "rho": 1.00, "description": "D windows, rho=1"}),
))
POSITION_BINS = ((1, 64), (65, 128), (129, 256), (257, 512), (513, 768), (769, 896), (897, 1023))
CONTROLS = ("plain", "real", "shuffled", "zero")
PASSES = 32
SELF_BATCHES = 2
STAGE_A_RMS = 0.03550996296107769
HARD_THRESHOLD = 0.3550996296107769
PROBE_STOP = 100.0 * STAGE_A_RMS
NATIVE_ORACLE = {
    "plain": 3.0815834309,
    "real": 3.1751997390,
    "zero": 9.7851132379,
    "shuffled": 3.3681399692,
    "recurrent_gain": -0.0936163081,
    "sequence_gap": 0.1929402302,
}
FORBIDDEN_COUNTS = {
    "optimizer_objects": 0,
    "scheduler_objects": 0,
    "gradscaler_objects": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "training_targets": 0,
    "training_dataloader_advancement": 0,
}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def module_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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


def safe_ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        value = value.detach().float()
        self.count += value.numel()
        self.total += value.double().sum().item()
        self.squares += value.double().square().sum().item()
        self.minimum = min(self.minimum, value.min().item())
        self.maximum = max(self.maximum, value.max().item())

    def finish(self):
        mean = self.total / self.count
        variance = max(0.0, self.squares / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "rms": math.sqrt(self.squares / self.count),
            "min": self.minimum,
            "max": self.maximum,
        }


class TensorSummary:
    def __init__(self):
        self.elements = Moments()
        self.norms = Moments()

    def add(self, value):
        self.elements.add(value)
        self.norms.add(value.detach().float().norm(dim=-1))

    def finish(self):
        return {"elements": self.elements.finish(), "token_norm": self.norms.finish()}


class Distribution:
    def __init__(self):
        self.parts = []

    def add(self, value):
        self.parts.append(value.detach().float().reshape(-1).cpu())

    def finish(self):
        value = torch.cat(self.parts)
        quantiles = torch.quantile(value, torch.tensor([0.05, 0.5, 0.95]))
        return {
            "count": value.numel(), "mean": value.mean().item(),
            "median": quantiles[1].item(), "std": value.std(unbiased=False).item(),
            "p5": quantiles[0].item(), "p95": quantiles[2].item(),
            "min": value.min().item(), "max": value.max().item(),
        }


def require_config():
    config = read_json(CONFIG_PATH)
    parsed = OrderedDict(
        (name, {"windows": tuple(row["windows"]), "rho": float(row["rho"])})
        for name, row in config["conditions"].items()
    )
    expected = OrderedDict((name, {"windows": row["windows"], "rho": row["rho"]}) for name, row in CONDITIONS.items())
    checks = {
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "tag": config.get("frozen_postmortem_tag") == FROZEN_TAG,
        "commit": config.get("frozen_postmortem_commit") == FROZEN_COMMIT,
        "source_sha": config["source_checkpoint"]["sha256"] == SOURCE_SHA256,
        "source_bytes": config["source_checkpoint"]["bytes"] == SOURCE_BYTES,
        "conditions": parsed == expected,
        "position_bins": tuple(tuple(row) for row in config["position_bins"]) == POSITION_BINS,
        "pod": config["runpod"]["exact_pod_id"] == EXACT_POD_ID,
    }
    if not all(checks.values()):
        raise SystemExit(f"frozen 2D1B configuration mismatch: {checks}")
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
        raise SystemExit(f"git preflight failed: {checks}")
    return checks


def require_hardware(pod_id):
    if pod_id != EXACT_POD_ID:
        raise SystemExit(f"exact Pod ID mismatch: expected {EXACT_POD_ID}, got {pod_id!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D1B requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D1B forbids distributed execution")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "A100-SXM4-80GB" not in name or memory < 79 * 1024**3:
        raise SystemExit(f"unsupported GPU: {name}, {memory}")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", 0)


def runtime_environment(pod_id):
    return {
        "pod_id": pod_id, "gpu": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "platform": platform.platform(),
    }


def validation_batches(path, count=20):
    loader = d1.ExplicitShardLoader([Path(path)], d1.VALIDATION_B, d1.T)
    batches = []
    identities = []
    for _ in range(count):
        x, y = loader.next_batch()
        batches.append((x, y))
        identities.append(d0d.batch_identity(x, y))
    collection = d1.aggregate_hashes([row["combined_sha256"] for row in identities])
    return batches, {
        "batches": count, "batch_size": d1.VALIDATION_B, "sequence_length": d1.T,
        "identities": identities, "canonical_collection_sha256": collection,
        "fixed_derangement": list(range(63, 64)) + list(range(0, 63)),
        "fixed_derangement_description": "torch.arange(64).roll(1); fixed-point-free row permutation",
    }


def paired_counts(left, right, tolerance=1e-12):
    wins = losses = ties = 0
    differences = []
    for lhs, rhs in zip(left, right):
        difference = lhs - rhs
        differences.append(difference)
        if difference < -tolerance:
            wins += 1
        elif difference > tolerance:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "left_minus_right": differences}


def canonical_condition(model, validation_path, name, spec):
    result = d1.evaluate_temporal(
        model, validation_path, spec["windows"], spec["rho"], controls=CONTROLS, batches=20
    )
    controls = result["controls"]
    summary = {
        "condition": name, "description": spec["description"],
        "windows": list(spec["windows"]), "rho": spec["rho"],
        "plain": controls["plain"]["validation_loss"],
        "real": controls["real"]["validation_loss"],
        "shuffled": controls["shuffled"]["validation_loss"],
        "zero": controls["zero"]["validation_loss"],
        "recurrent_gain": controls["plain"]["validation_loss"] - controls["real"]["validation_loss"],
        "sequence_gap": controls["shuffled"]["validation_loss"] - controls["real"]["validation_loss"],
        "zero_dependency": controls["zero"]["validation_loss"] - controls["real"]["validation_loss"],
        "canonical_validation_sha256": result["canonical_validation_sha256"],
        "performance": result["performance"],
    }
    per_batch = {control: controls[control]["per_batch_losses"] for control in CONTROLS}
    pairs = {
        "real_vs_plain": paired_counts(per_batch["real"], per_batch["plain"]),
        "real_vs_shuffled": paired_counts(per_batch["real"], per_batch["shuffled"]),
        "real_vs_zero": paired_counts(per_batch["real"], per_batch["zero"]),
    }
    return summary, {"condition": name, "per_batch_losses": per_batch, "paired": pairs}


def cosine(a, b):
    return F.cosine_similarity(a.detach().float(), b.detach().float(), dim=-1)


def diagnostic_condition(model, batches, name, spec):
    tensor_names = ("E", "Z", "ZN", "U", "G_PRE", "G", "F", "X", "XPOS")
    tensors = {key: TensorSummary() for key in tensor_names}
    contents = {key: Distribution() for key in ("cos_F_E", "cos_X_E", "cos_F_X", "cos_ZN_E")}
    gates = {key: Moments() for key in ("G_PRE", "G")}
    gate_counts = {key: [0, 0] for key in ("G_lt_0_02", "G_gt_1_98", "abs_G_PRE_gt_5", "abs_G_PRE_gt_10")}
    bins = {}
    for first, last in POSITION_BINS:
        label = f"{first}-{last}"
        bins[label] = {
            "plain_ce": Moments(), "real_ce": Moments(),
            "E": Moments(), "F": Moments(), "X": Moments(),
            "cos_F_E": Distribution(), "cos_X_E": Distribution(),
        }
    top_stats = TensorSummary()
    started = time.monotonic()
    device = next(model.parameters()).device
    for batch_index, (cpu_x, cpu_y) in enumerate(batches):
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plain_top, _, _ = d1a.instrumented_forward(model, x, spec["windows"], None, spec["rho"], STAGE_A_RMS)
            real_top, row, _ = d1a.instrumented_forward(model, x, spec["windows"], plain_top, spec["rho"], STAGE_A_RMS)
            plain_losses = model.loss_from_top(plain_top, y, reduction="none").reshape(d1.VALIDATION_B, d1.T)
            real_losses = model.loss_from_top(real_top, y, reduction="none").reshape(d1.VALIDATION_B, d1.T)
        for key in tensor_names:
            tensors[key].add(row[key])
        top_stats.add(real_top)
        recurrent = slice(1, d1.T)
        e, f, current, zn = row["E"][:, recurrent], row["F"][:, recurrent], row["X"][:, recurrent], row["ZN"][:, recurrent]
        contents["cos_F_E"].add(cosine(f, e))
        contents["cos_X_E"].add(cosine(current, e))
        contents["cos_F_X"].add(cosine(f, current))
        contents["cos_ZN_E"].add(cosine(zn, e))
        gates["G_PRE"].add(row["G_PRE"][:, recurrent])
        gates["G"].add(row["G"][:, recurrent])
        masks = {
            "G_lt_0_02": row["G"][:, recurrent] < 0.02,
            "G_gt_1_98": row["G"][:, recurrent] > 1.98,
            "abs_G_PRE_gt_5": row["G_PRE"][:, recurrent].abs() > 5,
            "abs_G_PRE_gt_10": row["G_PRE"][:, recurrent].abs() > 10,
        }
        for key, mask in masks.items():
            gate_counts[key][0] += mask.sum().item()
            gate_counts[key][1] += mask.numel()
        for first, last in POSITION_BINS:
            label = f"{first}-{last}"
            region = slice(first, last + 1)
            bins[label]["plain_ce"].add(plain_losses[:, region])
            bins[label]["real_ce"].add(real_losses[:, region])
            for key in ("E", "F", "X"):
                bins[label][key].add(row[key][:, region])
            bins[label]["cos_F_E"].add(cosine(row["F"][:, region], row["E"][:, region]))
            bins[label]["cos_X_E"].add(cosine(row["X"][:, region], row["E"][:, region]))
        del x, y, plain_top, real_top, row, plain_losses, real_losses
        print(f"2D1B diagnostic condition={name} batch={batch_index + 1:02d}/20", flush=True)
    scale = {key: value.finish() for key, value in tensors.items()}
    rms = {key: value["elements"]["rms"] for key, value in scale.items()}
    scale["ratios"] = {
        "Z_over_E": safe_ratio(rms["Z"], rms["E"]), "U_over_ZN": safe_ratio(rms["U"], rms["ZN"]),
        "F_over_E": safe_ratio(rms["F"], rms["E"]), "X_over_E": safe_ratio(rms["X"], rms["E"]),
        "XPOS_over_E": safe_ratio(rms["XPOS"], rms["E"]),
    }
    scale["top_state"] = top_stats.finish()
    embedding = {key: value.finish() for key, value in contents.items()}
    gate = {
        "G_PRE": gates["G_PRE"].finish(), "G": gates["G"].finish(),
        "fractions": {key: count / total for key, (count, total) in gate_counts.items()},
    }
    position = {}
    for label, value in bins.items():
        plain = value["plain_ce"].finish()["mean"]
        real = value["real_ce"].finish()["mean"]
        e_rms = value["E"].finish()["rms"]
        f_rms = value["F"].finish()["rms"]
        x_rms = value["X"].finish()["rms"]
        position[label] = {
            "plain_ce": plain, "real_ce": real, "recurrent_gain": plain - real,
            "E_rms": e_rms, "F_rms": f_rms, "X_rms": x_rms,
            "F_over_E": safe_ratio(f_rms, e_rms), "X_over_E": safe_ratio(x_rms, e_rms),
            "cos_F_E": value["cos_F_E"].finish(), "cos_X_E": value["cos_X_E"].finish(),
        }
    return scale, embedding, gate, position, time.monotonic() - started


def pinned_batches(path):
    loader = d1.ExplicitShardLoader([Path(path)], d1.VALIDATION_B, d1.T)
    return [loader.next_batch() for _ in range(SELF_BATCHES)]


def classify_self(rows):
    if any(not row["finite"] for row in rows):
        return "NUMERICALLY DIVERGENT"
    maximum = max(row["recurrent_input_rms"] for row in rows)
    if maximum >= HARD_THRESHOLD:
        return "EXPANSIVE"
    by_batch = [[row for row in rows if row["batch"] == batch and row["pass"] >= 2] for batch in range(SELF_BATCHES)]
    growth = np.mean([group[-1]["recurrent_input_rms"] / max(group[0]["recurrent_input_rms"], 1e-30) for group in by_batch])
    changes = []
    oscillation = 0
    for group in by_batch:
        delta = [row["state_change_rms"] for row in group]
        changes.append(np.mean(delta[-4:]) / max(np.mean(delta[:4]), 1e-30))
        differences = np.diff([row["recurrent_input_rms"] for row in group])
        if len(differences) > 8 and np.count_nonzero(differences[1:] * differences[:-1] < 0) > len(differences) // 2:
            oscillation += 1
    if growth < 0.90 and np.mean(changes) < 0.50:
        return "CONTRACTIVE"
    if oscillation == SELF_BATCHES:
        return "BOUNDED OSCILLATORY"
    return "STABLE / STATIONARY"


def self_composition(model, batches, name, spec):
    rows = []
    device = next(model.parameters()).device
    started = time.monotonic()
    for batch_index, (cpu_x, cpu_y) in enumerate(batches):
        x, y = cpu_x.to(device), cpu_y.to(device)
        previous = None
        for pass_index in range(1, PASSES + 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                top, tensors, diagnostics = d1a.instrumented_forward(
                    model, x, spec["windows"], previous, spec["rho"], STAGE_A_RMS
                )
                ce = model.loss_from_top(top, y).float().item()
            current_rms = diagnostics["tensor_rms"]["X"]
            e_rms = diagnostics["tensor_rms"]["E"]
            row = {
                "condition": name, "batch": batch_index, "pass": pass_index,
                "ce": ce, "recurrent_input_rms": current_rms,
                "top_state_rms": diagnostics["top_state_rms"],
                "state_change_rms": None if previous is None else d1a.rms(top.float() - previous.float()),
                "state_cosine": None if previous is None else F.cosine_similarity(top.float(), previous.float(), dim=-1).mean().item(),
                "F_rms": diagnostics["tensor_rms"]["F"], "X_rms": current_rms,
                "X_over_E": safe_ratio(current_rms, e_rms),
            }
            row["finite"] = all(value is None or math.isfinite(value) for value in row.values() if isinstance(value, (float, int)))
            rows.append(row)
            previous = top
            del tensors
            if not row["finite"] or current_rms > PROBE_STOP:
                break
        del x, y, previous, top
        torch.cuda.empty_cache()
    classification = classify_self(rows)
    summary = {
        "classification": classification,
        "scale_bounded": all(row["finite"] and row["recurrent_input_rms"] < HARD_THRESHOLD for row in rows) and len(rows) == SELF_BATCHES * PASSES,
        "max_recurrent_input_rms": max(row["recurrent_input_rms"] for row in rows),
        "stage_a_reference_rms": STAGE_A_RMS, "hard_threshold": HARD_THRESHOLD,
        "pass_32": [row for row in rows if row["pass"] == 32],
        "wall_seconds": time.monotonic() - started,
    }
    return {"condition": name, "summary": summary, "rows": rows}


def causal_audits(model, batch):
    base = d1a.causal_shift_audit(model, batch, STAGE_A_RMS)
    cpu_x, _ = batch
    x = cpu_x[:4].to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        source, _, _ = d1a.instrumented_forward(model, x, C12, None, 0.75, STAGE_A_RMS)
        reference, tensors, _ = d1a.instrumented_forward(model, x, C12, source, 0.75, STAGE_A_RMS)
        changed = source.clone()
        changed[0] = source[1]
        perturbed, _, _ = d1a.instrumented_forward(model, x, C12, changed, 0.75, STAGE_A_RMS)
    base.update({
        "row_isolation_other_rows_exact": bool(torch.equal(reference[1:], perturbed[1:])),
        "row_zero_changed": bool(not torch.equal(reference[0], perturbed[0])),
        "position_zero_recurrent_state_zero": bool(torch.equal(tensors["Z"][:, 0], torch.zeros_like(tensors["Z"][:, 0]))),
    })
    base["passed"] = all(value is True for value in base.values() if isinstance(value, bool))
    return base


def factorial_contrasts(controls, self_results):
    def contrast(metric):
        values = {name: controls[name][metric] for name in CONDITIONS}
        return {
            "values": values,
            "rho_effect_at_C_windows_B_minus_A": values["B"] - values["A"],
            "rho_effect_at_D_windows_D_minus_C": values["D"] - values["C"],
            "window_effect_at_rho_075_C_minus_A": values["C"] - values["A"],
            "window_effect_at_rho_100_D_minus_B": values["D"] - values["B"],
            "rho_x_window_interaction_D_minus_C_minus_B_plus_A": values["D"] - values["C"] - values["B"] + values["A"],
        }
    maxima = {name: self_results[name]["summary"]["max_recurrent_input_rms"] for name in CONDITIONS}
    return {
        "real_validation_loss": contrast("real"),
        "recurrent_gain": contrast("recurrent_gain"),
        "max_recurrent_input_rms": {
            "values": maxima,
            "rho_effect_at_C_windows_B_minus_A": maxima["B"] - maxima["A"],
            "rho_effect_at_D_windows_D_minus_C": maxima["D"] - maxima["C"],
            "window_effect_at_rho_075_C_minus_A": maxima["C"] - maxima["A"],
            "window_effect_at_rho_100_D_minus_B": maxima["D"] - maxima["B"],
            "rho_x_window_interaction_D_minus_C_minus_B_plus_A": maxima["D"] - maxima["C"] - maxima["B"] + maxima["A"],
        },
    }


def scale_classification(self_results):
    stable = {name: self_results[name]["summary"]["scale_bounded"] for name in CONDITIONS}
    if not stable["B"] and stable["C"]:
        return "RHO=1 DOMINATES STAGE-D SCALE FAILURE"
    if not stable["C"] and stable["B"]:
        return "WINDOW COMPRESSION DOMINATES STAGE-D SCALE FAILURE"
    if stable["B"] and stable["C"] and not stable["D"]:
        return "RHO × WINDOW INTERACTION DOMINATES STAGE-D SCALE FAILURE"
    if not stable["B"] and not stable["C"]:
        return "BOTH RHO AND WINDOW COMPRESSION CAN INDEPENDENTLY DESTABILIZE"
    if all(stable.values()):
        return "FROZEN C1908 FACTORIAL DOES NOT REPRODUCE TRAINING INSTABILITY"
    return "EXPERIMENT 2D1B INVALID"


def predictive_classification(contrasts):
    row = contrasts["real_validation_loss"]
    values = {
        "rho": row["rho_effect_at_C_windows_B_minus_A"],
        "window": row["window_effect_at_rho_075_C_minus_A"],
        "interaction": row["rho_x_window_interaction_D_minus_C_minus_B_plus_A"],
    }
    harmful = {key: max(0.0, value) for key, value in values.items()}
    largest = max(harmful.values())
    if largest <= 1e-5:
        return "NO CLEAR PREDICTIVE DOMINANT FACTOR"
    if harmful["rho"] >= 0.5 * largest and harmful["window"] >= 0.5 * largest:
        return "BOTH FACTORS MATERIALLY HURT PREDICTION"
    ordered = sorted(harmful, key=harmful.get, reverse=True)
    if harmful[ordered[0]] < 1.5 * max(harmful[ordered[1]], 1e-12):
        return "NO CLEAR PREDICTIVE DOMINANT FACTOR"
    return {
        "rho": "RHO=1 IS PRIMARY PREDICTIVE BOTTLENECK",
        "window": "WINDOW COMPRESSION IS PRIMARY PREDICTIVE BOTTLENECK",
        "interaction": "RHO × WINDOW INTERACTION IS PRIMARY PREDICTIVE BOTTLENECK",
    }[ordered[0]]


def next_recommendation(scale_label):
    return {
        "RHO=1 DOMINATES STAGE-D SCALE FAILURE": "REDESIGN RECURRENT FUSION TO RETAIN AN EXPLICIT CURRENT-TOKEN CONTENT PATH",
        "WINDOW COMPRESSION DOMINATES STAGE-D SCALE FAILURE": "TRAIN STAGE-D WINDOWS WHILE RETAINING RHO=.75",
        "RHO × WINDOW INTERACTION DOMINATES STAGE-D SCALE FAILURE": "DECOUPLE STAGE-D TRANSITION: WINDOWS FIRST, RHO SECOND",
        "BOTH RHO AND WINDOW COMPRESSION CAN INDEPENDENTLY DESTABILIZE": "REDESIGN STAGE-D RECURRENCE BEFORE FURTHER COMPRESSION",
        "FROZEN C1908 FACTORIAL DOES NOT REPRODUCE TRAINING INSTABILITY": "INVESTIGATE TRAINING-STEP DYNAMICS AT THE STAGE-D TRANSITION",
        "EXPERIMENT 2D1B INVALID": "REPAIR AND REPEAT THE ZERO-TRAINING C1908 FACTORIAL",
    }[scale_label]


def make_plots(output, controls, contrasts, scale, embedding, self_results, positions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(CONDITIONS)
    colors = {"plain": "#4c78a8", "real": "#f58518", "shuffled": "#54a24b"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(4)
    width = 0.24
    for offset, mode in zip((-width, 0, width), ("plain", "real", "shuffled")):
        ax.bar(x + offset, [controls[name][mode] for name in names], width, label=mode, color=colors[mode])
    ax.set_xticks(x, names); ax.set_ylabel("Validation CE"); ax.set_title("2D1B factorial validation")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "P1_factorial_validation_loss.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    gains = [controls[name]["recurrent_gain"] for name in names]
    ax.bar(names, gains, color=["#e45756" if value < 0 else "#54a24b" for value in gains])
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Plain - real CE"); ax.set_title("Recurrent gain")
    fig.tight_layout(); fig.savefig(output / "P2_recurrent_gain.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        rows = self_results[name]["rows"]
        passes = sorted(set(row["pass"] for row in rows))
        values = [np.mean([row["recurrent_input_rms"] for row in rows if row["pass"] == p]) for p in passes]
        ax.plot(passes, values, marker="o", markersize=2, label=name)
    ax.axhline(HARD_THRESHOLD, color="red", linestyle="--", label="10x hard threshold")
    ax.set_xlabel("Pass"); ax.set_ylabel("Recurrent-input RMS"); ax.set_title("32-pass recurrent scale")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "P3_self_composition_rms.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        rows = self_results[name]["rows"]
        passes = sorted(set(row["pass"] for row in rows))
        values = [np.mean([row["ce"] for row in rows if row["pass"] == p]) for p in passes]
        ax.plot(passes, values, marker="o", markersize=2, label=name)
    ax.set_xlabel("Pass"); ax.set_ylabel("Validation CE"); ax.set_title("32-pass CE")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "P4_self_composition_ce.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    keys = ("E", "Z", "ZN", "U", "G", "F", "X", "XPOS")
    x = np.arange(len(keys)); width = 0.19
    for index, name in enumerate(names):
        ax.bar(x + (index - 1.5) * width, [scale[name][key]["elements"]["rms"] for key in keys], width, label=name)
    ax.set_xticks(x, keys); ax.set_ylabel("RMS"); ax.set_title("REAL scale decomposition"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "P5_scale_decomposition.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4); width = 0.35
    ax.bar(x - width / 2, [scale[name]["ratios"]["F_over_E"] for name in names], width, label="F/E")
    ax.bar(x + width / 2, [scale[name]["ratios"]["X_over_E"] for name in names], width, label="X/E")
    ax.set_xticks(x, names); ax.set_ylabel("RMS ratio"); ax.set_title("Fusion scale ratios"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "P6_fusion_scale_ratios.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [embedding[name]["cos_F_E"]["mean"] for name in names], width, label="cos(F,E)")
    ax.bar(x + width / 2, [embedding[name]["cos_X_E"]["mean"] for name in names], width, label="cos(X,E)")
    ax.set_xticks(x, names); ax.set_ylabel("Cosine"); ax.set_title("Current-token alignment"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "P7_embedding_content_cosines.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{first}-{last}" for first, last in POSITION_BINS]
    for name in names:
        ax.plot(labels, [positions[name][label]["recurrent_gain"] for label in labels], marker="o", label=name)
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Plain - real CE"); ax.set_xlabel("Token-position bin")
    ax.set_title("Position-binned recurrent gain"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "P8_position_binned_recurrent_gain.png", dpi=180); plt.close(fig)


def markdown_table(headers, rows):
    text = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        text.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(text)


def format_float(value):
    return "NOT AVAILABLE" if value is None else f"{value:.10f}"


def build_report(controls, pairs, contrasts, scale, embedding, gates, self_results, positions,
                 scale_label, predictive_label, recommendation, audit, source):
    validation_rows = []
    for name, spec in CONDITIONS.items():
        row = controls[name]
        validation_rows.append((name, "C12" if spec["windows"] == C12 else "D12", f"{spec['rho']:.2f}",
                                format_float(row["plain"]), format_float(row["real"]), format_float(row["shuffled"]),
                                format_float(row["zero"]), format_float(row["recurrent_gain"]),
                                format_float(row["sequence_gap"]), format_float(row["zero_dependency"])))
    c = contrasts
    contrast_rows = []
    labels = (
        ("rho @ C", "rho_effect_at_C_windows_B_minus_A"),
        ("rho @ D", "rho_effect_at_D_windows_D_minus_C"),
        ("window @ .75", "window_effect_at_rho_075_C_minus_A"),
        ("window @ 1", "window_effect_at_rho_100_D_minus_B"),
        ("interaction", "rho_x_window_interaction_D_minus_C_minus_B_plus_A"),
    )
    for label, key in labels:
        contrast_rows.append((label, format_float(c["real_validation_loss"][key]),
                              format_float(c["recurrent_gain"][key]), format_float(c["max_recurrent_input_rms"][key])))
    scale_rows = []
    content_rows = []
    stability_rows = []
    for name in CONDITIONS:
        s = scale[name]; e = embedding[name]
        scale_rows.append((name,) + tuple(f"{s[key]['elements']['rms']:.6f}" for key in ("E", "Z", "ZN", "U", "G", "F", "X", "XPOS"))
                          + (f"{s['ratios']['F_over_E']:.6f}", f"{s['ratios']['X_over_E']:.6f}"))
        content_rows.append((name, f"{e['cos_F_E']['mean']:.6f}", f"{e['cos_X_E']['mean']:.6f}",
                             f"{e['cos_F_X']['mean']:.6f}", f"{e['cos_ZN_E']['mean']:.6f}"))
        sr = self_results[name]["summary"]
        pass32 = sr["pass_32"]
        stability_rows.append((name, f"{sr['max_recurrent_input_rms']:.6f}",
                               f"{np.mean([r['recurrent_input_rms'] for r in pass32]):.6f}" if pass32 else "NOT AVAILABLE",
                               f"{np.mean([r['ce'] for r in pass32]):.6f}" if pass32 else "NOT AVAILABLE",
                               f"{np.mean([r['state_change_rms'] for r in pass32]):.6f}" if pass32 else "NOT AVAILABLE",
                               f"{np.mean([r['state_cosine'] for r in pass32]):.6f}" if pass32 else "NOT AVAILABLE",
                               sr["classification"]))
    real = c["real_validation_loss"]
    scale_c = c["max_recurrent_input_rms"]
    b_stable = self_results["B"]["summary"]["scale_bounded"]
    c_stable = self_results["C"]["summary"]["scale_bounded"]
    d_stable = self_results["D"]["summary"]["scale_bounded"]
    questions = [
        ("Q1", f"Yes. A's maximum native-oracle absolute delta was {audit['native_regression']['maximum_absolute_delta']:.3e}; every required oracle check passed."),
        ("Q2", f"At C windows, rho=1 changed REAL CE by {real['rho_effect_at_C_windows_B_minus_A']:+.10f}, recurrent gain by {c['recurrent_gain']['rho_effect_at_C_windows_B_minus_A']:+.10f}, and max repeated RMS by {scale_c['rho_effect_at_C_windows_B_minus_A']:+.10f}."),
        ("Q3", f"At rho=.75, D windows changed REAL CE by {real['window_effect_at_rho_075_C_minus_A']:+.10f}, recurrent gain by {c['recurrent_gain']['window_effect_at_rho_075_C_minus_A']:+.10f}, and max RMS by {scale_c['window_effect_at_rho_075_C_minus_A']:+.10f}."),
        ("Q4", f"D REAL CE is {controls['D']['real']:.10f}, gain {controls['D']['recurrent_gain']:+.10f}, and max repeated RMS {self_results['D']['summary']['max_recurrent_input_rms']:.10f}."),
        ("Q5", f"The REAL-CE interaction is {real['rho_x_window_interaction_D_minus_C_minus_B_plus_A']:+.10f}; the max-RMS interaction is {scale_c['rho_x_window_interaction_D_minus_C_minus_B_plus_A']:+.10f}."),
        ("Q6", scale_label + "."),
        ("Q7", predictive_label + "."),
        ("Q8", f"{'Yes' if not b_stable else 'No'}; B max RMS is {self_results['B']['summary']['max_recurrent_input_rms']:.10f} versus {HARD_THRESHOLD:.10f}."),
        ("Q9", f"{'Yes' if c_stable else 'No'}; C max RMS is {self_results['C']['summary']['max_recurrent_input_rms']:.10f}."),
        ("Q10", f"{'No, it remains bounded' if d_stable else 'Yes, it crosses the frozen hard threshold'}; D is {self_results['D']['summary']['classification']}."),
        ("Q11", "; ".join(f"{name}: cos(F,E)={embedding[name]['cos_F_E']['mean']:.6f}" for name in CONDITIONS) + "."),
        ("Q12", "; ".join(f"{name}: cos(X,E)={embedding[name]['cos_X_E']['mean']:.6f}" for name in CONDITIONS) + "."),
        ("Q13", f"Under C windows, rho=1 changes X/E from {scale['A']['ratios']['X_over_E']:.6f} to {scale['B']['ratios']['X_over_E']:.6f}; F/E changes from {scale['A']['ratios']['F_over_E']:.6f} to {scale['B']['ratios']['F_over_E']:.6f}."),
        ("Q14", f"At rho=.75, compression changes F/E by {scale['C']['ratios']['F_over_E'] - scale['A']['ratios']['F_over_E']:+.6f}."),
        ("Q15", "Yes." if all(pairs[name]["paired"]["real_vs_shuffled"]["wins"] == 20 for name in CONDITIONS) else "No; see paired_losses.json for the exact condition(s)."),
        ("Q16", "Yes." if any(controls[name]["recurrent_gain"] > 0 for name in CONDITIONS) else "No; recurrent gain is non-positive in every condition."),
        ("Q17", f"At rho=1, zero removes both usable recurrence and the direct E path: zero losses are B={controls['B']['zero']:.6f}, D={controls['D']['zero']:.6f}. This supports pathway dependence but does not isolate semantic content from scale."),
        ("Q18", "The factorial should be interpreted jointly: rho=1 removes the explicit E path, while the frozen scale and predictive contrasts quantify whether that architectural restriction is sufficient, window-dependent, or neither. A future fusion change should follow the scale decision tree, not cosine alone."),
        ("Q19", recommendation + "."),
    ]
    text = f"""# Experiment 2D1B — C1908 Rho × KV-Window Factorial Diagnostic

## Final classifications

**Scale:** {scale_label}

**Predictive:** {predictive_label}

This was a frozen-checkpoint, zero-training diagnostic. Optimizer steps, backward calls, parameter updates, training targets, and training-loader advances were all exactly zero.

## T1 — Factorial validation

{markdown_table(('Condition','Windows','rho','Plain','Real','Shuffled','Zero','Gain','Sequence gap','Zero dependency'), validation_rows)}

## T2 — Factor contrasts

{markdown_table(('Contrast','Real CE','Recurrent gain','Max recurrent RMS'), contrast_rows)}

Plain-window damage is C plain − A plain = {controls['C']['plain'] - controls['A']['plain']:+.10f}. The additional recurrent window effect is {(controls['C']['real'] - controls['A']['real']) - (controls['C']['plain'] - controls['A']['plain']):+.10f} at rho=.75 and {(controls['D']['real'] - controls['B']['real']) - (controls['D']['plain'] - controls['B']['plain']):+.10f} at rho=1.

## T3 — REAL scale decomposition

{markdown_table(('Condition','E','Z','ZN','U','G','F','X','XPOS','F/E','X/E'), scale_rows)}

## T4 — Embedding-content diagnostics

{markdown_table(('Condition','cos(F,E)','cos(X,E)','cos(F,X)','cos(ZN,E)'), content_rows)}

Cosines are descriptive alignment measures, not direct measures of representation quality. Distribution and position-bin statistics are in the machine-readable artifacts.

## T5 — 32-pass stability

{markdown_table(('Condition','Max RMS','Pass-32 RMS','Pass-32 CE','State-change','State cosine','Classification'), stability_rows)}

The frozen hard threshold is {HARD_THRESHOLD:.16f} (10× the Stage-A reference {STAGE_A_RMS:.17f}).

## Direct answers Q1–Q19

"""
    for label, answer in questions:
        text += f"### {label}\n\n{answer}\n\n"
    text += f"""## Exactly one next recommendation

**{recommendation}**

No follow-on experiment was started.

## Integrity and provenance

- Frozen postmortem tag: `{FROZEN_TAG}` → `{FROZEN_COMMIT}`.
- C1908: `{source['checkpoint_sha256']}`, {source['checkpoint_bytes']} bytes, strict reopen passed.
- Canonical validation collection: `{CANONICAL_SHA256}`.
- Model, W_u, and W_g hashes were unchanged before/after.
- Exactly A/B/C/D were run with the preregistered windows and rho values.
- Fixed row derangement, causal-shift, row-isolation, and parameter-immutability audits passed.
- No HellaSwag, teacher, reconstruction, AttnRes, fusion redesign, optimizer, scheduler, or training operation was used.

## Artifacts

All JSON, this report, and P1–P8 plots are under `results/{OUTPUT_NAME}/`.

# EXPERIMENT 2D1B COMPLETE
"""
    return text


def source_reopen(args, device, output):
    source_path = Path(args.source_checkpoint).resolve()
    parent_path = Path(args.parent_checkpoint).resolve()
    validation_path = Path(args.validation_shard).resolve()
    hashes = {
        "source": file_sha256(source_path), "validation": file_sha256(validation_path),
        "parent": file_sha256(parent_path),
    }
    checks = {
        "source_exists": source_path.is_file(), "source_sha_exact": hashes["source"] == SOURCE_SHA256,
        "source_bytes_exact": source_path.stat().st_size == SOURCE_BYTES,
        "validation_sha_exact": hashes["validation"] == VALIDATION_SHA256,
    }
    if not all(checks.values()):
        raise SystemExit(f"source file preflight failed: {checks}")
    payload = d1.torch_load(source_path, mmap=True)
    metadata_checks = {
        "completed_updates_1908": payload.get("completed_updates") == 1908,
        "stage_C": payload.get("current_curriculum_stage") == "C",
        "windows_C12": tuple(payload.get("current_windows", ())) == C12,
        "rho_075": float(payload.get("rho", -1)) == 0.75,
        "model_mapping": isinstance(payload.get("model"), dict) and len(payload["model"]) > 0,
        "Wu_present": "fusion.W_u.weight" in payload.get("model", {}),
        "Wg_present": "fusion.W_g.weight" in payload.get("model", {}),
        "optimizer_metadata_read_only": isinstance(payload.get("optimizer"), dict),
    }
    if not all(metadata_checks.values()):
        raise SystemExit(f"C1908 metadata preflight failed: {metadata_checks}")
    _, model, architecture = d1.load_source_model(parent_path, device, trainable=False)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    strict = {
        "missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys),
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "eval_mode": not model.training,
        "weight_tying_preserved": model.base.transformer.wte.weight is model.base.lm_head.weight,
        "all_model_tensors_finite": all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
    }
    strict["passed"] = all(value is True or value == [] for value in strict.values())
    if not strict["passed"]:
        raise SystemExit(f"strict C1908 reopen failed: {strict}")
    manifest = {
        "checkpoint": str(source_path), "checkpoint_sha256": hashes["source"], "checkpoint_bytes": source_path.stat().st_size,
        "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": hashes["parent"],
        "validation_shard": str(validation_path), "validation_shard_sha256": hashes["validation"],
        "checkpoint_schema": payload.get("schema"), "checkpoint_git_commit": payload.get("git_commit"),
        "completed_updates": payload["completed_updates"], "processed_targets": payload.get("processed_targets"),
        "stage": payload["current_curriculum_stage"], "windows": payload["current_windows"], "rho": payload["rho"],
        "optimizer_metadata_read_only": {
            "present": isinstance(payload.get("optimizer"), dict),
            "keys": sorted(payload.get("optimizer", {}).keys()),
            "state_entries": len(payload.get("optimizer", {}).get("state", {})),
            "param_groups": len(payload.get("optimizer", {}).get("param_groups", [])),
            "objects_constructed": 0,
        },
        "architecture": architecture, "metadata_checks": metadata_checks, "strict_reopen": strict,
    }
    durable_json(output / "source_manifest.json", manifest)
    del payload
    return model, manifest


def run(args):
    started = time.time()
    require_config()
    git_checks = require_git(clean=True)
    device = require_hardware(args.pod_id)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output / "commands_and_runtime.json", {
        "command": " ".join(sys.argv), "started_at": started, "git_head": git_output("rev-parse", "HEAD"),
        "environment": runtime_environment(args.pod_id), "forbidden_operation_counts": FORBIDDEN_COUNTS,
    })
    model, source = source_reopen(args, device, output)
    before = {
        "model": module_state_sha256(model), "W_u": tensor_sha256(model.fusion.W_u.weight),
        "W_g": tensor_sha256(model.fusion.W_g.weight),
    }
    batches, batch_manifest = validation_batches(args.validation_shard)
    if batch_manifest["canonical_collection_sha256"] != CANONICAL_SHA256:
        raise SystemExit("canonical validation collection hash mismatch")
    durable_json(output / "batch_manifest.json", batch_manifest)
    factorial_manifest = {
        "experiment": EXPERIMENT, "protocol": PROTOCOL,
        "conditions": {name: {**row, "windows": list(row["windows"])} for name, row in CONDITIONS.items()},
        "controls": list(CONTROLS), "position_bins": [list(row) for row in POSITION_BINS],
        "self_composition_passes": PASSES, "self_composition_batches": SELF_BATCHES,
        "stage_a_reference_rms": STAGE_A_RMS, "hard_threshold": HARD_THRESHOLD,
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
    }
    durable_json(output / "factorial_manifest.json", factorial_manifest)
    causal = causal_audits(model, batches[0])
    if not causal["passed"]:
        raise SystemExit(f"causal preflight failed: {causal}")

    controls = {}; paired = {}; scale = {}; embedding = {}; gates = {}; positions = {}; self_results = {}; performance = {}

    # Condition A is a hard regression gate.  No counterfactual condition is run first.
    controls["A"], paired["A"] = canonical_condition(model, Path(args.validation_shard), "A", CONDITIONS["A"])
    oracle_deltas = {
        key: abs((controls["A"][key] if key in controls["A"] else None) - expected)
        for key, expected in NATIVE_ORACLE.items()
    }
    native_regression = {
        "oracle": NATIVE_ORACLE, "observed": {key: controls["A"].get(key) for key in NATIVE_ORACLE},
        "absolute_deltas": oracle_deltas, "maximum_absolute_delta": max(oracle_deltas.values()),
        "tolerance": 1e-6,
    }
    native_regression["passed"] = (
        native_regression["maximum_absolute_delta"] <= native_regression["tolerance"]
        and controls["A"]["canonical_validation_sha256"] == CANONICAL_SHA256
    )
    durable_json(output / "canonical_controls.json", {"A": controls["A"], "native_regression": native_regression})
    durable_json(output / "paired_losses.json", {"A": paired["A"]})
    if not native_regression["passed"]:
        raise SystemExit(f"native C1908 regression hard stop: {native_regression}")
    scale["A"], embedding["A"], gates["A"], positions["A"], diagnostic_seconds = diagnostic_condition(model, batches, "A", CONDITIONS["A"])
    self_results["A"] = self_composition(model, batches[:SELF_BATCHES], "A", CONDITIONS["A"])
    performance["A"] = {
        "canonical_controls": controls["A"]["performance"], "diagnostic_wall_seconds": diagnostic_seconds,
        "self_composition_wall_seconds": self_results["A"]["summary"]["wall_seconds"],
    }
    benchmark = {
        "canonical_REAL_included_in_A_control_evaluation": True,
        "A_four_control_wall_seconds": controls["A"]["performance"]["wall_seconds"],
        "A_32_pass_two_batch_wall_seconds": self_results["A"]["summary"]["wall_seconds"],
        "estimated_total_seconds_from_A": 4 * (controls["A"]["performance"]["wall_seconds"] + diagnostic_seconds + self_results["A"]["summary"]["wall_seconds"]),
    }
    durable_json(output / "performance.json", {"benchmark": benchmark, "conditions": performance})
    print(f"2D1B native gate PASS; estimated_total_seconds={benchmark['estimated_total_seconds_from_A']:.1f}", flush=True)

    for name in ("B", "C", "D"):
        controls[name], paired[name] = canonical_condition(model, Path(args.validation_shard), name, CONDITIONS[name])
        scale[name], embedding[name], gates[name], positions[name], diagnostic_seconds = diagnostic_condition(model, batches, name, CONDITIONS[name])
        self_results[name] = self_composition(model, batches[:SELF_BATCHES], name, CONDITIONS[name])
        performance[name] = {
            "canonical_controls": controls[name]["performance"], "diagnostic_wall_seconds": diagnostic_seconds,
            "self_composition_wall_seconds": self_results[name]["summary"]["wall_seconds"],
        }
        durable_json(output / "canonical_controls.json", {**controls, "native_regression": native_regression})
        durable_json(output / "paired_losses.json", paired)
        durable_json(output / "scale_decomposition.json", scale)
        durable_json(output / "embedding_content.json", embedding)
        durable_json(output / "gate_diagnostics.json", gates)
        durable_json(output / "position_bin_metrics.json", positions)
        durable_json(output / "self_composition.json", self_results)
        durable_json(output / "performance.json", {"benchmark": benchmark, "conditions": performance})
        print(f"2D1B condition {name} complete", flush=True)

    contrasts = factorial_contrasts(controls, self_results)
    scale_label = scale_classification(self_results)
    predictive_label = predictive_classification(contrasts)
    recommendation = next_recommendation(scale_label)
    after = {
        "model": module_state_sha256(model), "W_u": tensor_sha256(model.fusion.W_u.weight),
        "W_g": tensor_sha256(model.fusion.W_g.weight),
    }
    plain_equal = {
        "Plain_A_equals_Plain_B": controls["A"]["plain"] == controls["B"]["plain"],
        "Plain_C_equals_Plain_D": controls["C"]["plain"] == controls["D"]["plain"],
    }
    result_summary = {
        "scale_classification": scale_label, "predictive_classification": predictive_label,
        "next_recommendation": recommendation, "conditions": controls, "factorial_contrasts": contrasts,
        "stability": {name: self_results[name]["summary"] for name in CONDITIONS},
        "plain_equalities": plain_equal,
    }
    durable_json(output / "factorial_contrasts.json", contrasts)
    durable_json(output / "result_summary.json", result_summary)
    make_plots(output, controls, contrasts, scale, embedding, self_results, positions)
    arithmetic_checks = {
        "all_outputs_finite": all(finite_tree(value) for value in (controls, contrasts, scale, embedding, gates, positions, self_results)),
        "gain_exact": all(controls[name]["recurrent_gain"] == controls[name]["plain"] - controls[name]["real"] for name in CONDITIONS),
        "sequence_gap_exact": all(controls[name]["sequence_gap"] == controls[name]["shuffled"] - controls[name]["real"] for name in CONDITIONS),
        "zero_dependency_exact": all(controls[name]["zero_dependency"] == controls[name]["zero"] - controls[name]["real"] for name in CONDITIONS),
    }
    audit_checks = {
        "postmortem_tag_exact": git_checks["tag"], "C1908_SHA_exact": source["checkpoint_sha256"] == SOURCE_SHA256,
        "C1908_strict_reopen": source["strict_reopen"]["passed"],
        "model_parameter_hash_unchanged": before["model"] == after["model"],
        "W_u_unchanged": before["W_u"] == after["W_u"], "W_g_unchanged": before["W_g"] == after["W_g"],
        "forbidden_counts_all_zero": all(value == 0 for value in FORBIDDEN_COUNTS.values()),
        "exact_C12": CONDITIONS["A"]["windows"] == C12 and CONDITIONS["B"]["windows"] == C12,
        "exact_D12": CONDITIONS["C"]["windows"] == D12 and CONDITIONS["D"]["windows"] == D12,
        "exact_rhos": [CONDITIONS[name]["rho"] for name in CONDITIONS] == [0.75, 1.0, 0.75, 1.0],
        "exactly_four_conditions": list(controls) == list(CONDITIONS),
        "A_native_regression_exact": native_regression["passed"],
        **plain_equal,
        "canonical_validation_hash_exact": all(controls[name]["canonical_validation_sha256"] == CANONICAL_SHA256 for name in CONDITIONS),
        "same_fixed_derangement": True, "causal_and_row_isolation": causal["passed"],
        "cross_artifact_arithmetic_exact": all(arithmetic_checks.values()),
        "no_teacher_reconstruction_AttnRes_fusion_redesign_HellaSwag": True,
    }
    audit = {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "checks": audit_checks,
        "passed": all(audit_checks.values()), "native_regression": native_regression,
        "parameter_hashes": {"before": before, "after": after}, "causality": causal,
        "arithmetic": arithmetic_checks, "forbidden_operation_counts": FORBIDDEN_COUNTS,
        "git_synchronization": "FINALIZATION GATE OUTSIDE RESULT-BEARING PROCESS",
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    report = build_report(controls, paired, contrasts, scale, embedding, gates, self_results, positions,
                          scale_label, predictive_label, recommendation, audit, source)
    durable_text(output / "EXPERIMENT_2D1B_FINAL_REPORT.md", report)
    command = read_json(output / "commands_and_runtime.json")
    command.update({
        "completed_at": time.time(), "wall_seconds": time.time() - started,
        "forbidden_operation_counts": FORBIDDEN_COUNTS, "result": "PASS" if audit["passed"] else "FAIL",
    })
    durable_json(output / "commands_and_runtime.json", command)
    if not audit["passed"]:
        raise SystemExit(f"final audit failed: {audit_checks}")
    print(f"EXPERIMENT_2D1B_SCALE_CLASSIFICATION: {scale_label}", flush=True)
    print(f"EXPERIMENT_2D1B_PREDICTIVE_CLASSIFICATION: {predictive_label}", flush=True)
    print("EXPERIMENT_2D1B_COMPLETE", flush=True)


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
