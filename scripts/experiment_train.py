#!/usr/bin/env python3
"""Matched Experiment 1A/1B training harness."""

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import torch

import smoke_test as support


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb"
EXPECTED_INIT_SHA256 = "39de351efe080de4e2409355c572095f17dcbaea76154a2f55e375acfdafc3b6"


@torch.no_grad()
def evaluate_hellaswag(model, symbols, device, limit=None):
    model.eval()
    total = 0
    correct = 0
    for example in symbols["iterate_examples"]("val"):
        _, tokens, mask, label = symbols["render_example"](example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)
        prediction = symbols["get_most_likely_row"](tokens, mask, logits)
        total += 1
        correct += int(prediction == label)
        if total % 500 == 0:
            print(f"HellaSwag progress: {total} examples", flush=True)
        if limit is not None and total >= limit:
            break
    return {"examples": total, "correct_norm": correct, "accuracy_norm": correct / total}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def verify_shared_initialization(model, baseline_state):
    current = model.state_dict()
    missing = sorted(set(baseline_state) - set(current))
    shape_mismatches = sorted(
        key for key, value in baseline_state.items()
        if key in current and current[key].shape != value.shape
    )
    exact = 0
    mismatch_keys = []
    maximum_difference = 0.0
    if not missing and not shape_mismatches:
        for key, expected in baseline_state.items():
            actual = current[key]
            if torch.equal(actual, expected):
                exact += 1
            else:
                mismatch_keys.append(key)
                maximum_difference = max(
                    maximum_difference,
                    (actual.float() - expected.float()).abs().max().item(),
                )

    query_tensors = [
        value for key, value in current.items()
        if ".attnres." in key and key.endswith(".query")
    ]
    norm_tensors = [
        value for key, value in current.items()
        if ".attnres." in key and key.endswith(".norm.weight")
    ]
    report = {
        "shared_tensors": len(baseline_state),
        "exact_matches": exact,
        "mismatches": len(mismatch_keys) + len(missing) + len(shape_mismatches),
        "mismatch_keys": mismatch_keys,
        "missing_keys": missing,
        "shape_mismatches": shape_mismatches,
        "maximum_absolute_difference": maximum_difference,
        "attnres_query_tensors": len(query_tensors),
        "attnres_queries_exactly_zero": all(torch.count_nonzero(value).item() == 0 for value in query_tensors),
        "attnres_rmsnorm_tensors": len(norm_tensors),
        "attnres_rmsnorms_exactly_one": all(torch.equal(value, torch.ones_like(value)) for value in norm_tensors),
    }
    if report["mismatches"]:
        raise SystemExit(f"shared initialization mismatch: {report}")
    if model.config.residual_mode == "full_attnres" and (
        len(query_tensors) != 25
        or not report["attnres_queries_exactly_zero"]
        or len(norm_tensors) != 25
        or not report["attnres_rmsnorms_exactly_one"]
    ):
        raise SystemExit(f"Full AttnRes initialization mismatch: {report}")
    if model.config.residual_mode == "standard" and (query_tensors or norm_tensors):
        raise SystemExit(f"standard model unexpectedly contains AttnRes tensors: {report}")
    return report


def data_order_probe(loader, microbatches=8):
    start = {
        "shard_index": loader.current_shard,
        "shard_path": loader.shards[loader.current_shard],
        "position": loader.current_position,
    }
    combined = hashlib.sha256()
    batch_hashes = []
    for index in range(microbatches):
        x, y = loader.next_batch()
        payload = x.contiguous().numpy().tobytes() + y.contiguous().numpy().tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        combined.update(payload)
        batch_hashes.append({
            "microbatch": index,
            "sha256": digest,
            "first_input_token": int(x.view(-1)[0]),
            "last_target_token": int(y.view(-1)[-1]),
        })
    loader.reset()
    reset = {
        "shard_index": loader.current_shard,
        "shard_path": loader.shards[loader.current_shard],
        "position": loader.current_position,
    }
    if reset != start:
        raise SystemExit(f"data loader did not reset exactly: start={start}, reset={reset}")
    return {
        "microbatches_hashed": microbatches,
        "combined_sha256": combined.hexdigest(),
        "start": start,
        "reset": reset,
        "microbatch_hashes": batch_hashes,
    }


def routing_parameter_stats(model):
    if model.config.residual_mode != "full_attnres":
        return None
    query_norms = [router.query.detach().float().norm().item() for router in model.transformer.attnres]
    norm_l2_displacements = [
        (router.norm.weight.detach().float() - 1).norm().item()
        for router in model.transformer.attnres
    ]
    norm_max_displacements = [
        (router.norm.weight.detach().float() - 1).abs().max().item()
        for router in model.transformer.attnres
    ]
    return {
        "query_norms": query_norms,
        "maximum_query_norm": max(query_norms),
        "median_query_norm": statistics.median(query_norms),
        "rmsnorm_l2_displacements": norm_l2_displacements,
        "maximum_rmsnorm_l2_displacement": max(norm_l2_displacements),
        "median_rmsnorm_l2_displacement": statistics.median(norm_l2_displacements),
        "rmsnorm_max_abs_displacements": norm_max_displacements,
        "maximum_rmsnorm_abs_displacement": max(norm_max_displacements),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--residual-mode", choices=("standard", "full_attnres"), required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--expected-data-sha256")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if config["max_updates"] not in {191, 477}:
        raise SystemExit("only the reviewed 100M and 250M update budgets are accepted")
    if config["lr_schedule"] != "preserve_original_10b_recipe":
        raise SystemExit("unreviewed learning-rate protocol")
    if config["max_updates"] * support.TOTAL_BATCH_SIZE != config["actual_tokens"]:
        raise SystemExit("actual_tokens does not match max_updates * global batch")
    if config["max_updates"] == 191:
        required_targets = [524288, 10485760, 25690112, 50331648, 75497472, 100139008]
        if config["eval_token_targets"] != required_targets:
            raise SystemExit(f"100M evaluation targets must be {required_targets}")
        if config["hellaswag_examples"] is not None:
            raise SystemExit("Experiment 1A requires complete HellaSwag evaluation")
    if git_output("rev-parse", "baseline-gpt2-124m-10b^{commit}") != BASELINE_COMMIT:
        raise SystemExit("frozen baseline tag does not resolve to the expected commit")

    init_sha256 = support.file_sha256(args.init_checkpoint)
    if init_sha256 != EXPECTED_INIT_SHA256:
        raise SystemExit(
            f"initialization checkpoint SHA256 mismatch: expected {EXPECTED_INIT_SHA256}, got {init_sha256}"
        )
    dataset_report_path = Path(args.dataset_report)
    environment_report_path = Path(args.environment_report)
    dataset_report = json.loads(dataset_report_path.read_text())
    environment_report = json.loads(environment_report_path.read_text())
    if dataset_report.get("failures"):
        raise SystemExit(f"dataset verification has failures: {dataset_report['failures']}")
    if environment_report.get("failures"):
        raise SystemExit(f"environment verification has failures: {environment_report['failures']}")

    support.assert_cuda_environment(require_a100_80gb=True)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    symbols = support.load_training_symbols()

    model_config = symbols["GPTConfig"](vocab_size=50304, residual_mode=args.residual_mode)
    model = symbols["GPT"](model_config)
    init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if args.residual_mode == "standard":
        model.load_state_dict(init_checkpoint["model"], strict=True)
    else:
        model.load_shared_baseline_state(init_checkpoint["model"])
    initialization_verification = verify_shared_initialization(model, init_checkpoint["model"])
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=support.MAX_LR, device_type="cuda")

    DataLoaderLite = symbols["DataLoaderLite"]
    train_loader = DataLoaderLite(B=support.B, T=support.T, process_rank=0, num_processes=1, split="train")
    val_loader = DataLoaderLite(B=support.B, T=support.T, process_rank=0, num_processes=1, split="val")
    data_order = data_order_probe(train_loader)
    if args.expected_data_sha256 and data_order["combined_sha256"] != args.expected_data_sha256:
        raise SystemExit(
            f"training data order mismatch: expected {args.expected_data_sha256}, "
            f"got {data_order['combined_sha256']}"
        )

    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()
    metrics_file = run_dir / "metrics.jsonl"
    attnres_file = run_dir / "attnres_stats.jsonl"
    shutil.copy2(dataset_report_path, run_dir / "dataset.json")
    shutil.copy2(environment_report_path, run_dir / "environment.json")
    metrics_file.write_text("")
    if args.residual_mode == "full_attnres":
        attnres_file.write_text("")
    (run_dir / "initialization_verification.json").write_text(
        json.dumps(initialization_verification, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "data_order.json").write_text(json.dumps(data_order, indent=2, sort_keys=True) + "\n")

    metadata = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short", "--branch"),
        "baseline_parent_commit": BASELINE_COMMIT,
        "baseline_tag": "baseline-gpt2-124m-10b",
        "model_config": vars(model_config),
        "attnres_config": {
            "mode": args.residual_mode,
            "query_initialization": "zeros",
            "key_normalization": "RMSNorm",
            "rmsnorm_epsilon": model_config.attnres_rms_eps,
            "softmax_dimension": "residual_depth",
        },
        "seed": 1337,
        "dataset_verification": "dataset.json",
        "dataset_verification_sha256": support.file_sha256(dataset_report_path),
        "dataset_summary": {
            "number_of_shard_files": dataset_report["number_of_shard_files"],
            "total_token_count": dataset_report["total_token_count"],
            "dtype": dataset_report["dtype"],
            "validation_shards": dataset_report["validation_shards_upstream"],
            "training_shard_count": len(dataset_report["training_shards_upstream"]),
        },
        "environment_verification": "environment.json",
        "environment_verification_sha256": support.file_sha256(environment_report_path),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "B": support.B,
        "T": support.T,
        "gradient_accumulation": support.GRAD_ACCUM_STEPS,
        "global_batch_tokens": support.TOTAL_BATCH_SIZE,
        "precision": "BF16 autocast, FP32 parameters/residual accumulator",
        "optimizer": "fused AdamW, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1",
        "lr_schedule": "original 10B: 715-step warmup then cosine through 19073 updates",
        "training_token_budget": config["target_tokens"],
        "planned_updates": config["max_updates"],
        "baseline_init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "baseline_init_sha256": init_sha256,
        "initialization_verification": "initialization_verification.json",
        "data_order_verification": "data_order.json",
        "training_data_sha256": data_order["combined_sha256"],
        "experiment_config": config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    eval_targets = list(config["eval_token_targets"])
    eval_index = 0
    train_rows = []
    evaluation_rows = []
    run_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(config["max_updates"]):
        row = support.timed_optimizer_update(
            model, optimizer, train_loader, device, support.get_lr(step), metrics_file, step,
            wall_clock_start=run_start,
        )
        train_rows.append(row)
        print(
            f"step {step:4d} | tokens {row['tokens']:,} | loss {row['train_loss']:.6f} | "
            f"{row['tokens_per_second']:.1f} tok/s",
            flush=True,
        )

        while eval_index < len(eval_targets) and row["tokens"] >= eval_targets[eval_index]:
            target = eval_targets[eval_index]
            if args.residual_mode == "full_attnres":
                model.set_attnres_instrumentation(True)
            val_loss = support.validation_loss(model, val_loader, device, config["validation_steps"])
            stats = model.get_attnres_stats() if args.residual_mode == "full_attnres" else []
            if args.residual_mode == "full_attnres":
                model.set_attnres_instrumentation(False)
                routing_row = {
                    "step": step,
                    "tokens": row["tokens"],
                    "target_tokens": target,
                    "destinations": stats,
                    "parameters": routing_parameter_stats(model),
                }
                support.write_jsonl(attnres_file, routing_row)
            hella = evaluate_hellaswag(model, symbols, device, config["hellaswag_examples"])
            expected_hellaswag = config.get("expected_hellaswag_examples")
            if expected_hellaswag is not None and hella["examples"] != expected_hellaswag:
                raise SystemExit(
                    f"incomplete HellaSwag evaluation: expected {expected_hellaswag}, got {hella['examples']}"
                )
            elapsed = time.perf_counter() - run_start
            memory = support.gpu_memory_mb()
            common = {
                "step": step,
                "tokens": row["tokens"],
                "target_tokens": target,
                "wall_clock_seconds": elapsed,
                "gpu_allocated_mb": memory["allocated_mb"],
                "gpu_reserved_mb": memory["reserved_mb"],
                "gpu_peak_mb": memory["peak_allocated_mb"],
                "gpu_peak_reserved_mb": memory["peak_reserved_mb"],
            }
            support.write_jsonl(metrics_file, {"kind": "val", "val_loss": val_loss, **common})
            support.write_jsonl(metrics_file, {
                "kind": "hellaswag",
                "hellaswag_accuracy": hella["accuracy_norm"],
                "hellaswag_examples": hella["examples"],
                **common,
            })
            evaluation_rows.append({
                "step": step,
                "tokens": row["tokens"],
                "target_tokens": target,
                "validation_loss": val_loss,
                "hellaswag": hella,
                "wall_clock_seconds": elapsed,
                "routing_parameters": routing_parameter_stats(model),
            })
            print(
                f"evaluation step {step} | tokens {row['tokens']:,} | val {val_loss:.6f} | "
                f"HellaSwag {hella['accuracy_norm']:.6f} ({hella['correct_norm']}/{hella['examples']})",
                flush=True,
            )
            checkpoint_path = checkpoint_dir / f"model_tokens_{row['tokens']:012d}.pt"
            torch.save({
                "model": model.state_dict(),
                "config": vars(model.config),
                "step": step,
                "tokens": row["tokens"],
                "val_loss": val_loss,
                "baseline_init_sha256": metadata["baseline_init_sha256"],
            }, checkpoint_path)
            eval_index += 1

    if eval_index != len(eval_targets):
        raise SystemExit(f"not all evaluation targets were reached: {eval_index}/{len(eval_targets)}")

    memory = support.gpu_memory_mb()
    summary = {
        "residual_mode": args.residual_mode,
        "final_step": train_rows[-1]["step"],
        "processed_tokens": train_rows[-1]["tokens"],
        "final_train_loss": train_rows[-1]["train_loss"],
        "final_gradient_norm": train_rows[-1]["grad_norm"],
        "final_validation_loss": evaluation_rows[-1]["validation_loss"],
        "final_hellaswag": evaluation_rows[-1]["hellaswag"],
        "total_wall_clock_seconds": time.perf_counter() - run_start,
        "training_update_seconds": sum(row["step_time_ms"] for row in train_rows) / 1000,
        "mean_seconds_per_update": sum(row["step_time_ms"] for row in train_rows) / len(train_rows) / 1000,
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in train_rows) / len(train_rows),
        "peak_allocated_mb": memory["peak_allocated_mb"],
        "peak_reserved_mb": memory["peak_reserved_mb"],
        "parameter_count": metadata["parameter_count"],
        "initialization_verification": initialization_verification,
        "training_data_sha256": data_order["combined_sha256"],
        "evaluations": evaluation_rows,
        "final_routing_parameters": routing_parameter_stats(model),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
