#!/usr/bin/env python3
"""Assert the faithful GPT-2 124M / FineWeb-Edu 10B training constants.

This script intentionally parses train_gpt2.py instead of importing it, because
importing train_gpt2.py launches training.
"""

import argparse
import ast
import math
from pathlib import Path


EXPECTED = {
    "total_batch_size": 524288,
    "B": 64,
    "T": 1024,
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 715,
    "max_steps": 19073,
    "use_compile": False,
}


def eval_expr(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_expr(node.operand, values)
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, values)
        right = eval_expr(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def collect_top_level_assignments(tree):
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in EXPECTED:
            continue
        try:
            values[name] = eval_expr(node.value, values)
        except Exception as exc:
            raise SystemExit(f"could not evaluate {name}: {exc}") from exc
    return values


def assert_close(name, actual, expected):
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
            raise SystemExit(f"{name} mismatch: expected {expected}, found {actual}")
    elif actual != expected:
        raise SystemExit(f"{name} mismatch: expected {expected}, found {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="train_gpt2.py")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--require-world-size-one", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    source = train_path.read_text()
    tree = ast.parse(source, filename=str(train_path))
    values = collect_top_level_assignments(tree)

    missing = sorted(set(EXPECTED) - set(values))
    if missing:
        raise SystemExit(f"missing expected top-level assignments: {missing}")

    for name, expected in EXPECTED.items():
        assert_close(name, values[name], expected)

    if args.require_world_size_one and args.world_size != 1:
        raise SystemExit(f"world_size mismatch: expected 1, found {args.world_size}")

    total_batch = values["total_batch_size"]
    B = values["B"]
    T = values["T"]
    denom = B * T * args.world_size
    if total_batch % denom != 0:
        raise SystemExit("total_batch_size is not divisible by B * T * world_size")
    grad_accum_steps = total_batch // denom
    if args.world_size == 1 and grad_accum_steps != 8:
        raise SystemExit(f"grad_accum_steps mismatch: expected 8, found {grad_accum_steps}")

    required_snippets = {
        "BF16 autocast": "torch.autocast(device_type=device_type, dtype=torch.bfloat16)",
        "matmul precision high": "torch.set_float32_matmul_precision('high')",
        "seed 1337": "torch.manual_seed(1337)",
        "CUDA seed 1337": "torch.cuda.manual_seed(1337)",
        "AdamW betas": "betas=(0.9, 0.95)",
        "AdamW eps": "eps=1e-8",
        "weight decay": "weight_decay=0.1",
        "gradient clipping": "clip_grad_norm_(model.parameters(), 1.0)",
        "FineWeb dir": 'data_root = "edu_fineweb10B"',
        "GPT-2 vocab padding": "GPTConfig(vocab_size=50304)",
    }
    for label, snippet in required_snippets.items():
        if snippet not in source:
            raise SystemExit(f"missing expected training snippet for {label}: {snippet}")

    print("training config assertions passed")
    print(f"B={B} T={T} total_batch_size={total_batch} world_size={args.world_size} grad_accum_steps={grad_accum_steps}")
    print(f"max_lr={values['max_lr']} min_lr={values['min_lr']} warmup_steps={values['warmup_steps']} max_steps={values['max_steps']}")
    print("use_compile=False, BF16 autocast, AdamW, gradient clipping, and seed checks passed")


if __name__ == "__main__":
    main()

