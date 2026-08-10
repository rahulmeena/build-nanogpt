#!/usr/bin/env python3
"""Report and optionally enforce the environment needed for the reproduction."""

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd):
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"command": cmd, "returncode": 127, "stdout": "", "stderr": str(exc)}


def package_status(name):
    if importlib.util.find_spec(name) is None:
        return {"installed": False, "version": None}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"installed": True, "version": version}


def git_info():
    return {
        "commit_sha": run_command(["git", "rev-parse", "HEAD"])["stdout"],
        "branch": run_command(["git", "branch", "--show-current"])["stdout"],
        "status": run_command(["git", "status", "--short", "--branch"])["stdout"],
    }


def nvidia_smi_info():
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "full_output": "nvidia-smi not found",
            "query_output": "",
        }
    return {
        "available": True,
        "full_output": run_command(["nvidia-smi"])["stdout"],
        "query_output": run_command([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])["stdout"],
    }


def torch_info():
    info = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpus": [],
        "bf16_supported": False,
        "sdpa_available": False,
        "fused_adamw_available": False,
        "matmul_precision": None,
        "tf32_matmul_allowed": None,
    }
    try:
        import torch
    except Exception as exc:
        info["import_error"] = repr(exc)
        return info

    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version()
    info["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    info["fused_adamw_available"] = "fused" in inspect.signature(torch.optim.AdamW).parameters
    info["matmul_precision"] = torch.get_float32_matmul_precision()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        info["tf32_matmul_allowed"] = torch.backends.cuda.matmul.allow_tf32
    if info["cuda_available"]:
        info["gpu_count"] = torch.cuda.device_count()
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gib": props.total_memory / 1024**3,
                "major": props.major,
                "minor": props.minor,
            })
    return info


def verify_hellaswag_render():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from hellaswag import render_example
        data, tokens, mask, label = render_example({
            "ctx": "A person opens a notebook.",
            "label": 0,
            "endings": [
                "They write a short note.",
                "The ocean turns purple.",
                "A spaceship lands inside.",
                "The sentence disappears.",
            ],
        })
        return {
            "ok": True,
            "ctx_token_count": len(data["ctx_tokens"]),
            "tokens_shape": list(tokens.shape),
            "mask_shape": list(mask.shape),
            "label": label,
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def write_report(path, text):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write the report to this file")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-a100-80gb", action="store_true")
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--require-sdpa", action="store_true")
    parser.add_argument("--require-fused-adamw", action="store_true")
    args = parser.parse_args()

    deps = {
        name: package_status(name)
        for name in ["torch", "tiktoken", "datasets", "tqdm", "requests", "numpy", "matplotlib", "transformers"]
    }
    report = {
        "os": platform.platform(),
        "system": platform.system(),
        "linux_kernel": platform.release() if platform.system() == "Linux" else None,
        "python_version": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "git": git_info(),
        "dependencies": deps,
        "torch": torch_info(),
        "nvidia_smi": nvidia_smi_info(),
        "hellaswag_render": verify_hellaswag_render(),
    }

    failures = []
    torch_report = report["torch"]
    if args.require_cuda and not torch_report["cuda_available"]:
        failures.append("CUDA is required but not available")
    if args.require_bf16 and not torch_report["bf16_supported"]:
        failures.append("BF16 support is required but not available")
    if args.require_sdpa and not torch_report["sdpa_available"]:
        failures.append("scaled_dot_product_attention is required but not available")
    if args.require_fused_adamw and not torch_report["fused_adamw_available"]:
        failures.append("fused AdamW support is required but not available")
    if args.require_a100_80gb:
        matching = [
            gpu for gpu in torch_report["gpus"]
            if "A100" in gpu["name"] and gpu["total_memory_gib"] >= 79.0
        ]
        if not matching:
            failures.append("A100 with at least 79 GiB visible VRAM is required but not detected")
    missing_deps = [name for name, status in deps.items() if not status["installed"]]
    if missing_deps:
        failures.append(f"missing Python dependencies: {', '.join(missing_deps)}")
    if not report["hellaswag_render"]["ok"]:
        failures.append(f"hellaswag render check failed: {report['hellaswag_render'].get('error')}")

    report["failures"] = failures
    text = json.dumps(report, indent=2, sort_keys=True)
    write_report(args.out, text + "\n")
    print(text if args.json else text)
    if failures:
        raise SystemExit("environment verification failed")


if __name__ == "__main__":
    main()

