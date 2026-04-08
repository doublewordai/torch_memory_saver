#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from torch_memory_saver import torch_memory_saver


def gpu_mem_used_bytes(device: torch.device) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hook-mode", choices=["torch"], default="torch")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--artifact-backend", choices=["ram", "disk"], default=None)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--preload-before-resume", action="store_true")
    parser.add_argument("--preload-wait-seconds", type=float, default=0.0)
    args = parser.parse_args()

    assert args.model_path.exists(), args.model_path
    if args.artifact_backend != "disk" and args.artifact_path is not None:
        parser.error("--artifact-path is only valid with --artifact-backend disk")
    if args.artifact_backend == "disk" and args.artifact_path is None:
        parser.error("--artifact-backend disk requires --artifact-path")
    if args.preload_before_resume and args.artifact_backend != "disk":
        parser.error("--preload-before-resume requires --artifact-backend disk")
    if args.preload_wait_seconds < 0:
        parser.error("--preload-wait-seconds must be non-negative")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    torch_memory_saver.hook_mode = args.hook_mode

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

    t0 = time.perf_counter()
    region_kwargs = {"tag": "weights"}
    if args.artifact_backend is None:
        region_kwargs["enable_cpu_backup"] = True
    else:
        region_kwargs["artifact_backend"] = args.artifact_backend
        if args.artifact_path is not None:
            region_kwargs["artifact_path"] = str(args.artifact_path)

    with torch_memory_saver.region(**region_kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - t0

    model.eval()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    used_after_load = gpu_mem_used_bytes(device)

    input_ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=device)
    with torch.inference_mode():
        logits_before = model(input_ids=input_ids).logits.float()
    checksum_before = logits_before.sum().item()

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    torch_memory_saver.pause("weights")
    torch.cuda.synchronize(device)
    pause_s = time.perf_counter() - t0
    used_after_pause = gpu_mem_used_bytes(device)

    preload_s = None
    if args.preload_before_resume:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch_memory_saver.preload("weights")
        preload_s = time.perf_counter() - t0
        if args.preload_wait_seconds > 0:
            time.sleep(args.preload_wait_seconds)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    torch_memory_saver.resume("weights")
    torch.cuda.synchronize(device)
    resume_s = time.perf_counter() - t0
    used_after_resume = gpu_mem_used_bytes(device)

    with torch.inference_mode():
        logits_after = model(input_ids=input_ids).logits.float()
    checksum_after = logits_after.sum().item()
    max_abs_diff = (logits_before - logits_after).abs().max().item()

    print(json.dumps(
        {
            "model_path": str(args.model_path),
            "hook_mode": args.hook_mode,
            "artifact_backend": args.artifact_backend or "legacy_cpu",
            "param_bytes": param_bytes,
            "param_gb": round(param_bytes / 1e9, 3),
            "gpu_mem_after_load_gb": round(used_after_load / 1e9, 3),
            "gpu_mem_after_pause_gb": round(used_after_pause / 1e9, 3),
            "gpu_mem_after_resume_gb": round(used_after_resume / 1e9, 3),
            "load_s": round(load_s, 3),
            "pause_s": round(pause_s, 3),
            "preload_s": None if preload_s is None else round(preload_s, 3),
            "preload_wait_s": round(args.preload_wait_seconds, 3),
            "resume_s": round(resume_s, 3),
            "checksum_before": checksum_before,
            "checksum_after": checksum_after,
            "max_abs_diff": max_abs_diff,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
