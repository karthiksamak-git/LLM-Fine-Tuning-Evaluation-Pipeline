"""Efficiency: trainable params, adapter size, inference latency, mixed-precision bench."""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from mlfte.config import Config
from mlfte.device import cuda_bf16_supported, resolve_device, select_precision
from mlfte.logging_utils import get_logger

log = get_logger(__name__)


def adapter_size_bytes(adapter_path: str | Path) -> int:
    path = Path(adapter_path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def parameter_counts(model) -> dict[str, Any]:
    """Total, grad-enabled and LoRA parameter counts.

    ``lora`` is counted regardless of ``requires_grad``: an adapter loaded for inference
    has frozen parameters but still contains exactly the weights that were trained, so it
    is the honest "trained parameters" number for the efficiency page.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
        "lora": lora,
        "lora_percent": 100.0 * lora / total if total else 0.0,
    }


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1)


def peak_memory_mb() -> dict[str, float | None]:
    cuda = None
    if torch.cuda.is_available():
        cuda = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return {"cpu_rss_mb": rss_mb(), "cuda_peak_allocated_mb": cuda}


def measure_latency(
    model,
    tokenizer,
    prompt: str,
    repeats: int,
    max_new_tokens: int = 32,
    warmup: int = 2,
) -> dict[str, Any]:
    """Greedy latency p50/p95 and tokens/s for a single fixed prompt."""
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    latencies: list[float] = []
    tokens = 0
    with torch.inference_mode():
        for i in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                latencies.append(elapsed)
                tokens += int(out.shape[1] - input_ids.shape[1])
    total_time = sum(latencies)
    return {
        "repeats": repeats,
        "warmup": warmup,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": int(input_ids.shape[1]),
        "p50_ms": round(statistics.median(latencies) * 1000, 2),
        "p95_ms": round(
            statistics.quantiles(latencies, n=20)[-1] * 1000 if len(latencies) > 2
            else max(latencies) * 1000,
            2,
        ),
        "mean_ms": round(statistics.mean(latencies) * 1000, 2),
        "tokens_per_second": round(tokens / total_time, 2) if total_time else None,
        "device": str(device),
    }


def mixed_precision_benchmark(cfg: Config, steps: int = 6) -> dict[str, Any]:
    """Short fixed-step fp32 vs bf16/fp16 training comparison (CUDA only)."""
    device = resolve_device()
    if device != "cuda":
        return {
            "skipped": True,
            "reason": (
                f"mixed precision: not measured (device={device}; no CUDA device available, "
                "so fp32 vs bf16/fp16 cannot be compared and is not simulated)."
            ),
        }
    from transformers import AutoModelForCausalLM


    precision = select_precision("cuda", cuda_bf16_supported())
    torch.manual_seed(cfg.run.seed)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model.to(device)
    ids = torch.randint(0, 1000, (2, 128), device=device)
    results: dict[str, Any] = {"skipped": False, "precision": precision.to_dict(), "steps": steps}
    for label, use_autocast in (("fp32", False), ("mixed", True)):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            with torch.autocast(
                device_type="cuda", dtype=precision.autocast_dtype, enabled=use_autocast
            ):
                out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results[label] = {
            "seconds": round(elapsed, 3),
            "steps_per_second": round(steps / elapsed, 3),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def efficiency_report(cfg: Config, adapter_path: str | Path) -> dict[str, Any]:
    """Full efficiency section: sizes, params, latency for base/adapter/merged."""
    from mlfte.data.prepare import read_split, split_paths
    from mlfte.data.tokenize import load_tokenizer, render_prompt
    from mlfte.evaluation.generate import load_model_for_inference
    from mlfte.tasks import get_task

    task = get_task()
    examples = read_split(split_paths(cfg)["test"])
    prompt = render_prompt(load_tokenizer(cfg.model.name), task, examples[0])
    sizes = {
        "adapter_bytes": adapter_size_bytes(adapter_path),
        "adapter_mb": round(adapter_size_bytes(adapter_path) / 1024**2, 3),
        "base_model": cfg.model.name,
    }

    device = resolve_device()
    precision = select_precision(device, cuda_bf16_supported())
    latency: dict[str, Any] = {}
    memory: dict[str, Any] = {}

    model, tokenizer = load_model_for_inference(cfg)
    memory["base"] = peak_memory_mb()
    latency["base"] = measure_latency(
        model, tokenizer, prompt, cfg.efficiency.latency_repeats
    )
    sizes.update({f"base_{k}": v for k, v in parameter_counts(model).items()})
    del model
    gc.collect()

    if Path(adapter_path).exists():
        model, tokenizer = load_model_for_inference(cfg, adapter_path=adapter_path)
        memory["adapter_unmerged"] = peak_memory_mb()
        latency["adapter_unmerged"] = measure_latency(
            model, tokenizer, prompt, cfg.efficiency.latency_repeats
        )
        sizes.update({f"adapter_{k}": v for k, v in parameter_counts(model).items()})
        merged = model.merge_and_unload()
        gc.collect()
        memory["merged"] = peak_memory_mb()
        latency["merged"] = measure_latency(
            merged, tokenizer, prompt, cfg.efficiency.latency_repeats
        )
        del merged, model
        gc.collect()

    training_metrics = {}
    train_metrics_path = cfg.results_dir / "training" / "train_metrics.json"
    if train_metrics_path.exists():
        training_metrics = json.loads(train_metrics_path.read_text())
        training_metrics = {
            k: training_metrics.get(k)
            for k in (
                "train_tokens_per_second",
                "peak_memory_mb",
                "wall_clock_s",
                "trainable_parameters",
                "total_parameters",
                "trainable_percent",
                "effective_batch_size",
            )
        }

    return {
        "sizes": sizes,
        "latency": latency,
        "memory": memory,
        "training": training_metrics,
        "mixed_precision_benchmark": mixed_precision_benchmark(cfg),
        "precision": precision.to_dict(),
        "settings": {
            "latency_repeats": cfg.efficiency.latency_repeats,
            "batch_size": cfg.efficiency.batch_size,
            "note": "Single fixed prompt, greedy decoding, warmup then repeats.",
        },
    }
