"""PEFT LoRA configuration, target-module discovery and parameter accounting."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mlfte.config import Config

ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_proj")
# Architectures without the Llama-style names (kept for pluggability).
ATTENTION_FALLBACK = ("query", "key", "value", "dense", "c_attn", "out_proj")
# Output heads / embeddings are deliberately excluded: LoRA on a tied ``lm_head`` breaks
# weight tying on merge (PEFT warns about exactly this), and they are not attention/MLP.
EXCLUDED_MODULES = frozenset(
    {"lm_head", "embed_tokens", "wte", "output", "output_layer", "score", "classifier"}
)


def linear_module_names(model: nn.Module) -> list[str]:
    """Leaf names of every ``nn.Linear`` module, excluding heads and embeddings."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] not in EXCLUDED_MODULES:
            names.add(name.split(".")[-1])
    return sorted(names)


def discover_target_modules(model: nn.Module, variant: str | list[str] = "auto") -> list[str]:
    """Resolve the LoRA target modules against the real architecture.

    ``auto``/``all_linear`` -> every linear projection; ``attention`` -> attention
    projections only. Unknown explicit names raise instead of silently training nothing.
    """
    available = linear_module_names(model)
    if isinstance(variant, list):
        missing = [m for m in variant if m not in available]
        if missing:
            raise ValueError(f"target modules not found in model: {missing}; have {available}")
        return list(variant)

    if variant in ("auto", "all_linear"):
        wanted = set(available)
        return [m for m in available if m in wanted]
    if variant == "attention":
        chosen = [m for m in available if m in ATTENTION_SUFFIXES]
        if not chosen:
            chosen = [m for m in available if m in ATTENTION_FALLBACK]
        return chosen
    raise ValueError(f"unknown target_modules variant {variant!r}")


def build_lora_config(cfg: Config, model: nn.Module, r: int | None = None,
                      alpha: int | None = None, target_modules: Any | None = None):
    """Build a ``peft.LoraConfig`` with resolved target modules."""
    from peft import LoraConfig, TaskType

    variant = target_modules if target_modules is not None else cfg.lora.target_modules
    resolved = discover_target_modules(model, variant)
    return LoraConfig(
        r=r or cfg.lora.r,
        lora_alpha=alpha or (2 * (r or cfg.lora.r)),
        lora_dropout=cfg.lora.dropout,
        target_modules=resolved,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=False,
    )


def apply_lora(model: nn.Module, cfg: Config, **kwargs: Any):
    """Wrap the base model with LoRA adapters and return ``(model, info)``."""
    from peft import get_peft_model

    lora_config = build_lora_config(cfg, model, **kwargs)
    peft_model = get_peft_model(model, lora_config)
    info = {
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "task_type": str(lora_config.task_type),
        "target_modules": sorted(lora_config.target_modules),
        **parameter_accounting(peft_model),
    }
    return peft_model, info


def parameter_accounting(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = 0
    lora = 0
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)
            if "lora_" in name:
                lora += param.numel()
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
        "lora_parameters": lora,
        "trainable_parameter_names": trainable_names[:20],
        "n_trainable_tensors": len(trainable_names),
    }


def assert_only_lora_trainable(model: nn.Module) -> None:
    """Raise if any non-LoRA parameter requires gradients."""
    offenders = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]
    if offenders:
        raise AssertionError(f"non-LoRA parameters are trainable: {offenders[:5]}")


def cast_trainable_to_fp32(model: nn.Module) -> int:
    """Cast trainable params to fp32 (required under fp16 autocast + GradScaler).

    Returns the number of tensors cast. ChainLoRA/EmbeddingLoRA (non-float) params are
    skipped.
    """
    cast = 0
    for _name, param in model.named_parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            cast += 1
    return cast
