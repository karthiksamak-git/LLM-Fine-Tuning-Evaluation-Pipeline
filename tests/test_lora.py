"""LoRA wiring: target discovery, only-LoRA-trainable, fp32 cast."""

from __future__ import annotations

import pytest
import torch

from mlfte.config import load_config
from mlfte.training.lora import (
    apply_lora,
    assert_only_lora_trainable,
    cast_trainable_to_fp32,
    discover_target_modules,
    parameter_accounting,
)


def test_discovery_finds_llama_projections(tiny_model):
    modules = discover_target_modules(tiny_model, "auto")
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(modules)
    attention = discover_target_modules(tiny_model, "attention")
    assert set(attention) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_discovery_excludes_tied_output_head(tiny_model):
    modules = set(discover_target_modules(tiny_model, "auto"))
    assert "lm_head" not in modules
    assert "embed_tokens" not in modules
    assert "down_proj" in modules


def test_discovery_rejects_unknown_explicit_modules(tiny_model):
    with pytest.raises(ValueError):
        discover_target_modules(tiny_model, ["q_proj", "not_a_module"])


def test_apply_lora_makes_only_adapters_trainable(tiny_model):
    cfg = load_config(profile="smoke")
    model, info = apply_lora(tiny_model, cfg)
    assert_only_lora_trainable(model)
    assert info["lora_r"] == cfg.lora.r
    assert info["trainable_parameters"] > 0
    assert info["trainable_percent"] < 5.0
    assert info["lora_parameters"] == info["trainable_parameters"]


def test_parameter_accounting_totals(tiny_model):
    cfg = load_config(profile="smoke")
    model, _ = apply_lora(tiny_model, cfg)
    accounting = parameter_accounting(model)
    assert accounting["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert accounting["trainable_parameters"] <= accounting["total_parameters"]


def test_cast_trainable_to_fp32_under_fp16(tiny_model):
    cfg = load_config(profile="smoke")
    model, _ = apply_lora(tiny_model, cfg)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.half()
    cast = cast_trainable_to_fp32(model)
    assert cast > 0
    for name, param in model.named_parameters():
        if "lora_" in name and param.is_floating_point():
            assert param.dtype == torch.float32


def test_assert_only_lora_trainable_detects_violation(tiny_model):
    for param in tiny_model.parameters():
        param.requires_grad_(True)
    with pytest.raises(AssertionError):
        assert_only_lora_trainable(tiny_model)


def test_parameter_counts_counts_frozen_lora_adapter(tiny_model):
    """Regression: an adapter loaded for inference has frozen params but is still trained."""
    from mlfte.evaluation.efficiency import parameter_counts

    model, _ = apply_lora(tiny_model, load_config(profile="smoke"))
    counts = parameter_counts(model)
    assert counts["lora"] > 0
    assert counts["lora_percent"] < 5.0

    for param in model.parameters():
        param.requires_grad_(False)
    frozen = parameter_counts(model)
    assert frozen["trainable"] == 0
    assert frozen["lora"] == counts["lora"]
