"""Merge a LoRA adapter into the base model and verify output parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mlfte.config import Config
from mlfte.device import resolve_device
from mlfte.logging_utils import get_logger

log = get_logger(__name__)

PARITY_TOLERANCE_MAX_ABS = 5e-3


def merge_adapter(cfg: Config, adapter_path: str | Path, merged_dir: str | Path) -> Path:
    """Merge adapter weights into the base model and save the merged model fp32."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from mlfte.data.tokenize import load_tokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    load_tokenizer(cfg.model.name).save_pretrained(str(merged_dir))
    log.info("merged adapter -> %s", merged_dir)
    return merged_dir


def verify_parity(
    cfg: Config,
    adapter_path: str | Path,
    merged_dir: str | Path,
    batch_size: int = 2,
    seq_len: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare adapter-vs-merged logits on a fixed batch; returns the documented tolerance."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM


    device = resolve_device()
    torch.manual_seed(seed)
    input_ids = torch.randint(100, 4096, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32).to(device)
    adapter_model = PeftModel.from_pretrained(base, str(adapter_path)).to(device).eval()
    with torch.inference_mode():
        adapter_logits = adapter_model(input_ids=input_ids, attention_mask=attention_mask).logits
    del adapter_model, base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    merged_model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), dtype=torch.float32
    ).to(device).eval()
    with torch.inference_mode():
        merged_logits = merged_model(input_ids=input_ids, attention_mask=attention_mask).logits
    del merged_model

    diff = (adapter_logits - merged_logits).abs()
    max_abs = float(diff.max())
    result = {
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": float(diff.mean()),
        "tolerance_max_abs": PARITY_TOLERANCE_MAX_ABS,
        "passed": bool(max_abs <= PARITY_TOLERANCE_MAX_ABS),
        "batch_shape": [batch_size, seq_len],
        "seed": seed,
        "device": device,
    }
    log.info("merge parity: max diff %.3e (tolerance %.1e)", max_abs, PARITY_TOLERANCE_MAX_ABS)
    if not result["passed"]:
        raise AssertionError(
            f"merged model diverges from adapter: max abs diff {max_abs:.3e} > "
            f"{PARITY_TOLERANCE_MAX_ABS}"
        )
    return result
