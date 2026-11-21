"""Batched greedy generation with left padding; saves raw generations as JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mlfte.config import Config
from mlfte.data.tokenize import load_tokenizer, render_prompt
from mlfte.device import resolve_device, select_precision
from mlfte.logging_utils import JsonlWriter, Timer, get_logger, read_jsonl
from mlfte.tasks.base import Example, Task

log = get_logger(__name__)


def load_model_for_inference(
    cfg: Config,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
    merged: bool = False,
):
    """Load base (optionally + PEFT adapter) model and tokenizer for generation."""
    from transformers import AutoModelForCausalLM

    name = base_model or cfg.model.name
    device = resolve_device()
    precision = select_precision(device, _bf16_supported())
    tokenizer = load_tokenizer(name)
    tokenizer.padding_side = "left"
    if merged:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    elif adapter_path:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    log.info("loaded %s%s on %s (%s)", name, " +adapter" if adapter_path else "",
             device, precision.label)
    return model, tokenizer


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover
        return False


def _render_batch(
    tokenizer, task: Task, batch: list[Example], few_shot: list[Example] | None
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    texts = [render_prompt(tokenizer, task, ex, few_shot) for ex in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length if tokenizer.model_max_length < 10**6 else 2048,
    )
    return texts, encoded["input_ids"], encoded["attention_mask"]


def generate_predictions(
    model,
    tokenizer,
    task: Task,
    examples: list[Example],
    cfg: Config,
    out_path: str | Path,
    few_shot: list[Example] | None = None,
    batch_size: int | None = None,
    max_new_tokens: int | None = None,
    label: str = "model",
) -> dict[str, Any]:
    """Greedy-decode an answer for every example; append results to ``out_path``."""
    device = next(model.parameters()).device
    batch_size = batch_size or cfg.eval.batch_size
    max_new_tokens = max_new_tokens or cfg.eval.max_new_tokens
    tokenizer.padding_side = "left"
    writer = JsonlWriter(out_path)
    writer.reset()

    generated_tokens = 0
    total_seconds = 0.0
    with Timer() as timer:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                prompts, input_ids, attention_mask = _render_batch(
                    tokenizer, task, batch, few_shot
                )
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                t_batch = Timer()
                with t_batch:
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                # Greedy + left padding: only take the newly generated tokens.
                generated = output[:, input_ids.shape[1] :]
                generated_tokens += int((generated != tokenizer.pad_token_id).sum())
                total_seconds += t_batch.seconds
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                records = []
                for example, prompt, text in zip(batch, prompts, decoded, strict=True):
                    records.append(
                        {
                            "id": example.id,
                            "prompt": prompt,
                            "gold": example.answer,
                            "generation": text.strip(),
                            "parsed_sql": task.parse_output(text),
                            "latency_ms": round(t_batch.seconds * 1000 / len(batch), 3),
                        }
                    )
                writer.write(records)
                log.info("%s: %d/%d examples", label, min(start + batch_size, len(examples)),
                         len(examples))

    metadata = {
        "label": label,
        "output": str(out_path),
        "n": len(examples),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy (do_sample=False, num_beams=1)",
        "padding_side": "left",
        "few_shot_k": len(few_shot or []),
        "wall_clock_s": round(timer.seconds, 2),
        "generation_seconds": round(total_seconds, 2),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / total_seconds, 2) if total_seconds else None,
        "device": str(device),
    }
    meta_path = Path(out_path).with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("wrote %s (%s)", out_path, metadata["wall_clock_s"])
    return metadata


def load_generations(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
