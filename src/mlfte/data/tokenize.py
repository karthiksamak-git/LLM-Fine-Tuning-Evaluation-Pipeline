"""Tokenization with completion-only label masking.

* Prompt tokens get label ``-100`` (ignored by the cross-entropy loss).
* Answer tokens (plus EOS) keep their ids as labels.
* Over-length examples have the *prompt* truncated from the left so the question at the
  end of the prompt and the whole answer survive. The truncation fraction is reported
  and must stay below 1%; otherwise the gate raises and the caller must raise
  ``model.max_seq_len``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from mlfte.config import Config
from mlfte.logging_utils import get_logger
from mlfte.tasks.base import Example, Task

log = get_logger(__name__)

MIN_PROMPT_TOKENS = 8
MAX_TRUNCATION_FRACTION = 0.01
IGNORE_INDEX = -100


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def render_prompt(
    tokenizer, task: Task, example: Example, few_shot: list[Example] | None = None
) -> str:
    messages = [m.to_dict() for m in task.messages(example, few_shot)]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_token_ids(
    tokenizer, task: Task, example: Example, few_shot: list[Example] | None = None
) -> list[int]:
    text = render_prompt(tokenizer, task, example, few_shot)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def answer_token_ids(tokenizer, task: Task, example: Example, add_eos: bool = True) -> list[int]:
    ids = tokenizer(task.target(example), add_special_tokens=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return ids


def encode_example(
    tokenizer,
    task: Task,
    example: Example,
    max_seq_len: int,
    loss_masking: str = "completion",
) -> tuple[dict[str, Any] | None, str | None]:
    """Encode one example. Returns ``(record, drop_reason)``; exactly one is non-None."""
    prompt = prompt_token_ids(tokenizer, task, example)
    answer = answer_token_ids(tokenizer, task, example)

    if len(answer) >= max_seq_len:
        return None, "answer_exceeds_max_seq_len"
    truncated = False
    if len(prompt) + len(answer) > max_seq_len:
        budget = max_seq_len - len(answer)
        if budget < MIN_PROMPT_TOKENS:
            return None, "prompt_budget_too_small"
        prompt = prompt[-budget:]
        truncated = True

    input_ids = prompt + answer
    if loss_masking == "completion":
        labels = [IGNORE_INDEX] * len(prompt) + answer
    elif loss_masking == "full":
        labels = list(input_ids)
    else:
        raise ValueError(f"unknown loss_masking {loss_masking!r}")

    record = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
        "prompt_len": len(prompt),
        "answer_len": len(answer),
    }
    return record, None


def encode_split(
    tokenizer,
    task: Task,
    examples: list[Example],
    max_seq_len: int,
    loss_masking: str = "completion",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    drops: dict[str, int] = {}
    truncated = 0
    for ex in tqdm(examples, desc="tokenize", leave=False):
        rec, reason = encode_example(tokenizer, task, ex, max_seq_len, loss_masking)
        if rec is None:
            drops[reason or "unknown"] = drops.get(reason or "unknown", 0) + 1
            continue
        truncated += int(rec["truncated"])
        records.append(rec)
    stats = {
        "examples": len(examples),
        "kept": len(records),
        "dropped": drops,
        "truncated": truncated,
        "truncation_fraction": round(truncated / max(1, len(examples)), 6),
    }
    return records, stats


def _length_percentiles(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [r[key] for r in records]
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99, 100)}
    out["mean"] = round(float(arr.mean()), 2)
    return out


def histogram(records: list[dict[str, Any]], bins: int = 12) -> dict[str, Any]:
    total = [r["prompt_len"] + r["answer_len"] for r in records]
    if not total:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(np.asarray(total), bins=bins)
    return {"edges": [int(e) for e in edges], "counts": [int(c) for c in counts]}


def token_stats_for_splits(
    cfg: Config, task: Task, splits: dict[str, list[Example]], tokenizer=None
) -> dict[str, Any]:
    """Tokenize every split once and produce stats; raises if truncation is too high."""
    tok = tokenizer or load_tokenizer(cfg.model.name)
    report: dict[str, Any] = {}
    for name, examples in splits.items():
        records, stats = encode_split(
            tok, task, examples, cfg.model.max_seq_len, cfg.training.loss_masking
        )
        stats["prompt_tokens"] = _length_percentiles(records, "prompt_len")
        stats["answer_tokens"] = _length_percentiles(records, "answer_len")
        stats["total_tokens_histogram"] = histogram(records)
        report[name] = stats
        if stats["truncation_fraction"] >= MAX_TRUNCATION_FRACTION:
            raise RuntimeError(
                f"truncation fraction {stats['truncation_fraction']:.4f} for {name} split is "
                f">= {MAX_TRUNCATION_FRACTION}; raise model.max_seq_len"
            )
    report["policy"] = {
        "loss_masking": cfg.training.loss_masking,
        "prompt_truncation": "left (keeps the end of the prompt and the full answer)",
        "max_seq_len": cfg.model.max_seq_len,
        "ignore_index": IGNORE_INDEX,
    }
    return report


def build_dataset(cfg: Config, task: Task, examples: list[Example], tokenizer, split_name: str):
    """Return an in-memory HF Dataset with input_ids/labels/attention_mask."""
    from datasets import Dataset

    records, stats = encode_split(
        tokenizer, task, examples, cfg.model.max_seq_len, cfg.training.loss_masking
    )
    if not records:
        raise RuntimeError(f"no usable examples in the {split_name} split")
    log.info(
        "%s split: kept %d/%d, truncated %d (%s)",
        split_name,
        stats["kept"],
        stats["examples"],
        stats["truncated"],
        stats["dropped"],
    )
    ds = Dataset.from_list(records)
    ds = ds.select_columns(["input_ids", "labels", "attention_mask"])
    return ds, stats
