"""Forgetting checks: WikiText-2 perplexity and ARC-Easy accuracy via log-likelihood.

Both use a *fixed* slice (seeded once), so the before/after numbers are comparable.
Scoring is hand-written: no ``evaluate`` dependency.
"""

from __future__ import annotations

from typing import Any

import torch

from mlfte.config import Config
from mlfte.logging_utils import get_logger

log = get_logger(__name__)

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
ARC_DATASET = "allenai/ai2_arc"
ARC_CONFIG = "ARC-Easy"
ARC_PROMPT = "Question: {question}\nAnswer:"


def _blocks(tokenizer, text: str, block_size: int) -> list[list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
    return [b for b in blocks if len(b) == block_size]


def wikitext_perplexity(
    model, tokenizer, cfg: Config, block_size: int = 256, max_blocks: int | None = None
) -> dict[str, Any]:
    """Token-level perplexity on a fixed WikiText-2 test slice."""
    from datasets import load_dataset

    n_lines = cfg.eval.wikitext_slice
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="test")
    lines = [row["text"] for row in ds if row["text"].strip()]
    text = "\n\n".join(lines[:n_lines])
    blocks = _blocks(tokenizer, text, block_size)
    if max_blocks:
        blocks = blocks[:max_blocks]
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for block in blocks:
            ids = torch.tensor([block], device=device)
            out = model(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
            n_tokens = len(block) - 1
            total_nll += float(out.loss) * n_tokens
            total_tokens += n_tokens
    nll = total_nll / max(1, total_tokens)
    return {
        "perplexity": float(torch.exp(torch.tensor(nll))),
        "mean_nll": nll,
        "tokens_scored": total_tokens,
        "blocks": len(blocks),
        "block_size": block_size,
        "slice_lines": n_lines,
        "dataset": f"{WIKITEXT_DATASET}/{WIKITEXT_CONFIG} (test split, first {n_lines} non-empty lines)",
    }


def _option_logprob(model, tokenizer, prompt: str, option: str) -> float:
    """Mean log-probability of ``option`` tokens given ``prompt``."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    option_ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + option_ids], device=next(model.parameters()).device)
    labels = torch.full_like(ids, -100)
    labels[0, len(prompt_ids) :] = ids[0, len(prompt_ids) :]
    out = model(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids))
    n_tokens = max(1, len(option_ids))
    return -float(out.loss) * (ids.shape[1] - 1) / n_tokens


def arc_easy_accuracy(
    model, tokenizer, cfg: Config, max_items: int | None = None
) -> dict[str, Any]:
    """ARC-Easy accuracy by choosing the highest mean-log-prob answer option."""
    from datasets import load_dataset

    max_items = max_items or cfg.eval.arc_subset
    ds = load_dataset(ARC_DATASET, ARC_CONFIG, split="test")
    correct = 0
    scored = 0
    skipped = 0
    model.eval()
    with torch.inference_mode():
        for row in ds.select(range(min(len(ds), max_items * 2))):
            if scored >= max_items:
                break
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            if row["answerKey"] not in labels:
                skipped += 1
                continue
            prompt = ARC_PROMPT.format(question=row["question"].strip())
            scores = [_option_logprob(model, tokenizer, prompt, t) for t in texts]
            best = labels[int(max(range(len(scores)), key=lambda i: scores[i]))]
            correct += int(best == row["answerKey"])
            scored += 1
    return {
        "accuracy": correct / scored if scored else float("nan"),
        "n": scored,
        "correct": correct,
        "skipped": skipped,
        "method": "mean log-likelihood over answer options (hand-written scoring)",
        "dataset": f"{ARC_DATASET}/{ARC_CONFIG} (test split, first {scored} scored items)",
    }


def forgetting_report(model, tokenizer, cfg: Config, label: str) -> dict[str, Any]:
    """Run both forgetting checks for one model state."""
    log.info("forgetting checks for %s", label)
    ppl = wikitext_perplexity(model, tokenizer, cfg)
    arc = arc_easy_accuracy(model, tokenizer, cfg)
    return {"label": label, "wikitext": ppl, "arc_easy": arc}


def _direction(after: float, before: float, higher_is_better: bool, tol: float = 1e-9) -> str:
    """Honest direction label; a change inside ``tol`` is reported as unchanged."""
    if abs(after - before) <= tol:
        return "unchanged"
    improved = after > before if higher_is_better else after < before
    return "improved" if improved else "degraded"


def forgetting_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Before/after comparison with the honest direction of each change."""
    ppl_before = before["wikitext"]["perplexity"]
    ppl_after = after["wikitext"]["perplexity"]
    acc_before = before["arc_easy"]["accuracy"]
    acc_after = after["arc_easy"]["accuracy"]
    return {
        "wikitext_perplexity": {
            "before": ppl_before,
            "after": ppl_after,
            "delta": ppl_after - ppl_before,
            "relative_change": (ppl_after - ppl_before) / ppl_before if ppl_before else None,
            "direction": _direction(ppl_after, ppl_before, higher_is_better=False),
        },
        "arc_easy_accuracy": {
            "before": acc_before,
            "after": acc_after,
            "delta": acc_after - acc_before,
            "direction": _direction(acc_after, acc_before, higher_is_better=True),
            "n": before["arc_easy"]["n"],
        },
    }
