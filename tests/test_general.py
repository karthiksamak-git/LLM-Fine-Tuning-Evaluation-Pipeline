"""Forgetting-check scoring primitives (network-dependent parts skip gracefully)."""

from __future__ import annotations

import pytest
import torch

from mlfte.config import load_config
from mlfte.evaluation.general import _blocks, _option_logprob, forgetting_delta


def test_blocks_chunking(tokenizer):
    text = "select name from head where age > 10 " * 40
    blocks = _blocks(tokenizer, text, block_size=32)
    assert blocks
    assert all(len(b) == 32 for b in blocks)


def test_option_logprob_is_finite_negative(tokenizer, tiny_model):
    score = _option_logprob(tiny_model, tokenizer, "Question: 2+2?\nAnswer:", "4")
    assert torch.isfinite(torch.tensor(score))
    assert score < 0


def test_forgetting_delta_reports_direction():
    before = {"wikitext": {"perplexity": 50.0}, "arc_easy": {"accuracy": 0.25, "n": 100}}
    after = {"wikitext": {"perplexity": 60.0}, "arc_easy": {"accuracy": 0.20, "n": 100}}
    delta = forgetting_delta(before, after)
    assert delta["wikitext_perplexity"]["direction"] == "degraded"
    assert delta["arc_easy_accuracy"]["direction"] == "degraded"
    assert delta["arc_easy_accuracy"]["delta"] == pytest.approx(-0.05)


def test_forgetting_delta_detects_improvement():
    before = {"wikitext": {"perplexity": 60.0}, "arc_easy": {"accuracy": 0.2, "n": 10}}
    after = {"wikitext": {"perplexity": 50.0}, "arc_easy": {"accuracy": 0.3, "n": 10}}
    delta = forgetting_delta(before, after)
    assert delta["wikitext_perplexity"]["direction"] == "improved"
    assert delta["arc_easy_accuracy"]["direction"] == "improved"


@pytest.mark.slow
def test_wikitext_perplexity_runs_on_a_tiny_slice(tokenizer, tiny_model):
    from mlfte.evaluation.general import wikitext_perplexity

    cfg = load_config(profile="smoke")
    cfg.eval.wikitext_slice = 25
    try:
        result = wikitext_perplexity(tiny_model, tokenizer, cfg, block_size=32, max_blocks=1)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"dataset unavailable: {exc}")
    assert result["perplexity"] > 0
    assert result["tokens_scored"] > 0
